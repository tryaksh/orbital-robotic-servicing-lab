"""Fail if an evidence report predates the code that defines what it measured.

``check_evidence_currency.py`` answers "is this number about this *policy*". This
answers the other half, and it is the half that has now gone wrong four times:

* extraction's 68.36% was certified an hour before the settled-enough velocity
  limits were derived, and not one of its 6,156 counted successes satisfies them;
* the removal chain's 14.06% shared that defect;
* installation's 84.38% was certified 8.5 hours before the capture phase's budget
  went from 6 s to 10 s;
* capture's 96.10% was certified 9.4 hours before its own success tolerance went
  from 20 mm to 10 mm.

Every one was found by hand, late, by someone remembering to compare a timestamp
against ``git log``. Nothing ran automatically, so nothing caught them at the
time.

**A check that fires on everything is worse than no check**, because it gets
ignored, and the first version of this script did exactly that -- any edit to any
task configuration flagged all forty reports. So each report is compared only
against the files that can define *its* criterion: the configuration module its
own task is registered from, the two MDP modules that hold every predicate, the
evaluator, and -- only for chained runs -- the workflow driver, which owns the
phase budgets and the settling re-check.

A flag means "re-check this", not "this is wrong". Confirm with ``git log -S`` on
the constant the report actually depends on, which is what the rule asks for.

CPU only. It reads JSON and ``git log``; it imports nothing from Isaac Lab and
loads no checkpoints, so it runs while the GPU is busy.

Usage::

    python scripts/check_criterion_currency.py                  # every report
    python scripts/check_criterion_currency.py evidence/a.json  # named reports
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
PACKAGE = "src/zero_g_blade_swap/tasks/blade_swap"
REGISTRY = ROOT / PACKAGE / "__init__.py"

#: Files every report depends on whatever task produced it: the two MDP modules
#: that hold every success and failure predicate in this project, and the
#: evaluator that decides what a success is counted as.
ALWAYS = (
    f"{PACKAGE}/mdp/grapple.py",
    f"{PACKAGE}/mdp/insertion.py",
    "src/zero_g_blade_swap/evaluation.py",
)
#: Chained runs are driven by this rather than by ``play.py``, and it owns the
#: per-phase budgets and the settling re-check that decide their outcome.
DRIVER = "scripts/run_workflow_demo.py"
CHAINED = ("Workflow", "Vision")


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def task_configuration_modules() -> dict[str, str]:
    """Map each registered task id to the module it is configured from.

    Read out of the registry source rather than guessed, and rather than
    imported: importing it needs a simulator. Registrations come in two shapes --
    a direct ``gym.register`` and a loop over ``(id, class)`` pairs -- and both
    name their configuration module in an ``env_cfg_entry_point`` f-string.
    """

    source = REGISTRY.read_text(encoding="utf-8")
    modules: dict[str, str] = {}
    # Every task id in the file, in order, paired with the next entry point that
    # follows it. For the loop form the entry point sits after the whole tuple
    # list, so ids collected since the last entry point all share it.
    pending: list[str] = []
    for token in re.finditer(
        r'"(Isaac-[A-Za-z0-9\-]+)"|env_cfg_entry_point["\']?\s*[:=]\s*\(?\s*f"\{__name__\}\.([a-z_]+):',
        source,
    ):
        task_id, module = token.group(1), token.group(2)
        if task_id:
            pending.append(task_id)
        elif module:
            for identifier in pending:
                modules[identifier] = f"{PACKAGE}/{module}.py"
            pending = []
    return modules


def last_commit(relative: str) -> tuple[datetime, str] | None:
    recorded = _git("log", "-1", "--format=%cI%x09%s", "--", relative)
    if not recorded:
        return None
    stamp, _, subject = recorded.partition("\t")
    return datetime.fromisoformat(stamp), subject


def relevant_files(tasks: list[str], modules: dict[str, str]) -> list[str]:
    files = set(ALWAYS)
    for task in tasks:
        module = modules.get(task)
        if module:
            files.add(module)
        if any(label in task for label in CHAINED):
            files.add(DRIVER)
    return sorted(files)


#: The handover table, and only the handover table.
#:
#: This is what turns the check from a list into a gate, and the narrower choice
#: is the right one. ``docs/status.md`` cites almost every report in
#: ``evidence/`` *on purpose* -- it is the record, and it keeps retracted and
#: superseded results deliberately -- so gating on it flags sixty reports and
#: means nothing. ``CLAUDE.md`` cites the handful of numbers this project asserts
#: are **current**, which is exactly the set where staleness misleads.
PUBLISHING_DOCUMENTS = ("CLAUDE.md",)


def published_reports() -> set[str]:
    """Evidence filenames the handover asserts as the current state."""

    cited: set[str] = set()
    for relative in PUBLISHING_DOCUMENTS:
        path = ROOT / relative
        if not path.is_file():
            continue
        cited.update(re.findall(r"([A-Za-z0-9_]+\.json)", path.read_text(encoding="utf-8")))
    return cited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Print only what needs re-checking.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check every report, not only the ones the published documents cite.",
    )
    args = parser.parse_args()

    if args.reports:
        reports = args.reports
    elif args.all:
        reports = sorted(EVIDENCE.glob("*.json"))
    else:
        cited = published_reports()
        reports = sorted(path for path in EVIDENCE.glob("*.json") if path.name in cited)
        print(f"Checking the {len(reports)} reports CLAUDE.md asserts as current. --all for the rest.\n")
    modules = task_configuration_modules()
    if not modules:
        print("Could not read any task registrations; is this a git checkout of the package?")
        return 1

    stale = 0
    checked = 0
    for report in reports:
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stamp = payload.get("generated_utc")
        tasks = payload.get("protocol", {}).get("tasks") or []
        if not isinstance(stamp, str) or not tasks:
            continue
        try:
            generated = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        checked += 1

        newer = []
        for relative in relevant_files(list(tasks), modules):
            commit = last_commit(relative)
            if commit and commit[0] > generated:
                newer.append((commit[0], relative, commit[1]))
        if not newer:
            if not args.quiet:
                print(f"CURRENT  {report.name}")
            continue

        stale += 1
        print(f"RE-CHECK {report.name}  generated {generated:%Y-%m-%d %H:%M %z}")
        for when, relative, subject in sorted(newer, reverse=True):
            # Not every flag carries the same weight, and saying so is what makes
            # the list actionable. The MDP modules are large and shared, so a
            # commit touching one is weak evidence that *this* report's criterion
            # moved. The report's own task configuration and the workflow driver
            # are narrow: they hold episode budgets, reward and termination sets,
            # phase budgets and the settling re-check, so a commit there is a
            # strong reason to look.
            weight = "weak  " if "/mdp/" in relative or relative.endswith("evaluation.py") else "STRONG"
            print(f"         [{weight}] {when:%Y-%m-%d %H:%M %z}  {relative}")
            print(f"                    {subject}")

    print(f"\n{stale} of {checked} reports predate a change to a file that can define their criterion.")
    if stale:
        print(
            "A prompt, not a verdict, and STRONG lines first: confirm with `git log -S` on the constant "
            "each report actually depends on, and re-run the ones it moved. This project changes its "
            "criteria often enough that a long list here is the honest answer rather than a broken check."
        )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
