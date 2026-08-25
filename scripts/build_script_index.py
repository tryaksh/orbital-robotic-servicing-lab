"""Write ``scripts/README.md``: what each of the seventy scripts here is for.

``scripts/`` is flat and large. An agent looking for "how do I check the
geometry" or "which script produced this evidence file" currently has to list the
directory and guess from names, or open several to find out -- which is exactly
the kind of avoidable reading this repository is trying to stop paying for.

The summary for each script is its **own first documentation line**: the first
line of the module docstring for Python, the first non-shebang comment for shell
and PowerShell. Nothing is written by hand here, so the index cannot drift into
describing a script differently from how the script describes itself. If an entry
reads badly, fix the script's docstring and regenerate.

Grouping is by filename prefix, which this repository already uses as a verb:
``check_`` proves something without a simulator, ``certify_`` produces a gated
report, ``run_`` drives a workflow, ``report_``/``analyse_`` turn rows into a
finding, ``measure_``/``sweep_``/``solve_`` produce a number.

CPU only. Reads files; imports nothing from Isaac Lab.

Usage::

    python scripts/build_script_index.py           # write the index
    python scripts/build_script_index.py --check   # fail if it is stale
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
INDEX = SCRIPTS / "README.md"

#: Prefix -> (heading, what this family of scripts is for). Order is the order
#: the sections appear in, which is roughly the order a newcomer needs them.
GROUPS: tuple[tuple[str, str, str], ...] = (
    (
        "check_",
        "Checks — prove something, mostly without a simulator",
        "These are the ones to run first and the ones that run in CI. A requirement "
        "only a GPU can check is a requirement that stops being checked.",
    ),
    (
        "run_",
        "Runners — drive a workflow or a training job",
        "One stage per question. `run_robot_carried.sh` is the chain.",
    ),
    (
        "certify_",
        "Certifications — produce a gated report under `evidence/`",
        "Held-out seeds, pooled, with a Wilson interval and a pass/fail gate.",
    ),
    (
        "report_",
        "Reports — turn recorded rows into a finding",
        "These read `.npz` episode metrics and write `evidence/*.json`.",
    ),
    (
        "analyse_",
        "Analyses — diagnose a specific failure",
        "Written for one question each; kept because the question recurs.",
    ),
    (
        "measure_",
        "Measurements — produce a physical number",
        "Envelopes, budgets and design windows.",
    ),
    (
        "solve_",
        "Solvers — compute a configuration in closed form",
        "Deterministic, gated, and re-runnable without a simulator.",
    ),
    (
        "sweep_",
        "Sweeps — one variable at a time around a certified point",
        "Coarse on purpose: these rank variables rather than measure them.",
    ),
    (
        "build_",
        "Builders — regenerate a tracked artifact",
        "Each takes `--check` so CI can prove the artifact is current.",
    ),
    (
        "collect_",
        "Collectors — record a dataset",
        "",
    ),
    (
        "train",
        "Training and evaluation entry points",
        "`train.py` and `play.py` are the two the rest wrap.",
    ),
)


def summary(path: Path) -> str:
    """The script's own first documentation line."""
    try:
        # utf-8-sig, because two of these files carry a BOM and ``ast.parse``
        # raises SyntaxError on it -- which silently produced an empty summary
        # for train.py and play.py, the two scripts everything else wraps.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return ""
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(text))
        except SyntaxError:
            doc = None
        if doc:
            return doc.strip().splitlines()[0].strip()
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!") or not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        break
    return ""


def _classify(names: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    buckets: dict[str, list[str]] = {prefix: [] for prefix, _, _ in GROUPS}
    rest: list[str] = []
    for name in names:
        for prefix, _, _ in GROUPS:
            if name.startswith(prefix):
                buckets[prefix].append(name)
                break
        else:
            rest.append(name)
    return buckets, rest


def build() -> str:
    names = sorted(
        path.name
        for path in SCRIPTS.iterdir()
        if path.is_file() and path.suffix in {".py", ".sh", ".ps1"} and path.name != "README.md"
    )
    buckets, rest = _classify(names)

    lines = [
        "# Scripts",
        "",
        "**Generated by `scripts/build_script_index.py`. Do not hand-edit** — each",
        "summary is the script's own first documentation line, so fix the script's",
        "docstring and regenerate.",
        "",
        f"{len(names)} scripts. Grouped by the verb their filename starts with.",
        "",
        "The few that matter most are in [`../AGENTS.md`](../AGENTS.md); this is the",
        "complete list, for when the one you need is not there.",
        "",
    ]
    for prefix, heading, blurb in GROUPS:
        entries = buckets[prefix]
        if not entries:
            continue
        lines += [f"## {heading}", ""]
        if blurb:
            lines += [blurb, ""]
        lines += ["| Script | What it is |", "| --- | --- |"]
        for name in entries:
            lines.append(f"| `{name}` | {summary(SCRIPTS / name) or '—'} |")
        lines.append("")
    if rest:
        lines += [
            "## Everything else",
            "",
            "| Script | What it is |",
            "| --- | --- |",
        ]
        for name in rest:
            lines.append(f"| `{name}` | {summary(SCRIPTS / name) or '—'} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify scripts/README.md.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if the index is stale.")
    args = parser.parse_args()

    rendered = build()
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.is_file() else ""
        if current != rendered:
            print("scripts/README.md is stale. Run: python scripts/build_script_index.py")
            return 1
        print("scripts/README.md is current")
        return 0

    INDEX.write_text(rendered, encoding="utf-8")
    print(f"wrote {INDEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
