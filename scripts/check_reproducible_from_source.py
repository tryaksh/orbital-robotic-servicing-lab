#!/usr/bin/env python3
"""Which published reports can this repository still rebuild from its own source?

Thirty-seven of the sixty canonical reports predate the ``source_revision``
field, so the manifest cannot say whether they are reproducible -- that is the
provenance gap the project has carried as T0. For the reports whose generator
needs no simulator and no checkpoint, the question does not need a recorded
commit to answer. Run the generator and compare.

That is what this does. Each named generator is run, its report is compared with
the committed copy, and the working tree is restored either way. Differences that
are only a regenerated timestamp are reported separately from differences in the
numbers, because they mean opposite things: a timestamp-only difference is a
clean reproduction, and any other difference means the code and the published
report have diverged.

**This writes to ``evidence/`` and then restores it.** It refuses to run unless
that directory is clean in git, because restoring is `git checkout -- evidence`
and doing that over uncommitted work would destroy it. Nothing here runs on the
GPU; a generator that needs one is not in the list.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: ``generator: reports it writes``. Only CPU-only generators belong here. A
#: generator that loads a checkpoint or steps the simulator cannot be rerun as a
#: check, and its report's provenance has to come from the recorded commit.
GENERATORS: dict[str, tuple[str, ...]] = {
    "check_rack_sightlines.py": ("rack_sightline_datum_pair_v1.json", "rack_sightline_occlusion_v1.json"),
    "report_insert_depth_limit.py": ("insert_depth_is_attitude.json",),
    "derive_rack_requirement.py": (),
    "check_service_latch_clearance.py": ("service_latch_clearance.json",),
    "check_servicing_camera_geometry.py": (),
}

#: Keys whose value is expected to change on every run and whose change alone
#: does not mean the numbers moved.
VOLATILE = ("generated_utc", "generated", "timestamp")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout


def _strip_volatile(text: str) -> str:
    try:
        payload = json.loads(text)
    except ValueError:
        return text
    if isinstance(payload, dict):
        for key in VOLATILE:
            payload.pop(key, None)
    return json.dumps(payload, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if _git("status", "--porcelain", "evidence").strip():
        raise SystemExit(
            "evidence/ has uncommitted changes. This check restores the directory with "
            "`git checkout -- evidence`, which would destroy them. Commit or stash first."
        )

    before = {
        path.name: path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "evidence").glob("*.json")
    }

    identical: list[str] = []
    timestamp_only: list[str] = []
    diverged: list[str] = []
    unrunnable: list[str] = []

    python = ROOT / ".venv" / "Scripts" / "python.exe"
    interpreter = str(python) if python.exists() else sys.executable

    try:
        for generator, reports in GENERATORS.items():
            script = ROOT / "scripts" / generator
            if not script.exists():
                unrunnable.append(f"{generator} (missing)")
                continue
            outcome = subprocess.run(
                [interpreter, str(script)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            if outcome.returncode != 0:
                unrunnable.append(f"{generator} (exit {outcome.returncode})")
                continue
            for report in reports:
                path = ROOT / "evidence" / report
                if not path.exists() or report not in before:
                    unrunnable.append(f"{report} (not written)")
                    continue
                now = path.read_text(encoding="utf-8", errors="replace")
                if now == before[report]:
                    identical.append(report)
                elif _strip_volatile(now) == _strip_volatile(before[report]):
                    timestamp_only.append(report)
                else:
                    diverged.append(report)
    finally:
        _git("checkout", "--", "evidence")

    print(f"  {len(identical)} byte-identical: {', '.join(sorted(identical)) or '--'}")
    print(f"  {len(timestamp_only)} identical but for the timestamp: {', '.join(sorted(timestamp_only)) or '--'}")
    print(f"  {len(diverged)} DIVERGED: {', '.join(sorted(diverged)) or '--'}")
    if unrunnable:
        print(f"  {len(unrunnable)} not checked: {', '.join(unrunnable)}")
    print("\n  evidence/ restored.")

    if diverged:
        print("\n  A diverged report and the source that claims to produce it disagree.")
        print("  Either the code moved without the report being regenerated, or the report")
        print("  was not produced by the generator named here. Both are worth knowing.")
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
