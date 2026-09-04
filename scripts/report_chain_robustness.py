"""Rank what breaks the chain, from the sweep's own per-episode rows.

``scripts/sweep_chain_robustness.sh`` runs one point per variable around the
certified configuration. This reads the rows those runs wrote, puts a Wilson
interval on each, and orders them by how far each point falls below the nominal
one -- so "the chain is brittle" becomes a list with numbers on it.

Sixteen episodes a point is deliberately coarse. The interval is reported at
every point so nothing here can be read as a precise rate, and the ranking is
the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zero_g_blade_swap.evaluation import wilson_interval  # noqa: E402


def _load(path: Path) -> tuple[float, int, dict[str, float]]:
    data = np.load(path, allow_pickle=True)
    rows = data["rows"]
    fields = [str(name) for name in data["fields"]]
    index = {name: position for position, name in enumerate(fields)}
    success = rows[:, index["success"]] > 0.5
    detail = {
        name: float(np.median(rows[:, index[name]]))
        for name in ("axial_error_m", "lateral_error_m", "orientation_error_rad")
        if name in index
    }
    return float(success.mean()), int(len(rows)), detail


from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402


def _refuse_a_dirty_worktree(revision: dict) -> None:
    """Fail before writing evidence that cannot be reproduced.

    Ten older reports in this repository record source hashes that match no
    commit, because they were produced from uncommitted code. The rule that
    replaced them is that an evidence generator refuses a dirty tracked
    worktree, and a generator without that rule is how the next ten happen.
    """

    if revision.get("available") and revision.get("dirty"):
        changed = ", ".join(str(line) for line in revision.get("tracked_changes", [])[:5])
        raise SystemExit(
            "refusing to write evidence from a dirty tracked worktree; commit first. " + changed
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, default=PROJECT_ROOT / "artifacts" / "robustness")
    parser.add_argument("--nominal", default="nominal")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--note",
        action="append",
        default=None,
        help=(
            "A line recorded in the report beside the points. It exists because a sweep "
            "directory can legitimately hold points measured under different arms -- the "
            "clearance re-measurement replaces two of seven -- and a reader cannot tell that "
            "from the numbers. Repeatable."
        ),
    )
    arguments = parser.parse_args()

    _revision = git_source_revision(PROJECT_ROOT)
    _refuse_a_dirty_worktree(_revision)
    points: dict[str, dict[str, object]] = {}
    for path in sorted(arguments.sweep_dir.glob("*.npz")):
        rate, episodes, detail = _load(path)
        low, high = wilson_interval(int(round(rate * episodes)), episodes)
        points[path.stem] = {
            "success_rate": round(rate, 6),
            "episodes": episodes,
            "wilson_95": {"low": round(low, 6), "high": round(high, 6)},
            "median_terminal": {name: round(value, 6) for name, value in detail.items()},
        }
    if arguments.nominal not in points:
        print(f"no {arguments.nominal} point in {arguments.sweep_dir}; nothing to rank against")
        return 1

    baseline = float(points[arguments.nominal]["success_rate"])
    for entry in points.values():
        entry["points_below_nominal"] = round(100.0 * (baseline - float(entry["success_rate"])), 2)
    ranked = sorted(
        (name for name in points if name != arguments.nominal),
        key=lambda name: -float(points[name]["points_below_nominal"]),
    )

    print(f"{'point':>22} {'episodes':>9} {'success':>9} {'wilson 95':>18} {'vs nominal':>11}")
    for name in [arguments.nominal, *ranked]:
        entry = points[name]
        interval = entry["wilson_95"]
        print(
            f"{name:>22} {entry['episodes']:9d} {entry['success_rate'] * 100:8.2f}% "
            f"[{interval['low'] * 100:5.1f}, {interval['high'] * 100:5.1f}]".rjust(18)
            + f"{entry['points_below_nominal']:11.2f}"
        )

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(
                {
                    "title": "What breaks the robot-carried chain first",
                    "evidence_type": "simulation_only",
                    "generated_utc": datetime.now(UTC).isoformat(),
                    "question": (
                        "The chain is certified at one point in a space a real cell varies across. "
                        "Which variable takes it down soonest?"
                    ),
                    "method": (
                        "scripts/sweep_chain_robustness.sh: one variable moved at a time around the "
                        "certified configuration, same seed, same checkpoints, same everything else."
                    ),
                    "source_revision": _revision,
                    "notes": list(arguments.note or []),
                    "nominal_point": arguments.nominal,
                    "nominal_success_rate": baseline,
                    "ranked_by_points_below_nominal": ranked,
                    "points": points,
                    "scope_and_limitations": (
                        "Sixteen episodes a point on one seed. The Wilson interval on each is about "
                        "twenty points wide, so this ranks variables and does not measure any of them. "
                        "Whatever it ranks first is what deserves the 96-episode protocol."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
