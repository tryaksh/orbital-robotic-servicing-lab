#!/usr/bin/env python3
"""Pool a sweep point measured at several seeds into one entry, at the full n.

The boundary validator compares each design point against nominal and calls a
loss separated only when the Wilson intervals do not overlap. At 64 episodes
those intervals are about twenty points wide, so a point can move in exactly the
direction its criterion predicts and still not separate. That is what happened to
the module-section axis: it moved the right way at 64 and cleared nothing, and
three seeds at 192 episodes separated it decisively.

Those three seeds exist only as episode archives. This turns them into a sweep
report the validator can read, so the paper's central table can be regenerated at
the sample size that actually answers the question.

Two rules keep the substitution honest:

* Statistics come from ``zero_g_blade_swap.evaluation.wilson_interval``, the same
  function every certification uses, rather than a second implementation here.
* Every pooled point records how many episodes and which seeds it came from, and
  the merged report says in its own notes which points were replaced and which
  were left at the original sample. A reader must never have to guess whether two
  rows in one table were measured at the same n.

Nothing is recomputed from the simulator and no tolerance moves. This reads
archives that already exist.

Example::

    python scripts/pool_sweep_points.py \\
        --base evidence/chain_robustness_sweep_n64_channel_v1.json \\
        --dir artifacts/robustness64_corrected \\
        --dir artifacts/robustness64_seed5070 \\
        --dir artifacts/robustness64_seed6070 \\
        --point nominal --point section_120x16 --point section_140x26 \\
        --output evidence/chain_robustness_sweep_section_n192_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zero_g_blade_swap.evaluation import wilson_interval  # noqa: E402
from src.zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

#: Terminal quantities the sweep report carries for every point.
TERMINAL_FIELDS = ("axial_error_m", "lateral_error_m", "orientation_error_rad")


def _load(path: Path) -> tuple[np.ndarray, list[str], dict]:
    archive = np.load(path, allow_pickle=True)
    metadata = {}
    if "metadata" in archive:
        try:
            metadata = json.loads(str(archive["metadata"]))
        except (ValueError, TypeError):
            metadata = {}
    return archive["rows"], [str(f) for f in archive["fields"]], metadata


def pool_point(directories: list[Path], tag: str) -> dict:
    """One sweep-report entry from the same point measured in several places."""

    stacked: list[np.ndarray] = []
    fields: list[str] | None = None
    seeds: list[int] = []
    for directory in directories:
        path = directory / f"{tag}.npz"
        if not path.exists():
            raise SystemExit(f"{path} does not exist; refusing to pool a point that is missing a seed")
        rows, point_fields, metadata = _load(path)
        if fields is None:
            fields = point_fields
        elif point_fields != fields:
            raise SystemExit(f"{path} has different columns from the first archive; refusing to pool")
        stacked.append(rows)
        seed = metadata.get("seed")
        if seed is None:
            report = directory / f"{tag}_report.json"
            if report.exists():
                seed = json.loads(report.read_text(encoding="utf-8")).get("seed")
        seeds.append(int(seed) if seed is not None else -1)

    assert fields is not None
    rows = np.concatenate(stacked, axis=0)
    success = rows[:, fields.index("success")]
    successes = int(success.sum())
    episodes = int(rows.shape[0])
    low, high = wilson_interval(successes, episodes)

    entry: dict = {
        "success_rate": round(successes / episodes, 6),
        "successes": successes,
        "episodes": episodes,
        "wilson_95": {"low": round(low, 6), "high": round(high, 6)},
        "median_terminal": {
            name: round(float(np.median(rows[:, fields.index(name)])), 6)
            for name in TERMINAL_FIELDS
            if name in fields
        },
        "pooled_from": {
            "seeds": seeds,
            "episodes_each": [int(part.shape[0]) for part in stacked],
            "directories": [str(d) for d in directories],
        },
    }
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="sweep report to merge the pooled points into")
    parser.add_argument("--dir", type=Path, action="append", required=True, dest="dirs")
    parser.add_argument("--point", action="append", required=True, dest="points")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pooled_dir",
        type=Path,
        help=(
            "Also write the concatenated episode archives here, one per pooled point. "
            "report_boundary_failure_modes.py reads a single sweep directory, and this "
            "lets it score the pooled cohort without teaching it about pooling -- the "
            "script that produced published evidence is left exactly as it was."
        ),
    )
    args = parser.parse_args()

    if len(args.dirs) < 2:
        raise SystemExit("pooling one directory is not pooling; pass --dir at least twice")

    base = json.loads(args.base.read_text(encoding="utf-8"))
    points = dict(base.get("points", {}))

    pooled: dict[str, dict] = {}
    for tag in args.points:
        pooled[tag] = pool_point(args.dirs, tag)
        points[tag] = pooled[tag]

    if "nominal" not in pooled:
        raise SystemExit(
            "nominal was not pooled. Every point in this report is read against nominal, "
            "so comparing a pooled point with a nominal measured at a smaller n would put "
            "the sample-size difference into the comparison itself."
        )

    untouched = sorted(set(points) - set(pooled))
    report = dict(base)
    report["title"] = base.get("title", "Chain robustness sweep") + ", module-section axis pooled over three seeds"
    report["generated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["source_revision"] = git_source_revision(ROOT)
    report["points"] = points
    report["nominal_success_rate"] = pooled["nominal"]["success_rate"]
    report["pooling"] = {
        "pooled_points": sorted(pooled),
        "pooled_episodes": pooled["nominal"]["episodes"],
        "left_at_original_sample": untouched,
        "base_report": str(args.base),
        "why": (
            "At 64 episodes a Wilson interval on this chain is about twenty points wide, so a "
            "point can move exactly as its criterion predicts and still not separate from "
            "nominal. The module-section axis did that. These points are the same "
            "configuration measured at three held-out seeds and pooled, so the comparison "
            "against nominal is answered at 192 episodes."
        ),
        "read_this_before_comparing_rows": (
            "Points listed in left_at_original_sample are still at the base report's sample "
            "size. Do not compare a pooled point's interval width with theirs and conclude "
            "anything about the design; only the pooled points and pooled nominal are "
            "measured at the larger n."
        ),
    }
    notes = list(report.get("notes", []) if isinstance(report.get("notes"), list) else [])
    notes.append(
        f"Points {', '.join(sorted(pooled))} pooled over {len(args.dirs)} seeds to "
        f"{pooled['nominal']['episodes']} episodes each; all other points unchanged from {args.base.name}."
    )
    report["notes"] = notes

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")

    if args.pooled_dir:
        args.pooled_dir.mkdir(parents=True, exist_ok=True)
        for tag in args.points:
            stacked: list[np.ndarray] = []
            fields: list[str] | None = None
            for directory in args.dirs:
                rows, point_fields, _ = _load(directory / f"{tag}.npz")
                fields = fields or point_fields
                stacked.append(rows)
            assert fields is not None
            metadata = json.dumps(
                {
                    "pooled": True,
                    "point": tag,
                    "seeds": pooled[tag]["pooled_from"]["seeds"],
                    "directories": pooled[tag]["pooled_from"]["directories"],
                    "note": "Concatenated episode rows, not a simulator output. See scripts/pool_sweep_points.py.",
                }
            )
            np.savez(
                args.pooled_dir / f"{tag}.npz",
                rows=np.concatenate(stacked, axis=0),
                fields=np.array(fields),
                metadata=np.array(metadata),
            )
        print(f"  pooled archives written to {args.pooled_dir}")

    for tag in sorted(pooled):
        entry = pooled[tag]
        interval = entry["wilson_95"]
        print(
            f"  {tag:18s} {entry['successes']:3d}/{entry['episodes']:<4d} "
            f"{entry['success_rate']:.4f}  [{interval['low']:.4f}, {interval['high']:.4f}]"
        )
    if untouched:
        print(f"  left at the base sample: {', '.join(untouched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
