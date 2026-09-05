#!/usr/bin/env python3
"""What the destination channel's relief is actually for.

The design library calls the shipped channel 3.897 mm per side too wide and
prescribes removing the relief. Removing it costs 12 episodes in 192 once the
retention pawls are fitted, and those 12 fail in a way the shipped relief never
produces: they reach the insert phase, the insertion predicate never fires, and
the episode ends with the module stopped short of the seated plane.

Reading the insert trace says why, and it is not a lateral story.

**Every insertion swings.** In both configurations the module's orientation
error peaks around 76 mrad partway along the stroke -- median 75.67 mrad with
the relief and 75.69 mrad without -- so the swing itself is the normal motion of
this interface and the channel accommodates it. What differs is the *tail*. With
the relief the peak swing is tightly bounded: p95 76.85 mrad, maximum 79.52. At
the prescribed clearance the tail runs out to 95.08 mrad, and the episodes that
wedge are the ones that get there: 90.1, 90.6 and 86.9 mrad at seed 4070.

They all stop at the same place, **x = 0.220 m**, against a seated plane at
0.676 m -- about ninety millimetres into a stroke that should run half a metre.
The guard reads `clear_to_advance = 0` and holds, which is the guard doing its
job rather than forcing a jammed module.

So the relief is not slack to be designed out. It is what bounds the entry swing
below the angle at which the module wedges, and the library's upper bound --
derived from where a *seated* module may rest -- does not see that function at
all. The lateral error those episodes report, 32.0 to 32.9 mm, is a consequence
of the rotation and not an independent translation.

    python scripts/report_entry_swing.py --report evidence/entry_swing_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

#: The seated plane the guarded advance drives to, from the workflow driver.
SEATED_PLANE_X_M = 0.676
#: An episode whose module never gets within this of the seated plane did not
#: complete the stroke. Set well clear of the observed jam at 0.220 m and of the
#: observed completions at 0.675-0.677 m, so nothing lands near the boundary.
STROKE_COMPLETE_X_M = 0.60

ARMS = {
    "relief 4.61 mm (shipped), pawls": "artifacts/retention_nominal/seed{seed}/nominal_trace.npz",
    "relief 0.00 mm (prescribed), pawls": "artifacts/prescription_retained/seed{seed}/nominal_trace.npz",
    "relief 4.61 mm (shipped), no pawls": "artifacts/traced_nominal/seed{seed}/nominal_trace.npz",
}
SEEDS = (4070, 5070, 6070)


def profile(paths: list[Path]) -> dict | None:
    peaks: list[float] = []
    finals: list[float] = []
    for path in paths:
        if not path.exists():
            return None
        trace = np.load(path, allow_pickle=True)
        fields = [str(name) for name in trace["insert_fields"]]
        block = trace["insert"]
        if block.size == 0:
            continue
        for env in np.unique(block[:, fields.index("env")]).astype(int):
            rows = block[block[:, fields.index("env")] == env]
            rows = rows[np.argsort(rows[:, fields.index("step")])]
            peaks.append(float(rows[:, fields.index("orientation_error_rad")].max()))
            finals.append(float(rows[-1, fields.index("true_blade_x_m")]))
    if not peaks:
        return None
    peak = np.array(peaks)
    final = np.array(finals)
    short = final < STROKE_COMPLETE_X_M
    return {
        "episodes": len(peak),
        "peak_swing_median_rad": float(np.median(peak)),
        "peak_swing_p95_rad": float(np.percentile(peak, 95)),
        "peak_swing_max_rad": float(peak.max()),
        "stopped_short_of_seated_plane": int(short.sum()),
        "stopped_at_x_m": sorted(round(float(v), 4) for v in final[short]),
        "peak_swing_of_stopped_rad": sorted(round(float(v), 5) for v in peak[short]),
        "final_x_median_m": float(np.median(final)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    arms: dict[str, dict] = {}
    for label, template in ARMS.items():
        result = profile([ROOT / template.format(seed=s) for s in SEEDS])
        if result is None:
            print(f"{label}: traces missing, skipping")
            continue
        arms[label] = result

    print(
        f"{'arm':>38} {'n':>5} {'median':>8} {'p95':>8} {'max':>8} {'short':>7}"
    )
    for label, arm in arms.items():
        print(
            f"{label:>38} {arm['episodes']:>5} "
            f"{arm['peak_swing_median_rad'] * 1000:>8.2f} "
            f"{arm['peak_swing_p95_rad'] * 1000:>8.2f} "
            f"{arm['peak_swing_max_rad'] * 1000:>8.2f} "
            f"{arm['stopped_short_of_seated_plane']:>7}"
        )
    print("  (swing in mrad; 'short' = stopped before the seated plane at 0.676 m)")
    print()
    for label, arm in arms.items():
        if arm["stopped_short_of_seated_plane"]:
            print(f"{label}: stopped at x = {arm['stopped_at_x_m']} m")
            print(f"    their peak swing (mrad): {[round(v * 1000, 1) for v in arm['peak_swing_of_stopped_rad']]}")

    document = {
        "title": "The destination channel's relief bounds the entry swing",
        "evidence_type": "second_reading_of_recorded_traces",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Removing the channel relief, as the design library prescribes, costs 12 "
            "episodes in 192 once the pawls are fitted, and those episodes never seat. "
            "What does the relief do that its absence undoes?"
        ),
        "source_revision": git_source_revision(ROOT),
        "seated_plane_x_m": SEATED_PLANE_X_M,
        "stroke_complete_threshold_x_m": STROKE_COMPLETE_X_M,
        "seeds": list(SEEDS),
        "arms": arms,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "This is a second reading of existing episodes, not a new measurement.",
            "The peak swing is the maximum orientation error recorded during the insert "
            "phase at the trace's 6 Hz sampling, so a briefer excursion between samples "
            "would not appear. It is a lower bound on the true peak.",
            "Every episode in both configurations exceeds the angle sometimes quoted as "
            "the channel's corner-to-corner limit. That figure describes where a seated "
            "module may rest, not what the stroke passes through, and nothing here "
            "should be read as contradicting it.",
            "One point of the sweep -- nominal -- at three seeds, with one frozen "
            "checkpoint set. The wedge angle is not measured as a threshold; it is "
            "observed that the episodes which stop short are the ones with the largest "
            "swing.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
