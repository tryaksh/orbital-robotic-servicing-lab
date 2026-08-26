"""Why the insert skill stops short: it is attitude again, one layer down.

After the channel throat was derived from the seated criterion, the skill's
*orientation* condition stopped being the blocker -- 94.6% of episodes that reach
seated depth are now inside 52.36 mrad, against 0% before. What replaced it is
**depth**: only 35.8% of episodes reach the seated plane at all, and the rest end
a median of 174.5 mm short with the clock still running.

"Stops short with time left" reads like creeping, and this project has already
refuted creeping once. So the stalled episodes were compared against the seated
ones on every terminal quantity, and they differ on exactly one:

    stalled   96.8 mrad of attitude, 6.13 mm/s, 5.10 mm lateral
    seated    46.9 mrad of attitude, 0.69 mm/s, 2.46 mm lateral

The grip is identical on both -- 11.5 mm along the pin, the measured feed -- so
nothing is slipping. And the acceptance law closes it: a module held at ``theta``
can only engage ``2c/theta`` before it wedges, so at 96.8 mrad in this bay's
relieved channel the deepest engagement is 261 mm. **The stalled episodes are not
stopping early, they are seated as deep as their own attitude permits.**

That makes depth and attitude the same failure, and it is bimodal rather than
graded: an episode either stays near 47 mrad and drives home, or reaches ~97 mrad
and wedges partway. What separates the two is what happens in the first control
steps, when the lock softens into the mating compliance while the module is
already inside the channel.

CPU only. Reads the ``.npz`` rows an evaluation already wrote.

Usage::

    python scripts/report_insert_depth_limit.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "insert_depth_is_attitude.json"
ROWS = ROOT / "artifacts" / "insert_wall" / "v23lock_newrack.npz"

#: ``mdp.insertion.INSERTION_AXIAL_DEPTH_TOLERANCE_M``.
SEATED_DEPTH_M = 0.012
MODULE_LENGTH_M = 0.45

#: The destination bay's clearances per side *with* its relief, measured out of
#: the built configuration in ``evidence/destination_channel_geometry.json``.
RELIEVED_LATERAL_M = 0.015678
RELIEVED_VERTICAL_M = 0.012613


def _group(rows: np.ndarray, index: dict[str, int], mask: np.ndarray) -> dict[str, object]:
    def quantiles(field: str, scale: float) -> dict[str, float]:
        values = rows[mask, index[field]] * scale
        return {
            "p25": round(float(np.percentile(values, 25)), 3),
            "median": round(float(np.median(values)), 3),
            "p75": round(float(np.percentile(values, 75)), 3),
        }

    return {
        "episodes": int(mask.sum()),
        "orientation_mrad": quantiles("orientation_error_rad", 1000.0),
        "lateral_mm": quantiles("lateral_error_m", 1000.0),
        "shortfall_mm": quantiles("axial_error_m", 1000.0),
        "linear_velocity_mm_s": quantiles("blade_linear_velocity_mps", 1000.0),
        "angular_velocity_mrad_s": quantiles("blade_angular_velocity_radps", 1000.0),
        "grip_along_the_pin_mm": quantiles("grip_offset_approach_axis_m", 1000.0),
        "control_steps_median": float(np.median(rows[mask, index["control_steps"]])),
    }


def main() -> int:
    if not ROWS.is_file():
        raise SystemExit(f"missing evaluation rows: {ROWS}")
    archive = np.load(ROWS, allow_pickle=True)
    index = {str(name): position for position, name in enumerate(archive["fields"])}
    rows = archive["rows"]

    seated = rows[:, index["axial_error_m"]] <= SEATED_DEPTH_M
    stalled = ~seated
    stalled_attitude = float(np.median(rows[stalled, index["orientation_error_rad"]]))
    tightest = min(RELIEVED_LATERAL_M, RELIEVED_VERTICAL_M)
    deepest = 2.0 * tightest / stalled_attitude

    report = {
        "title": "The insert skill's depth limit is its attitude, through 2c/theta",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "With the throat derived from the seated criterion, orientation stopped being the "
            "insert skill's failing condition and depth took over: 35.8% reach the seated plane "
            "and the rest end a median 174.5 mm short with the clock still running. Is that "
            "creeping, jamming, or something else?"
        ),
        "answer": (
            "Something else, and it is attitude again. The stalled episodes sit at "
            f"{stalled_attitude * 1000:.1f} mrad against {float(np.median(rows[seated, index['orientation_error_rad']])) * 1000:.1f} "
            "mrad for the ones that seat, and a module held at that angle can only engage "
            f"{deepest * 1000:.0f} mm before it wedges -- which is the travel the stalled "
            "episodes actually achieve. They are not stopping short of a depth they could reach; "
            "they are as deep as their own attitude permits."
        ),
        "law": "a module held at theta can engage at most 2 * clearance_per_side / theta",
        "relieved_lateral_clearance_per_side_m": RELIEVED_LATERAL_M,
        "relieved_vertical_clearance_per_side_m": RELIEVED_VERTICAL_M,
        "deepest_engagement_at_the_stalled_attitude_m": round(deepest, 4),
        "seated_depth_tolerance_m": SEATED_DEPTH_M,
        "groups": {
            "stalled": _group(rows, index, stalled),
            "reached_seated_depth": _group(rows, index, seated),
        },
        "what_it_rules_out": [
            "Creeping. The stalled episodes are still moving at 6.13 mm/s at the median, and "
            "the previously refuted reading was that they dawdle; they are held, not slow.",
            "Grip loss. Both groups hold the pin at 11.5 mm along its axis, the measured feed, "
            "with the quartiles overlapping to a tenth of a millimetre.",
            "A wall at one plane. Terminal module x is spread from 0.43 m to the 0.676 m goal "
            "with no single stopping plane, which is what a per-episode attitude limit gives and "
            "a fixed obstruction does not.",
        ],
        "what_it_points_at": (
            "The split is bimodal, not graded: an episode either holds ~47 mrad and drives home "
            "or reaches ~97 mrad and wedges partway. Both start square -- insert_reset_bank.json "
            "reports attitude_residual_rad 0.0 at every station -- so the divergence happens "
            "during the episode. The one event both share early is the form lock softening into "
            "the remote-centre mating compliance at control step 5, with the module already "
            "inside the channel. The chain never does that: it holds the lock rigid through "
            "transit and softens at the mouth, and it gates its advance on the estimate staying "
            "inside the entry envelope. Reproducing either of those in the skill is the next "
            "measurement, and it is docs/NEXT_WORK.md T13."
        ),
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "One checkpoint (grapple_insert_l0_seed70_v23lock), one held-out seed, stage 0, "
            "260 episodes, and the checkpoint was trained on the previous rack.",
            "The 2c/theta figure uses the tightest relieved clearance; the lateral one gives "
            "324 mm rather than 261, so the arithmetic brackets the observed travel rather than "
            "predicting it to a millimetre.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for name, group in report["groups"].items():
        print(
            f"  {name:<22} n={group['episodes']:>4}  "
            f"attitude {group['orientation_mrad']['median']:>6.1f} mrad  "
            f"short {group['shortfall_mm']['median']:>6.1f} mm  "
            f"speed {group['linear_velocity_mm_s']['median']:>5.2f} mm/s"
        )
    print(f"  2c/theta at the stalled attitude: {deepest * 1000:.0f} mm of engagement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
