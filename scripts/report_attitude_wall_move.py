"""Does the insert skill's attitude floor move with the channel throat?

T9 ended with the insert skill failing on orientation alone, and with a claim
about why: the terminal attitude of every episode that reaches seated depth sits
in a band under a milliradian wide, which is a wall rather than a behaviour, and
the wall is the channel's lateral clearance through ``2c/L``.

That claim was inferred from a coincidence of numbers, and this project has been
wrong about exactly that shape of thing five times. So it is tested the only way
a wall can be tested: **move the wall and see whether the floor moves with it**,
with the same checkpoint, the same seed and the same episode count, so nothing
but the rack differs.

``GUIDE_CENTER_OFFSET_Y`` went 86.689 -> 85.065 mm, taking the channel's
unrelieved lateral clearance from 12.689 mm to 11.065 mm. The checkpoint is
``grapple_insert_l0_seed70_v23lock``, trained on the *old* rack, so this is
zero-shot on the new one and the policy is off its training distribution. That
weakens the success number and does not weaken the floor, which is geometry.

CPU only. Reads the ``.npz`` rows the two evaluations already wrote.

Usage::

    python scripts/report_attitude_wall_move.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "insert_attitude_wall_moved.json"

#: ``mdp.insertion.INSERTION_ORIENTATION_TOLERANCE_RAD``, kept as a literal so
#: this script needs no simulator import.
SUCCESS_ORIENTATION_RAD = 0.0523599
#: ``mdp.insertion.INSERTION_AXIAL_DEPTH_TOLERANCE_M``.
SEATED_DEPTH_M = 0.012
#: ``mdp.insertion.INSERTION_LATERAL_TOLERANCE_M``.
LATERAL_TOLERANCE_M = 0.0025
MODULE_LENGTH_M = 0.45

ARMS = (
    (
        "artifacts/insert_attitude/v23lock_s0_seed1070.npz",
        "old rack, GUIDE_CENTER_OFFSET_Y 86.689 mm",
        0.012689,
    ),
    (
        "artifacts/insert_wall/v23lock_newrack.npz",
        "new rack, GUIDE_CENTER_OFFSET_Y 85.065 mm",
        0.011065,
    ),
)


def _arm(path: Path, label: str, clearance_m: float) -> dict[str, object]:
    archive = np.load(path, allow_pickle=True)
    index = {str(name): position for position, name in enumerate(archive["fields"])}
    rows = archive["rows"]

    axial = rows[:, index["axial_error_m"]]
    orientation = rows[:, index["orientation_error_rad"]]
    lateral = rows[:, index["lateral_error_m"]]
    deep = axial <= SEATED_DEPTH_M
    seated = orientation[deep]
    wall = 2.0 * clearance_m / MODULE_LENGTH_M

    return {
        "label": label,
        "lateral_clearance_per_side_m": round(clearance_m, 6),
        "acceptance_law_wall_mrad": round(wall * 1000.0, 3),
        "episodes": int(len(rows)),
        "success_rate": round(float(rows[:, index["success"]].mean()), 6),
        "reached_seated_depth": int(deep.sum()),
        "fraction_reaching_seated_depth": round(float(deep.mean()), 4),
        "median_shortfall_mm": round(float(np.median(axial)) * 1000.0, 3),
        "median_lateral_mm": round(float(np.median(lateral)) * 1000.0, 3),
        "seated_orientation_mrad": {
            "min": round(float(seated.min()) * 1000.0, 3),
            "p5": round(float(np.percentile(seated, 5)) * 1000.0, 3),
            "median": round(float(np.median(seated)) * 1000.0, 3),
            "p95": round(float(np.percentile(seated, 95)) * 1000.0, 3),
        },
        "floor_over_wall": round(float(seated.min()) / wall, 4),
        "fraction_inside_success_orientation": round(
            float((orientation < SUCCESS_ORIENTATION_RAD).mean()), 4
        ),
        "seated_conditions_passing": {
            "orientation": round(float((seated < SUCCESS_ORIENTATION_RAD).mean()), 4),
            "lateral": round(float((lateral[deep] <= LATERAL_TOLERANCE_M).mean()), 4),
        },
    }


def main() -> int:
    arms = []
    for pattern, label, clearance in ARMS:
        path = ROOT / pattern
        if not path.is_file():
            raise SystemExit(f"missing evaluation rows: {path}")
        arms.append(_arm(path, label, clearance))

    before, after = arms
    report = {
        "title": "The insert skill's attitude floor is the channel throat, and it moves with it",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "T9 read the insert skill's terminal attitude band as a wall set by the channel's "
            "lateral clearance through 2c/L, from a coincidence of numbers. Does the band move "
            "when the clearance moves?"
        ),
        "answer": (
            f"Yes. Narrowing the channel from {before['lateral_clearance_per_side_m'] * 1000:.3f} mm "
            f"to {after['lateral_clearance_per_side_m'] * 1000:.3f} mm per side moves the floor of "
            f"the seated-attitude band from {before['seated_orientation_mrad']['min']:.2f} mrad to "
            f"{after['seated_orientation_mrad']['min']:.2f} mrad, on the same checkpoint, seed and "
            "episode count. The band is geometry, not behaviour, and the geometry is the throat."
        ),
        "method": (
            "One checkpoint, grapple_insert_l0_seed70_v23lock, trained on the OLD rack and "
            "evaluated on both. Zero-shot on the new one, so the policy is off its training "
            "distribution there; that weakens the success number and not the floor."
        ),
        "law": "a module fully inside a channel with c per side wedges at 2c/L and cannot be squarer",
        "success_orientation_tolerance_mrad": SUCCESS_ORIENTATION_RAD * 1000.0,
        "arms": arms,
        "what_it_settles": [
            "The attitude band is set by the channel's lateral throat: the floor moved 10.28 mrad "
            "when the throat moved 1.62 mm, on an unchanged checkpoint. The move is 1.4x what the "
            f"law alone predicts -- 2c/L falls by 7.22 mrad -- so the floor sits at "
            f"{before['floor_over_wall']:.3f} of its wall on the old rack and "
            f"{after['floor_over_wall']:.3f} on the new one. The throat is the cause; it is not "
            "the whole of the quantitative story, and a retrained policy is what would say "
            "whether the remaining 3.4 mrad is the policy having room it did not have before.",
            "It is not the relief. The relieved channel is 15.68 x 12.61 mm and admits 69.7 mrad "
            "of yaw; the module rests far inside that, against the unrelieved throat the lead-ins "
            "hold at the mouth. evidence/destination_channel_geometry.json measures both.",
            "It is not the policy's. Nothing about the checkpoint changed between these two rows.",
            "And with the throat out of the way it stops binding: the floor lands at 45.75 mrad, "
            "3.43 mrad inside the new wall, where it sat 0.36 mrad inside the old one. 45.75 mrad "
            "is also where the CHAIN's successful episodes seat -- min 45.754, p50 45.815, p95 "
            "46.452 over 94 episodes in "
            "workflow_robot_carried_m130pin_guarded_c11065_certification.json -- which is what the "
            "form lock delivers when no rack surface is fighting it. The skill's tail is still "
            "wider than the chain's, p95 52.35 against 46.45.",
        ],
        "what_it_does_not_settle": [
            "This is not a certification. One seed, stage 0, 256 episodes, and the policy was "
            "trained on the other rack.",
            "Success is now limited by lateral alignment rather than by attitude: among episodes "
            f"that reach seated depth on the new rack, "
            f"{after['seated_conditions_passing']['orientation'] * 100:.1f}% are inside the "
            f"orientation tolerance and {after['seated_conditions_passing']['lateral'] * 100:.1f}% "
            "are inside the 2.5 mm lateral one. That is the policy's to improve and is what a "
            "retrain on this rack is for.",
        ],
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "One PPO training seed. The evaluation seed is held out.",
            "Stage 0 only. The published skill certifications pool three curriculum stages.",
        ],
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    header = f"{'arm':<42}{'wall':>8}{'floor':>8}{'median':>8}{'deep':>7}{'success':>9}"
    print(header)
    for arm in arms:
        print(
            f"{arm['label']:<42}"
            f"{arm['acceptance_law_wall_mrad']:>8.2f}"
            f"{arm['seated_orientation_mrad']['min']:>8.2f}"
            f"{arm['seated_orientation_mrad']['median']:>8.2f}"
            f"{arm['fraction_reaching_seated_depth'] * 100:>6.1f}%"
            f"{arm['success_rate'] * 100:>8.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
