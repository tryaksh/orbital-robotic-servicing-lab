"""Solve the paired arm and module poses the insertion actually has to start from.

The insert skill resets at **one** module pose per bay, the certified staging
pose, 167 mm from the seated plane. The chain hands it the module at the mouth,
529 mm out, so every state the chain produces is 362 mm outside the distribution
the policy trained on. That is not a robustness gap a wider joint-noise box can
close, and this project already measured why: a hand-off is a point on a
manifold, not a box, and widening per-joint noise produces a large joint
deviation *and* a large grip error at once, so the pads close on nothing.

The manifold is not mysterious here. It is one number -- how far along the
stroke the module is -- and the arm configuration that goes with it, and the
kinematics are solved in closed form and validated against the simulator's own
recorded configurations. So the bank is derived rather than sampled from runs:

    module at x, on its bay's centre line, square
    tool at the pin's grip point for that module pose, head-on
    joints from zero_g_blade_swap.arm_kinematics.solve_ik

The same solver the transit legs are commanded from, and the same one
``scripts/check_workcell_geometry.py`` validates before it reports anything.
Every station here is checked for residual, for the DLS controller's realised
authority, and for the gripper staying clear of the rack mouth, and the script
refuses to write a bank if any station fails.

Writes ``src/zero_g_blade_swap/tasks/blade_swap/insert_reset_bank.py`` and, with
``--report``, the evidence that says what was solved and how well.
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

from zero_g_blade_swap.arm_kinematics import (  # noqa: E402
    quaternion_to_matrix,
    realised_authority,
    solve_ik,
    tool_pose,
)
from zero_g_blade_swap.grapple_geometry import (  # noqa: E402
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_PIN_COLLAR_HALF_HEIGHT,
    GRAPPLE_PIN_COLLAR_X,
    GRAPPLE_PIN_GRIP_OFFSET,
    SLOT_MOUTH_X,
    TRANSIT_CLEAR_BLADE_CENTRE_X,
)

HEAD_ON = quaternion_to_matrix(*GRAPPLE_HEAD_ON_TOOL_ROT)

#: Wrist seeds. The second bay's middle stage did not converge from the zero
#: wrist seed and did from the -90 degree one, which is the difference between
#: "unreachable" and "seeded badly"; sweep them rather than trust one.
SEEDS = [
    np.array([-0.274119, -1.325038, 1.929535, 2.537083, -1.296664, -1.570782]),
    np.array([-0.225190, -1.125436, 1.644417, 2.622605, -1.345595, -1.570812]),
    np.array([-0.588512, -1.271002, 1.856667, 2.555911, -0.982291, -1.570782]),
    np.array([-0.30, -1.10, 1.60, 2.60, -1.35, 0.0]),
]

#: Residuals a station has to meet to be written. The calibrated poses already
#: in the task converged to 0.0060 mm and 0.000011 rad, so this is not a
#: relaxation of anything.
POSITION_TOLERANCE_M = 1.0e-5
ATTITUDE_TOLERANCE_RAD = 1.0e-5
#: Below this the DLS controller does not deliver what a leg asks for; section 6a
#: of the interface specification measures the consequence.
AUTHORITY_FLOOR = 0.90


def stations(count: int, deepest: float, shallowest: float) -> list[float]:
    """Module centres along the stroke, shallow end first."""

    return [float(value) for value in np.linspace(shallowest, deepest, count)]


def solve_station(blade_x: float, bay_y: float, base: np.ndarray) -> dict[str, object]:
    """Return the arm configuration that presents the pin at ``blade_x``."""

    tool_world = np.array([blade_x + GRAPPLE_PIN_GRIP_OFFSET[0], bay_y, 0.72])
    joints, position_residual, attitude_residual = solve_ik(tool_world - base, HEAD_ON, SEEDS)
    authority = realised_authority(joints)
    achieved, _ = tool_pose(joints)
    # The collar is taller than the guided channel, so it is a depth stop and
    # must never be written inside the mouth. This is the one collision the
    # reset can create on its own; the module and the shaft both fit.
    collar_leading_x = blade_x + GRAPPLE_PIN_COLLAR_X[1]
    return {
        "blade_centre_x_m": round(blade_x, 6),
        "bay_centre_y_m": round(bay_y, 6),
        "joints_rad": [round(float(value), 6) for value in joints],
        "position_residual_m": float(position_residual),
        "attitude_residual_rad": float(attitude_residual),
        "authority_worst_rotation_axis": float(authority["authority_worst_rotation_axis"]),
        "jacobian_min_singular_value": float(authority["jacobian_min_singular_value"]),
        "tool_world_m": [round(float(value), 6) for value in (np.array(achieved) + base)],
        "collar_leading_x_m": round(collar_leading_x, 6),
        "collar_clear_of_mouth": bool(collar_leading_x <= SLOT_MOUTH_X),
        "collar_half_height_m": GRAPPLE_PIN_COLLAR_HALF_HEIGHT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=int, default=9, help="Axial stations per bay.")
    parser.add_argument(
        "--shallowest",
        type=float,
        default=round(TRANSIT_CLEAR_BLADE_CENTRE_X, 6),
        help="Module centre where the transit hands over, the shallow end of the stroke.",
    )
    parser.add_argument(
        "--deepest",
        type=float,
        default=0.5829,
        help="Module centre the certified insert reset uses, the deep end.",
    )
    parser.add_argument("--bays", type=float, nargs="*", default=[0.0, -0.22])
    parser.add_argument("--base", type=float, nargs=3, default=[-0.65, 0.0, 0.15])
    parser.add_argument(
        "--module",
        type=Path,
        default=PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "insert_reset_bank.py",
    )
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args()

    base = np.array(arguments.base)
    solved: list[list[dict[str, object]]] = []
    for bay_y in arguments.bays:
        rows = [
            solve_station(blade_x, bay_y, base)
            for blade_x in stations(arguments.stations, arguments.deepest, arguments.shallowest)
        ]
        solved.append(rows)

    print(f"{'bay y':>7} {'blade x':>9} {'pos mm':>9} {'att mrad':>9} {'authority':>10} {'sigma':>8} {'clear':>6}")
    rejected = []
    for rows in solved:
        for row in rows:
            ok = (
                row["position_residual_m"] <= POSITION_TOLERANCE_M
                and row["attitude_residual_rad"] <= ATTITUDE_TOLERANCE_RAD
                and row["authority_worst_rotation_axis"] >= AUTHORITY_FLOOR
                and row["collar_clear_of_mouth"]
            )
            if not ok:
                rejected.append(row)
            print(
                f"{row['bay_centre_y_m']:7.3f} {row['blade_centre_x_m']:9.4f} "
                f"{row['position_residual_m'] * 1000:9.5f} {row['attitude_residual_rad'] * 1000:9.5f} "
                f"{row['authority_worst_rotation_axis']:10.4f} {row['jacobian_min_singular_value']:8.4f} "
                f"{str(row['collar_clear_of_mouth']):>6}"
            )
    if rejected:
        print(f"\n{len(rejected)} stations failed their own gate; refusing to write a bank")
        return 1

    lines = [
        '"""The paired arm and module poses an insertion may start from.',
        "",
        "Generated by ``scripts/solve_insert_reset_bank.py``; do not hand-edit. Each",
        "entry is one axial station along the seating stroke and the arm",
        "configuration that presents the pin's grip point there, solved in closed",
        "form by ``zero_g_blade_swap.arm_kinematics`` and gated on residual, on the",
        "DLS controller's realised authority, and on the collar staying clear of the",
        "rack mouth. ``evidence/insert_reset_bank.json`` records all of it.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "#: Shallow end first: the module centre the transit hands over at, through",
        "#: to the certified staging pose the single-pose reset used.",
        "INSERT_STROKE_ARM_JOINT_POS = (",
    ]
    for rows in solved:
        lines.append("    (")
        for row in rows:
            lines.append(f"        {tuple(row['joints_rad'])!r},")
        lines.append("    ),")
    lines.append(")")
    lines.append("")
    lines.append("INSERT_STROKE_BLADE_POSE = (")
    for rows in solved:
        lines.append("    (")
        for row in rows:
            pose = (row["blade_centre_x_m"], row["bay_centre_y_m"], 0.72, 1.0, 0.0, 0.0, 0.0)
            lines.append(f"        {pose!r},")
        lines.append("    ),")
    lines.append(")")
    lines.append("")
    lines.append('__all__ = ["INSERT_STROKE_ARM_JOINT_POS", "INSERT_STROKE_BLADE_POSE"]')
    arguments.module.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {arguments.module}")

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(
                {
                    "title": "Paired arm and module poses along the seating stroke",
                    "evidence_type": "geometry_check_no_simulator",
                    "generated_utc": datetime.now(UTC).isoformat(),
                    "question": (
                        "What states does the insertion actually have to start from, and can the arm "
                        "hold the head-on attitude at all of them?"
                    ),
                    "method": (
                        "Closed-form UR10e inverse kinematics from zero_g_blade_swap.arm_kinematics, the "
                        "same solver the transit legs are commanded from and the one "
                        "scripts/check_workcell_geometry.py validates against the simulator's own "
                        "recorded configurations before reporting."
                    ),
                    "robot_root_local_m": list(arguments.base),
                    "gates": {
                        "position_residual_m": POSITION_TOLERANCE_M,
                        "attitude_residual_rad": ATTITUDE_TOLERANCE_RAD,
                        "authority_worst_rotation_axis": AUTHORITY_FLOOR,
                        "collar_clear_of_mouth": True,
                    },
                    "stations_by_bay": solved,
                    "scope_and_limitations": (
                        "Kinematics only. It says the arm can present the pin at every station; it says "
                        "nothing about whether the pads take the pin there, which is what the smoke run "
                        "and the certification are for."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
