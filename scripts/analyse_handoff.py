"""Compare the state a chain hands a skill against the state that skill trains on.

A skill has to be trained across the states its predecessor actually produces.
That sentence is in CLAUDE.md because ignoring it has now cost this project three
separate failures, and until this script existed nothing here measured what those
states are: the reset distributions were chosen, and the hand-off was inferred
from whatever number happened to be at hand.

Reads the .npz `run_workflow_demo.py --handoff_trace` writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PHASE_NAMES = ("capture", "seat", "extract", "transit", "insert", "done")
#: The arm pose the insert and extract tasks reset around, and the per-joint
#: uniform noise they add. Read from the task rather than restated: see
#: assets.GRAPPLE_HEAD_ON_ARM_JOINT_POS and grapple_pin_env_cfg.
NOMINAL_HELP = "assets.GRAPPLE_HEAD_ON_ARM_JOINT_POS[stage]"


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    p = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p01": float(p[0]),
        "p05": float(p[1]),
        "p50": float(p[2]),
        "p95": float(p[3]),
        "p99": float(p[4]),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--to_phase",
        default="insert",
        choices=PHASE_NAMES,
        help="Report the state handed to this phase.",
    )
    parser.add_argument(
        "--nominal",
        type=float,
        nargs=6,
        default=None,
        help=f"Arm joint pose the receiving skill resets around ({NOMINAL_HELP}).",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.trace)
    handoff = data["handoff"]
    fields = [str(name) for name in data["handoff_fields"]]
    index = {name: position for position, name in enumerate(fields)}
    target = PHASE_NAMES.index(args.to_phase)
    rows = handoff[handoff[:, index["to_phase"]] == target]

    report: dict[str, object] = {
        "trace": str(args.trace),
        "handed_to": args.to_phase,
        "hand_offs": int(rows.shape[0]),
    }
    scalar_columns = [
        "grip_error_m",
        "grip_attitude_rad",
        "finger_angle_rad",
        "drive_torque_nm",
        "blade_x_m",
        "blade_y_m",
        "blade_z_m",
        "blade_linear_velocity_mps",
        "blade_angular_velocity_radps",
    ]
    report["state"] = {name: _percentiles(rows[:, index[name]]) for name in scalar_columns if name in index}

    joints = np.stack([rows[:, index[f"arm_joint_{axis}"]] for axis in range(6)], axis=-1)
    report["arm_joint_pos"] = {f"joint_{axis}": _percentiles(joints[:, axis]) for axis in range(6)}
    if args.nominal is not None:
        deviation = np.abs(joints - np.asarray(args.nominal))
        report["arm_joint_deviation_from_nominal_rad"] = {
            f"joint_{axis}": _percentiles(deviation[:, axis]) for axis in range(6)
        }
        report["arm_joint_deviation_worst_axis_rad"] = _percentiles(deviation.max(axis=-1))

    if "settle" in data and data["settle"].shape[0]:
        settle = data["settle"]
        settle_fields = [str(name) for name in data["settle_fields"]]
        si = {name: position for position, name in enumerate(settle_fields)}
        elapsed = settle[:, si["steps_since_done"]]
        curve = {}
        for step in sorted({int(value) for value in elapsed}):
            window = settle[elapsed == step]
            curve[f"step_{step:02d}"] = {
                "t_s": round(step / 30.0, 3),
                "n": int(window.shape[0]),
                "blade_linear_velocity_mps_p50": float(np.percentile(window[:, si["blade_linear_velocity_mps"]], 50)),
                "blade_angular_velocity_radps_p50": float(
                    np.percentile(window[:, si["blade_angular_velocity_radps"]], 50)
                ),
                "grip_error_m_p50": float(np.percentile(window[:, si["grip_error_m"]], 50)),
                "drive_torque_nm_p50": float(np.percentile(window[:, si["drive_torque_nm"]], 50)),
                "blade_x_m_p50": float(np.percentile(window[:, si["blade_x_m"]], 50)),
            }
        report["settling_window"] = curve

    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
