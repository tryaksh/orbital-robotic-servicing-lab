#!/usr/bin/env python3
"""Which way is the module cocked when it wedges: yaw, pitch, or roll?

Twelve episodes in 192 still fail when the destination channel is rebuilt at the
design library's prescribed clearance, with the retention pawls fitted. They
reach the insert phase, the insertion predicate never fires, and the module
stops at x = 0.220 m against a seated plane at 0.676 m with a peak orientation
error of 83.6 to 90.8 mrad. Two explanations fit that equally well:

* **yaw** -- rotation about the vertical axis, wedging between the two side
  guides. Lateral clearance governs, and the library's ``2c/theta`` bound should
  have forecast it. It did not: at these angles it permits 250.7 mm of
  engagement and the module wedges at 88.0 mm.
* **pitch** -- rotation about a horizontal axis, wedging under the bay's
  vertical lead-in. The governing clearance is the vertical one and the lateral
  bound was never the relevant gate, which would explain the factor of three.

``orientation_error_rad`` is an axis-angle norm and reads identically for both,
which is why this could not be settled from the traces that existed. The insert
trace now carries the full quaternion.

This decomposes it and reports, for the episodes that wedge against those that
seat: the rotation split by axis, and where the module's corners end up. The
corner positions are **derived from the recorded pose and the module's
dimensions**, not read from a contact sensor -- the workflow scene has none, and
adding one would change the scene these numbers were measured in.

    python scripts/report_jam_mechanism.py --report evidence/jam_mechanism_v1.json
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

from zero_g_blade_swap.grapple_geometry import (  # noqa: E402
    BLADE_LENGTH_M,
    BLADE_THICKNESS_M,
    BLADE_WIDTH_M,
)
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

SEEDS = (4070, 5070, 6070)
ARMS = {
    "reference (relief 4.61 mm, pawls)": "artifacts/jam_mechanism/reference/seed{seed}/nominal_trace.npz",
    "prescribed (relief 0.00 mm, pawls)": "artifacts/jam_mechanism/prescribed/seed{seed}/nominal_trace.npz",
    "prescribed, no pawls": "artifacts/jam_mechanism/prescribed_bare/seed{seed}/nominal_trace.npz",
}
#: An episode whose module never gets this far did not complete the stroke.
STROKE_COMPLETE_X_M = 0.60


def euler_from_quat(quat: np.ndarray) -> np.ndarray:
    """Intrinsic roll (about x), pitch (about y), yaw (about z), in radians.

    ``quat`` is (w, x, y, z), the convention Isaac uses. x is the insertion
    axis in this scene, y the lateral direction across the bay, z vertical, so
    the three components map onto the three explanations directly.
    """

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def corner_extents(position: np.ndarray, quat: np.ndarray) -> dict[str, float]:
    """Half-extents of the module's bounding corners in world y and z.

    Derived from the recorded pose and the module's dimensions. This is where
    the corners *are*, which is where contact must be if there is contact; it is
    not a contact report and does not claim a normal force.
    """

    half = np.array([BLADE_LENGTH_M, BLADE_WIDTH_M, BLADE_THICKNESS_M]) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local = signs * half
    w, x, y, z = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    corners = position + local @ rotation.T
    return {
        "corner_y_span_m": float(corners[:, 1].max() - corners[:, 1].min()),
        "corner_z_span_m": float(corners[:, 2].max() - corners[:, 2].min()),
        "corner_y_max_m": float(corners[:, 1].max()),
        "corner_z_max_m": float(corners[:, 2].max()),
        "corner_z_min_m": float(corners[:, 2].min()),
    }


def analyse(paths: list[Path]) -> dict | None:
    jammed: list[dict] = []
    seated: list[dict] = []
    for path in paths:
        if not path.exists():
            return None
        trace = np.load(path, allow_pickle=True)
        fields = [str(n) for n in trace["insert_fields"]]
        if "blade_qw" not in fields:
            return {"error": f"{path.name} predates the quaternion columns"}
        block = trace["insert"]
        if block.size == 0:
            continue
        column = {n: fields.index(n) for n in fields}
        for env in np.unique(block[:, column["env"]]).astype(int):
            rows = block[block[:, column["env"]] == env]
            rows = rows[np.argsort(rows[:, column["step"]])]
            final = rows[-1]
            quat = final[[column["blade_qw"], column["blade_qx"], column["blade_qy"], column["blade_qz"]]]
            position = final[[column["true_blade_x_m"], column["true_blade_y_m"], column["true_blade_z_m"]]]
            euler = euler_from_quat(quat)
            entry = {
                "env": int(env),
                "final_x_m": float(final[column["true_blade_x_m"]]),
                "peak_orientation_rad": float(rows[:, column["orientation_error_rad"]].max()),
                "roll_rad": float(euler[0]),
                "pitch_rad": float(euler[1]),
                "yaw_rad": float(euler[2]),
                **corner_extents(position, quat),
            }
            (jammed if entry["final_x_m"] < STROKE_COMPLETE_X_M else seated).append(entry)
    if not jammed and not seated:
        return None

    def summarise(group: list[dict]) -> dict:
        if not group:
            return {}
        return {
            "episodes": len(group),
            **{
                key: float(np.median([e[key] for e in group]))
                for key in ("roll_rad", "pitch_rad", "yaw_rad", "corner_z_span_m", "corner_y_span_m")
            },
        }

    return {"jammed": summarise(jammed), "seated": summarise(seated), "jammed_detail": jammed}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    arms: dict[str, dict] = {}
    for label, template in ARMS.items():
        result = analyse([ROOT / template.format(seed=s) for s in SEEDS])
        if result is None:
            print(f"{label}: traces missing, skipping")
            continue
        if "error" in result:
            print(f"{label}: {result['error']}")
            continue
        arms[label] = result

    print(f"{'arm':>36} {'group':>8} {'n':>4} {'roll':>9} {'pitch':>9} {'yaw':>9}")
    for label, arm in arms.items():
        for group in ("seated", "jammed"):
            entry = arm.get(group) or {}
            if not entry:
                continue
            print(
                f"{label:>36} {group:>8} {entry['episodes']:>4} "
                f"{entry['roll_rad'] * 1000:>9.2f} {entry['pitch_rad'] * 1000:>9.2f} "
                f"{entry['yaw_rad'] * 1000:>9.2f}"
            )
    print("  (median Euler components in mrad; x is the insertion axis, y lateral, z vertical)")
    print("   yaw is rotation between the side guides; pitch is rotation under the lead-in")

    document = {
        "title": "Which axis the module rotates about when the prescribed channel jams it",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Twelve episodes still fail at the prescribed clearance with the pawls "
            "fitted. Is the module wedging in yaw between the side guides, where the "
            "library's lateral bound would govern, or in pitch under the vertical "
            "lead-in, where it never applied?"
        ),
        "source_revision": git_source_revision(ROOT),
        "seeds": list(SEEDS),
        "module_dimensions_m": {
            "length": BLADE_LENGTH_M,
            "width": BLADE_WIDTH_M,
            "thickness": BLADE_THICKNESS_M,
        },
        "stroke_complete_threshold_x_m": STROKE_COMPLETE_X_M,
        "arms": arms,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "Corner positions are derived from the recorded pose and the module's "
            "nominal dimensions. They say where the corners are, which is where "
            "contact must be if there is contact. They are not a contact report and "
            "carry no normal force: the workflow scene has no contact sensor, and "
            "adding one would change the scene these numbers were measured in.",
            "Euler components are taken at the last recorded insert sample, so for a "
            "jammed episode they describe the resting pose rather than the instant of "
            "first contact.",
            "One point of the sweep at three seeds with one frozen checkpoint set. "
            "Nothing here says what a differently trained controller would do.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
