"""Prove the robot-side service latch fits, from measurements and no simulator.

``scripts/check_pin_gripper_clearance.py`` does this for the module's grapple
pin, and it exists because this project once specified a mushroom head that no
closure of this gripper could ever admit. The latch is the same kind of risk on
the other side of the interface: it is a new solid body bolted to the hand, in a
workcell whose whole difficulty is that the hand must not enter the rack.

Everything checked here is derived from two sources and nothing else:

* ``evidence/gripper_collision_envelope.json`` -- the measured 2F-85 envelope in
  the ``wrist_3_link`` frame;
* ``src/zero_g_blade_swap/service_latch.py`` and ``grapple_geometry.py`` -- the
  latch, pin, and rack dimensions.

Run with no arguments to check the shipped latch. ``--report`` writes the same
result as evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zero_g_blade_swap import service_latch as latch
from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    FINGERS_ONLY_DEPTH_FROM_FLANGE_M,
    GRAPPLE_PIN_COLLAR_HALF_HEIGHT,
    GRAPPLE_PIN_HALF_WIDTH_Y,
    GRAPPLE_PIN_SHAFT_HALF_HEIGHT,
    PAD_HALF_WIDTH_M,
    SLOT_ENTRY_RAMP_WIDTH_M,
    SLOT_FLOOR_TOP_Z,
    SLOT_LIP_BOTTOM_Z,
    SLOT_MOUTH_X,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = PROJECT_ROOT / "evidence" / "gripper_collision_envelope.json"
#: Blade centre when the module is fully seated in a bay. The pin axis sits at
#: the module's own mid-height.
SEATED_BLADE_CENTRE_X = 0.75
PIN_AXIS_Z = 0.72
#: Half-width of the rack channel, from the guide centres.
CHANNEL_HALF_WIDTH_Y = 0.08975


def _boxes(*, engaged: bool, seek_m: float) -> list[tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """Both jaws and both rails, as ``(name, x_range, y_range, z_range)``."""

    spans: list[tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for name, centre, size in latch.jaw_boxes(engaged=engaged, seek_m=seek_m):
        low = [centre[axis] - 0.5 * size[axis] for axis in range(3)]
        high = [centre[axis] + 0.5 * size[axis] for axis in range(3)]
        for sign, side in ((1.0, "left"), (-1.0, "right")):
            y_low, y_high = sorted((sign * low[1], sign * high[1]))
            spans.append((f"{side}_{name}", (low[0], high[0]), (y_low, y_high), (low[2], high[2])))
    for sign, side in ((1.0, "left"), (-1.0, "right")):
        y_low, y_high = sorted((sign * latch.RAIL_INNER_HALF_GAP_M, sign * latch.RAIL_OUTER_HALF_GAP_M))
        spans.append(
            (
                f"{side}_rail",
                (-latch.RAIL_HALF_HEIGHT_M, latch.RAIL_HALF_HEIGHT_M),
                (y_low, y_high),
                latch.RAIL_DEPTH_FROM_FLANGE_M,
            )
        )
    return spans


def check() -> dict[str, object]:
    envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    derived = envelope["derived"]
    hand_min = derived["gripper_envelope_min_m"]
    hand_max = derived["gripper_envelope_max_m"]
    axes = envelope["axes"]

    checks: list[dict[str, object]] = []

    def record(name: str, margin_m: float, requirement: str, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "requirement": requirement,
                "margin_m": round(margin_m, 6),
                "passed": margin_m > 0.0,
                "detail": detail,
            }
        )

    def hand_half_width_at(depth_m: float) -> tuple[float, str]:
        """How wide the hand is on the third axis at this depth from the flange.

        Taking the global maximum here is what a first version of this check
        did, and it condemned a stowed lip that parks 22 mm past the last
        knuckle. The envelope's own decomposition is the right reading: past
        ``FINGERS_ONLY_DEPTH_FROM_FLANGE_M`` the only body left is an inner
        finger, and an inner finger is 13.5 mm wide.
        """

        if depth_m >= hand_max[2]:
            return 0.0, "past every gripper body"
        if depth_m >= FINGERS_ONLY_DEPTH_FROM_FLANGE_M:
            return PAD_HALF_WIDTH_M, "inner fingers only"
        return hand_max[1], "full hand"

    # 1. The latch must never share volume with the hand it is bolted to. The
    #    hand is widest on the closing axis and the latch is deliberately placed
    #    beyond it on the third axis, so this is the margin that matters.
    for engaged in (True, False):
        state = "engaged" if engaged else "stowed"
        seeks = latch.AXIAL_SEEK_RANGE_M if engaged else (0.0,)
        for seek in seeks:
            for name, _x_range, y_range, z_range in _boxes(engaged=engaged, seek_m=seek):
                half_width, band = hand_half_width_at(z_range[0])
                if half_width <= 0.0:
                    continue  # deeper than anything the hand owns
                inner = min(abs(y_range[0]), abs(y_range[1]))
                label = f"latch_{state}_clear_of_hand:{name}"
                if engaged:
                    label += f"@seek{seek * 1000.0:+.0f}mm"
                record(
                    label,
                    inner - half_width,
                    f"latch inner face clears the {band} half-extent of {half_width:.4f} m at this depth",
                    f"seek={seek:+.4f} m, jaw depth from {z_range[0]:.4f} m, inner face at {inner:.4f} m",
                )

    # 2. Nothing on the latch may be inside the rack when the module is seated,
    #    because the latch is released before seating and a released mechanism
    #    still has to be somewhere. This is the rack half of the rule section
    #    3.1 of the interface specification applies to the gripper.
    # Depth from the flange maps to world x through the seated module: the
    # module's own face sits MODULE_FACE_FROM_FLANGE_M from the flange, and in
    # world coordinates it is half a blade length behind the seated centre.
    module_face_world_x = SEATED_BLADE_CENTRE_X - 0.5 * BLADE_LENGTH_M
    stowed_far = max(z_range[1] for _n, _x, _y, z_range in _boxes(engaged=False, seek_m=0.0))
    stowed_world_x = module_face_world_x - (latch.MODULE_FACE_FROM_FLANGE_M - stowed_far)
    record(
        "stowed_latch_clear_of_the_slot_mouth",
        SLOT_MOUTH_X - stowed_world_x,
        "with the module fully seated, no stowed latch body is past the slot mouth",
        f"deepest stowed body reaches x = {stowed_world_x:.4f} m against a mouth at {SLOT_MOUTH_X:.4f} m",
    )

    # 3. Where the *engaged* latch would foul, reported rather than required:
    #    this is the release condition the driver has to honour.
    release_before_blade_x = latch.release_before_blade_centre_x_m(
        SLOT_MOUTH_X, 0.5 * BLADE_LENGTH_M, latch.AXIAL_SEEK_RANGE_M[1]
    )

    # 3b. The destination bay's vertical lead-in is new geometry at exactly the
    #     carriage's own height, and the carriage follows the module to the
    #     mouth. A ramp that spans the module spans the carriage.
    stowed_outer = latch.WEB_OUTER_HALF_GAP_M + latch.CLOSE_STROKE_M
    stowed_inner = latch.WEB_INNER_HALF_GAP_M + latch.CLOSE_STROKE_M
    record(
        "stowed_carriage_passes_outside_the_entry_ramps",
        stowed_inner - 0.5 * SLOT_ENTRY_RAMP_WIDTH_M,
        "the stowed jaw's inner face is outboard of the vertical lead-in's half-width",
        f"stowed jaw from {stowed_inner:.4f} m to {stowed_outer:.4f} m against a ramp half-width "
        f"of {0.5 * SLOT_ENTRY_RAMP_WIDTH_M:.4f} m",
    )

    # 4. The rails follow the module toward the mouth, so they have to fit the
    #    channel on both the third axis and the closing axis even though they
    #    never enter it.
    rail_span = max(abs(latch.RAIL_OUTER_HALF_GAP_M), abs(latch.RAIL_INNER_HALF_GAP_M))
    record(
        "latch_rails_inside_the_channel_width",
        CHANNEL_HALF_WIDTH_Y - rail_span,
        "the carriage rails are narrower than the rack channel they run beside",
        f"rail outer face at {rail_span:.4f} m against a channel half-width of {CHANNEL_HALF_WIDTH_Y:.4f} m",
    )

    # 5. An engaged jaw may not reach the module's own face, or it is clamping
    #    the chassis instead of the shaft.
    record(
        "engaged_jaw_stops_short_of_the_module_face",
        latch.MODULE_FACE_FROM_FLANGE_M - latch.engaged_jaw_far_depth_m(latch.AXIAL_SEEK_RANGE_M[1]),
        "at the far end of its seek travel the jaw is still on the shaft",
        f"jaw reaches {latch.engaged_jaw_far_depth_m(latch.AXIAL_SEEK_RANGE_M[1]):.4f} m from the flange, "
        f"module face at {latch.MODULE_FACE_FROM_FLANGE_M:.4f} m",
    )

    # 6. The lip has to be backed by collar, not overhanging its rim, and it has
    #    to clear the shaft it sits above.
    record(
        "lip_is_backed_by_collar",
        GRAPPLE_PIN_COLLAR_HALF_HEIGHT - latch.JAW_HALF_HEIGHT_M,
        "the lip's outer edge is inside the collar's own half-height",
        f"jaw half-height {latch.JAW_HALF_HEIGHT_M:.4f} m, collar half-height {GRAPPLE_PIN_COLLAR_HALF_HEIGHT:.4f} m",
    )
    record(
        "lip_clears_the_shaft",
        latch.LIP_INNER_HALF_HEIGHT_M - GRAPPLE_PIN_SHAFT_HALF_HEIGHT,
        "the lip band starts above the shaft's own half-height",
        f"lip inner edge {latch.LIP_INNER_HALF_HEIGHT_M:.4f} m, shaft half-height {GRAPPLE_PIN_SHAFT_HALF_HEIGHT:.4f} m",
    )
    record(
        "lip_overlaps_the_collar_shoulder",
        GRAPPLE_PIN_HALF_WIDTH_Y - latch.LIP_INNER_HALF_GAP_M,
        "the lip reaches inboard of the collar's third-axis half-width, so it has shoulder to bear on",
        f"lip reaches {latch.LIP_INNER_HALF_GAP_M:.4f} m, collar half-width {GRAPPLE_PIN_HALF_WIDTH_Y:.4f} m",
    )

    # 7. Released, the lip must clear the collar or the module cannot leave.
    record(
        "released_lip_clears_the_collar",
        (latch.LIP_INNER_HALF_GAP_M + latch.CLOSE_STROKE_M) - GRAPPLE_PIN_HALF_WIDTH_Y,
        "after the release stroke the lip is outboard of the collar",
        f"released lip at {latch.LIP_INNER_HALF_GAP_M + latch.CLOSE_STROKE_M:.4f} m",
    )

    # 8. The whole point: the latch lives where the hand cannot reach.
    record(
        "latch_lives_past_the_hand",
        latch.COLLAR_SHOULDER_FROM_FLANGE_M - hand_max[2],
        "the engaged latch begins past the deepest measured gripper body",
        f"collar shoulder {latch.COLLAR_SHOULDER_FROM_FLANGE_M:.4f} m, hand reaches {hand_max[2]:.4f} m",
    )

    passed = all(row["passed"] for row in checks)
    return {
        "status": "passed" if passed else "failed",
        "title": "Robot-side service latch clearance, derived from the measured gripper envelope",
        "evidence_type": "geometric_derivation_no_simulator",
        "inputs": {
            "gripper_envelope": ENVELOPE.relative_to(PROJECT_ROOT).as_posix(),
            "gripper_envelope_axes": axes,
            "gripper_envelope_min_m": hand_min,
            "gripper_envelope_max_m": hand_max,
            "latch_module": "src/zero_g_blade_swap/service_latch.py",
        },
        "latch": {
            "frame": "wrist_3_link; x closing, y third, z approach out of the flange",
            "window_from_flange_m": list(latch.LATCH_WINDOW_FROM_FLANGE_M),
            "stowed_depth_from_flange_m": list(latch.STOWED_DEPTH_FROM_FLANGE_M),
            "engaged_depth_from_flange_m": list(latch.ENGAGED_DEPTH_FROM_FLANGE_M),
            "axial_seek_range_m": list(latch.AXIAL_SEEK_RANGE_M),
            "close_stroke_m": latch.CLOSE_STROKE_M,
            "extend_stroke_m": latch.EXTEND_STROKE_M,
            "lip_total_bearing_area_m2": latch.LIP_TOTAL_BEARING_AREA_M2,
            "lip_bearing_stress_at_rating_mpa": latch.lip_bearing_stress_mpa(),
            "rated_force_n": latch.RATED_FORCE_N,
            "rated_torque_nm": latch.RATED_TORQUE_NM,
            "required_axial_capacity_n": latch.REQUIRED_AXIAL_CAPACITY_N,
        },
        "release_requirement": {
            "release_before_blade_centre_x_m": round(release_before_blade_x, 6),
            "release_before_blade_centre_x_at_zero_seek_m": round(
                latch.release_before_blade_centre_x_m(SLOT_MOUTH_X, 0.5 * BLADE_LENGTH_M, 0.0), 6
            ),
            "seated_blade_centre_x_m": SEATED_BLADE_CENTRE_X,
            "note": (
                "An engaged jaw at the far end of its seek travel enters the slot mouth once the "
                "module centre passes this x. The workflow releases the latch before the final "
                "insertion leg, which is earlier than this, and the stowed carriage clears the mouth "
                "at the seated pose by the margin recorded above."
            ),
        },
        "channel": {
            "half_width_y_m": CHANNEL_HALF_WIDTH_Y,
            "floor_top_z_m": SLOT_FLOOR_TOP_Z,
            "lip_bottom_z_m": SLOT_LIP_BOTTOM_Z,
            "pin_axis_z_m": PIN_AXIS_Z,
        },
        "checks": checks,
        "scope_and_limitations": [
            "Axis-aligned bounding volumes, in the frame the gripper envelope was measured in.",
            "The latch is authored as visual geometry; its load path is the break-rated fixed joint "
            "the workflow reports, not contact between these boxes and the pin.",
            "Simulation only. No latch has been built or loaded on hardware.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args()

    result = check()
    for row in result["checks"]:
        flag = "ok  " if row["passed"] else "FAIL"
        print(f"[{flag}] {row['check']:<58} margin {row['margin_m'] * 1000.0:+8.2f} mm")
    print(f"\n{result['status'].upper()}")
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {arguments.report}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
