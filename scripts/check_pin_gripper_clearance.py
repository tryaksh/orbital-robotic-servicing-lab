#!/usr/bin/env python3
"""Does the grapple pin fit inside the gripper that grips it?

Every section of the pin occupies a depth band from the flange when the pads are
seated, and the gripper's own bodies occupy that same volume.  This intersects
the two using ``evidence/gripper_collision_envelope.json`` and reports the
interference in millimetres.  No simulator, so it is a geometry gate that can run
on every commit -- which matters, because the failure it was written to find is
one no amount of policy training can fix and no reset-noise sweep can reveal.

**It is measured against the vertex-sampled throat profile, not against body
bounding boxes, and the difference decided the question.** A first version
intersected axis-aligned bounds and reported that the *certified* tapered pin
interferes with the inner knuckle by 21.3 mm -- a geometry that extracts at
99.02%. A probe that condemns the working control is measuring the wrong thing:
an axis-aligned box around a slanted knuckle claims volume the knuckle does not
occupy, and taking a tapered section at its widest point claims volume the pin
does not occupy either. Both are fixed here. Run ``--control`` to re-check that
the tapered pin still passes; if it ever does not, distrust this script before
distrusting the pin.

The frame is ``wrist_3_link``: ``z`` is the approach axis measured from the
flange, ``x`` is the closing axis the pads move along, ``y`` is the third axis.
A pin point at blade-local ``bx`` sits at depth ``bx - GRIP_OFFSET_X + TOOL_Z``,
because a seated grip puts the blade's grip point on the tool frame origin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zero_g_blade_swap import grapple_geometry as geom

REPO = Path(__file__).resolve().parent.parent
ENVELOPE = REPO / "evidence" / "gripper_collision_envelope.json"
#: The throat profile is sampled for a few third-axis half-widths; the pin's is
#: 15 mm, so that is the column to read.
THIRD_AXIS_KEY = f"{geom.GRAPPLE_PIN_HALF_WIDTH_Y:.3f}"


def _depth_of(blade_local_x: float) -> float:
    """Depth from the flange of a blade-local point, with the grip seated."""

    return blade_local_x - geom.GRAPPLE_PIN_GRIP_OFFSET[0] + geom.GRAPPLE_TOOL_OFFSET_POS[2]


def _pin_half_height_at_depth(depth_m: float) -> tuple[str, float] | None:
    """(section name, half-height on the closing axis) of the pin at a depth."""

    blade_x = depth_m + geom.GRAPPLE_PIN_GRIP_OFFSET[0] - geom.GRAPPLE_TOOL_OFFSET_POS[2]
    if hasattr(geom, "GRAPPLE_PIN_KEY_X"):
        sections = [
            ("nose", geom.GRAPPLE_PIN_NOSE_X, lambda _: geom.GRAPPLE_PIN_NOSE_HALF_HEIGHT),
            ("key", geom.GRAPPLE_PIN_KEY_X, lambda _: geom.GRAPPLE_PIN_KEY_HALF_HEIGHT),
        ]
    else:
        # The taper's half-height varies along its length; taking it at its widest
        # point over the whole span is what produced the false positive above.
        sections = [("wedge", geom.GRAPPLE_PIN_WEDGE_X, geom.wedge_half_height_at)]
    sections += [
        ("collar", geom.GRAPPLE_PIN_COLLAR_X, lambda _: geom.GRAPPLE_PIN_COLLAR_HALF_HEIGHT),
        ("shaft", geom.GRAPPLE_PIN_SHAFT_X, lambda _: geom.GRAPPLE_PIN_SHAFT_HALF_HEIGHT),
    ]
    for name, (start, end), half_height in sections:
        if start - 1.0e-9 <= blade_x <= end + 1.0e-9:
            return name, half_height(blade_x)
    return None


def _achieved_closure() -> tuple[float, float]:
    """(pad half-opening, finger_joint) the drive actually reaches on this pin.

    **The closure is not a free variable, and assuming it was is what condemned
    the working control twice.** The pads are rigid and 57 mm long, so they close
    until they meet the tallest thing they straddle; every body in the hand,
    including the knuckles that sweep the throat, is then positioned by that
    stop. Checking a pin against a closure its own section prevents asks whether
    it would fit inside a hand closed on nothing, which is not a question about
    this interface.

    On the taper this lands near 0.19 rad, and the taper's profile is then very
    close to the throat's own -- which is the real reason a cone works on this
    hand and a step does not.
    """

    near, far = geom.PAD_SPAN_FROM_FLANGE_M
    steps = int(round((far - near) / 0.001)) + 1
    tallest = 0.0
    for index in range(steps):
        found = _pin_half_height_at_depth(near + index * 0.001)
        if found is not None:
            tallest = max(tallest, found[1])
    opening = 2.0 * tallest
    return tallest, (geom.MAX_CLEAR_OPENING_M - opening) / geom.CLOSING_RATE_M_PER_RAD


def analyse(closure_range: tuple[float, float] | None) -> dict:
    envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    profile = envelope["throat_profile"]
    seated_half_height, achieved = _achieved_closure()
    if closure_range is None:
        # Every sampled closure at or below the one the pin allows. Anything
        # tighter is a pose the pin physically prevents the hand from reaching.
        closure_range = (0.0, achieved)

    findings = []
    unknown_slices = 0
    for command in profile["by_command"]:
        closure = command["finger_joint_rad"]
        if not (closure_range[0] - 1.0e-9 <= closure <= closure_range[1] + 1.0e-9):
            continue
        for entry in command["slices"]:
            near, far = entry["depth_from_flange_m"]
            # The pin is widest at whichever end of the slice its profile grows
            # toward, so both ends are checked and the taller taken.
            candidates = [
                value
                for value in (_pin_half_height_at_depth(near), _pin_half_height_at_depth(far))
                if value is not None
            ]
            if not candidates:
                continue
            section, half_height = max(candidates, key=lambda item: item[1])
            nearest = entry["nearest_other_bodies_m"].get(THIRD_AXIS_KEY)
            if nearest is None:
                # The envelope's own caveat: a null slice has no sampled vertices,
                # which is not proof of clearance. Counted, never scored as clear.
                unknown_slices += 1
                continue
            interference = half_height - nearest
            if interference > 0.0:
                findings.append(
                    {
                        "section": section,
                        "depth_from_flange_m": [near, far],
                        "finger_joint_rad": closure,
                        "pin_half_height_m": round(half_height, 6),
                        "nearest_gripper_body_m": nearest,
                        "interference_m": round(interference, 6),
                    }
                )

    findings.sort(key=lambda item: -item["interference_m"])
    return {
        "title": "Grapple-pin sections against the measured gripper throat profile",
        "evidence_type": "geometric_derivation",
        "source_envelope": str(ENVELOPE.relative_to(REPO)).replace("\\", "/"),
        "measurement": "collision-mesh vertices sampled per 5 mm depth slice, not body bounding boxes",
        "frame": "wrist_3_link: z from the flange along the approach axis, x the closing axis",
        "interface": "keyed" if hasattr(geom, "GRAPPLE_PIN_KEY_X") else "tapered",
        "closure_range_rad": list(closure_range),
        "seated_pad_half_opening_m": round(seated_half_height, 6),
        "closure_the_pin_allows_rad": round(achieved, 6),
        "pad_span_from_flange_m": list(geom.PAD_SPAN_FROM_FLANGE_M),
        "palm_face_from_flange_m": geom.PALM_FACE_FROM_FLANGE_M,
        "interferences": findings,
        "worst_interference_m": findings[0]["interference_m"] if findings else 0.0,
        "slices_without_sampled_vertices": unknown_slices,
        "passed": not findings,
        "scope_and_limitations": [
            "Static geometry with the grip seated. It says nothing about an approach.",
            "A slice with no sampled vertices is reported as unknown, never as clear.",
            "The pads are excluded by construction: the throat profile separates them from the "
            "other bodies, and the pads are meant to touch the pin.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=None, help="Write the finding as JSON.")
    parser.add_argument(
        "--closure_range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Restrict to the finger_joint band the task commands. Default: every sampled closure.",
    )
    args = parser.parse_args()

    result = analyse(tuple(args.closure_range) if args.closure_range else None)
    print(f"  interface: {result['interface']}")
    print(f"  the pin stops the pads at {result['seated_pad_half_opening_m']*1000:.1f} mm half-opening, "
          f"finger_joint {result['closure_the_pin_allows_rad']:.3f} rad")
    print(f"  checked over closures {result['closure_range_rad'][0]:.3f}-"
          f"{result['closure_range_rad'][1]:.3f} rad")
    if result["passed"]:
        print("  no section interferes with any non-pad body at any sampled closure")
    else:
        print(f"  {len(result['interferences'])} interfering slices, worst first:")
        for entry in result["interferences"][:12]:
            near, far = entry["depth_from_flange_m"]
            print(f"    {entry['section']:6s} depth {near*1000:6.1f}-{far*1000:6.1f} mm  "
                  f"pin {entry['pin_half_height_m']*1000:5.1f} mm vs room "
                  f"{entry['nearest_gripper_body_m']*1000:5.1f} mm  -> "
                  f"{entry['interference_m']*1000:5.1f} mm at finger_joint {entry['finger_joint_rad']:.3f}")
    print(f"  slices with no sampled vertices (unknown, not clear): "
          f"{result['slices_without_sampled_vertices']}")
    print(f"  passed: {result['passed']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.report}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
