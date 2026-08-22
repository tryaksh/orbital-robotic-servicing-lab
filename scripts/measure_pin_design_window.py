#!/usr/bin/env python3
"""How tall a feature can the gripper accept, at each depth along its approach?

``check_pin_gripper_clearance.py`` asks whether a pin *as designed* fits.  This
asks the prior question: what is there room for at all.  For each 1 mm slice of
depth from the flange it reports the largest half-height, on the closing axis, a
30 mm-wide pin feature could have without touching anything but the pads.

That profile is what a capture feature has to be designed inside, and it settles
an argument no amount of policy training can: whether a *positive axial stop
ahead of the pads* -- the thing a keyed interface needs and a taper approximates
by wedging -- is geometrically available on this hand.

Reads ``evidence/gripper_collision_envelope.json`` and nothing else, so it needs
no simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zero_g_blade_swap import grapple_geometry as geom

REPO = Path(__file__).resolve().parent.parent
ENVELOPE = REPO / "evidence" / "gripper_collision_envelope.json"
PAD_BODIES = ("left_inner_finger", "right_inner_finger")
#: The pin is this wide on the third axis; a body that misses it there cannot
#: obstruct it however close it comes on the closing axis.
PIN_HALF_WIDTH_Y = geom.GRAPPLE_PIN_HALF_WIDTH_Y


def profile(slice_m: float = 0.001, max_depth_m: float = 0.26,
            closures: tuple[float, float] | None = None) -> list[dict]:
    envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    bodies = envelope["bodies"]
    #: The tallest thing worth reporting: past this the feature is wider than the
    #: gripper's own envelope and the question stops being meaningful.
    ceiling = max(sample["max_m"][0] for samples in bodies.values() for sample in samples) + 0.010

    slices = []
    steps = int(round(max_depth_m / slice_m))
    for index in range(steps):
        near = index * slice_m
        far = near + slice_m
        allowed = ceiling
        limiter = None
        for body, samples in bodies.items():
            if body in PAD_BODIES:
                continue
            for sample in samples:
                if closures is not None and not (
                    closures[0] - 1.0e-9 <= sample["finger_joint_rad"] <= closures[1] + 1.0e-9
                ):
                    continue
                low, high = sample["min_m"], sample["max_m"]
                if min(far, high[2]) - max(near, low[2]) <= 0.0:
                    continue
                if min(PIN_HALF_WIDTH_Y, high[1]) - max(-PIN_HALF_WIDTH_Y, low[1]) <= 0.0:
                    continue
                # The body spans low[0]..high[0] on the closing axis. A feature
                # centred on that axis is stopped by whichever face is nearer to
                # it; a body straddling the axis leaves no room at all.
                room = 0.0 if low[0] <= 0.0 <= high[0] else min(abs(low[0]), abs(high[0]))
                if room < allowed:
                    allowed = room
                    limiter = {"body": body, "finger_joint_rad": sample["finger_joint_rad"]}
        slices.append(
            {
                "depth_from_flange_m": round(near, 4),
                "max_half_height_m": round(allowed, 6),
                "limited_by": limiter,
            }
        )
    return slices


def stop_feasibility() -> list[dict]:
    """Can a positive axial stop ever sit ahead of the pads on this hand?

    A keyed interface needs a shoulder forward of the pads: the pads seat in a
    pocket and the shoulder stops them pulling off. The shoulder has to be taller
    than the gripped section, or it is not a stop.

    The catch is that the two are coupled. A shorter gripped section lets the hand
    close further, and the knuckles that sweep the throat come *in* as it does, so
    the room ahead of the pads shrinks with the very number that was supposed to
    make the stop easier to fit. This sweeps the gripped half-height and reports
    both sides of that trade at each value.
    """

    envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    profile = envelope["throat_profile"]
    third_axis = f"{PIN_HALF_WIDTH_Y:.3f}"
    pad_near = geom.PAD_SPAN_FROM_FLANGE_M[0]

    rows = []
    for step in range(1, 45):
        half_height = step * 0.001
        closure = (geom.MAX_CLEAR_OPENING_M - 2.0 * half_height) / geom.CLOSING_RATE_M_PER_RAD
        if not 0.0 <= closure <= geom.FINGER_JOINT_RANGE_RAD[1]:
            continue
        # Where a pocket wall has to live: between the palm face, the deepest the
        # pin can reach without passing through the hand, and the seated pads.
        # Taking the best slice anywhere ahead of the pads is wrong and said so
        # loudly -- it picked a slice 20 mm from the flange, behind the palm,
        # which no pin can reach without going through it.
        window = (geom.PALM_FACE_FROM_FLANGE_M, pad_near)
        rooms = []
        for command in profile["by_command"]:
            if abs(command["finger_joint_rad"] - closure) > 0.038:
                continue
            for entry in command["slices"]:
                near, far = entry["depth_from_flange_m"]
                if near < window[0] - 1.0e-9 or far > window[1] + 1.0e-9:
                    continue
                room = entry["nearest_other_bodies_m"].get(third_axis)
                if room is not None:
                    rooms.append(room)
        # The wall spans the window, so the tightest slice in it is what governs.
        best_room = min(rooms) if rooms else 0.0
        rows.append(
            {
                "gripped_half_height_m": round(half_height, 6),
                "closure_rad": round(closure, 6),
                "tallest_stop_that_fits_ahead_of_pads_m": round(best_room, 6),
                "stop_is_possible": best_room > half_height,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--slice_mm", type=float, default=1.0)
    parser.add_argument(
        "--stop_feasibility",
        action="store_true",
        help="Sweep the gripped section height and ask whether a forward axial stop can ever fit.",
    )
    parser.add_argument(
        "--closure_range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help=(
            "Restrict to the finger_joint band the task actually commands. Without it the "
            "profile is over every closure, which is the right question for a hand that may "
            "close fully and the wrong one for a pin that stops it."
        ),
    )
    args = parser.parse_args()

    if args.stop_feasibility:
        rows = stop_feasibility()
        print("  gripped half-height -> closure it forces -> tallest stop that fits ahead of the pads")
        for row in rows:
            if int(round(row["gripped_half_height_m"] * 1000)) % 2:
                continue
            verdict = "POSSIBLE" if row["stop_is_possible"] else "impossible"
            print(f"    {row['gripped_half_height_m']*1000:5.1f} mm   closure "
                  f"{row['closure_rad']:.3f} rad   room "
                  f"{row['tallest_stop_that_fits_ahead_of_pads_m']*1000:5.1f} mm   {verdict}")
        possible = [row for row in rows if row["stop_is_possible"]]
        print(f"\n  gripped heights admitting a forward stop: {len(possible)} of {len(rows)}")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({
                "title": "Can a positive axial stop sit ahead of the pads on the 2F-85?",
                "evidence_type": "geometric_derivation",
                "source_envelope": str(ENVELOPE.relative_to(REPO)).replace("\\", "/"),
                "question": (
                    "A keyed interface needs a shoulder forward of the pads, taller than the gripped "
                    "section. The gripped section sets the closure, and the closure sets how much room "
                    "the knuckles leave ahead of the pads."
                ),
                "gripped_heights_admitting_a_forward_stop": len(possible),
                "gripped_heights_tested": len(rows),
                "rows": rows,
                "scope_and_limitations": [
                    "Collision-mesh vertices per 5 mm depth slice; a slice with no vertices is skipped, "
                    "so the room reported is the best sampled, not a guaranteed clearance.",
                    "A stop is assumed centred on the tool axis and 30 mm wide on the third axis.",
                ],
            }, indent=2) + "\n", encoding="utf-8")
            print(f"  wrote {args.report}")
        return 0

    closures = tuple(args.closure_range) if args.closure_range else None
    slices = profile(slice_m=args.slice_mm / 1000.0, closures=closures)
    pad_near, pad_far = geom.PAD_SPAN_FROM_FLANGE_M

    # The window a capture feature ahead of the pads has to live in: from the
    # flange out to where the seated pads begin.
    ahead = [entry for entry in slices if entry["depth_from_flange_m"] < pad_near]
    ahead_best = max((entry["max_half_height_m"] for entry in ahead), default=0.0)
    behind = [entry for entry in slices if entry["depth_from_flange_m"] >= pad_far]
    behind_best = max((entry["max_half_height_m"] for entry in behind), default=0.0)

    print("  depth (mm)   max half-height (mm)   limited by")
    last = None
    for entry in slices:
        value = round(entry["max_half_height_m"] * 1000.0, 1)
        if value == last:
            continue
        last = value
        limiter = entry["limited_by"]
        name = f"{limiter['body']} @ {limiter['finger_joint_rad']:.3f} rad" if limiter else "-"
        print(f"  {entry['depth_from_flange_m']*1000:8.0f}   {value:18.1f}   {name}")

    key_half = getattr(geom, "GRAPPLE_PIN_KEY_HALF_HEIGHT", None)
    print(f"\n  pads seat at {pad_near*1000:.0f}-{pad_far*1000:.0f} mm from the flange")
    print(f"  AHEAD of the pads  (0-{pad_near*1000:.0f} mm): tallest feature is "
          f"{ahead_best*1000:.1f} mm half-height")
    print(f"  BEHIND the pads   ({pad_far*1000:.0f}+ mm): tallest feature is "
          f"{behind_best*1000:.1f} mm half-height")
    if key_half is not None:
        print(f"  the gripped section is {key_half*1000:.1f} mm half-height, so a stop ahead of the "
              f"pads can stand {max(0.0, ahead_best - key_half)*1000:.1f} mm proud of it")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "title": "Room for a pin feature at each depth into the gripper",
                    "evidence_type": "geometric_derivation",
                    "source_envelope": str(ENVELOPE.relative_to(REPO)).replace("\\", "/"),
                    "frame": "wrist_3_link: z from the flange, x the closing axis",
                    "pin_half_width_on_third_axis_m": PIN_HALF_WIDTH_Y,
                    "closure_range_rad": list(closures) if closures else None,
                    "pad_span_from_flange_m": list(geom.PAD_SPAN_FROM_FLANGE_M),
                    "tallest_half_height_ahead_of_pads_m": ahead_best,
                    "tallest_half_height_behind_pads_m": behind_best,
                    "slices": slices,
                    "scope_and_limitations": [
                        "Collision hulls, taken over every sampled closure, so this is the volume the "
                        "hand sweeps rather than its pose at any one instant.",
                        "A feature is assumed centred on the tool axis and 30 mm wide on the third axis.",
                        "Says nothing about whether a feature that fits can be reached by an approach.",
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
