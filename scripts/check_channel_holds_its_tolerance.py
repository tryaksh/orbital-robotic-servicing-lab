"""Can this channel hold this module inside its own acceptance criterion?

**The one question this repository would hand a hardware engineer**, and until now
it has only been answerable by reading a library. A slot is on a drawing. A module
goes in it. There is an acceptance test that decides the module is seated. Ask:

* how square can the slot hold the module once it is resting in there;
* how square does the acceptance test demand it be;
* are those two numbers compatible at all.

If they are not, the bay is asking for something its own geometry forbids, and no
controller closes the gap -- the module is already at rest when the gate is read.
This project spent ten seating checkpoints and three reward functions on a bay in
exactly that state, and the cheapest way to find out is arithmetic before anyone
cuts metal.

**It bites harder in orbit than on a bench.** On the ground gravity and a chamfer
centre a released part for free, so the clearance is an upper bound something else
closes. In zero gravity a module released at an offset stays there, and the
clearance is the whole error budget.

With no arguments it reports the bay this repository actually ships, reading the
scene's own dimensions out of ``assets.py`` and the criterion out of
``insertion.py`` rather than restating either. Give it ``--lateral_clearance_mm``
and ``--vertical_clearance_mm`` to ask about a channel that does not exist yet.

CPU only. No simulator, no GPU, no checkpoint.

Usage::

    python scripts/check_channel_holds_its_tolerance.py
    python scripts/check_channel_holds_its_tolerance.py --relief_mm 0
    python scripts/check_channel_holds_its_tolerance.py \\
        --module_length_mm 300 --lateral_clearance_mm 4 --vertical_clearance_mm 4
    python scripts/check_channel_holds_its_tolerance.py --report evidence/channel_verdict_v1.json
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zero_g_blade_swap.grapple_geometry import SLOT_FLOOR_TOP_Z, SLOT_LIP_BOTTOM_Z
from zero_g_blade_swap.servicing_design import ChannelVerdict, channel_verdict

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap"
ASSETS = TASKS / "assets.py"
INSERTION = TASKS / "mdp" / "insertion.py"
#: The certified geometry check owns the three workcell numbers that are neither
#: a scene dimension nor a criterion -- the guide's own thickness, the measured
#: hand-over attitude and the relief the chain passes. They are read out of that
#: file rather than copied into this one, because a second copy is a second thing
#: to drift and `tests/test_servicing_design.py` binds that file to the library
#: cell by cell.
WORKCELL_CHECK = ROOT / "scripts" / "check_workcell_geometry.py"
#: The distance the module covers between the transit-clear plane and the derived
#: seated plane: 0.1468 m to 0.676 m of module centre. Derived in
#: `evidence/rack_sightline_occlusion_v1.json`, whose `stroke` block records both
#: planes and their derivation, and held fixed at this value by the published
#: `evidence/rack_requirement_sweep_v1.json`. It is NOT
#: `BLADE_INSERTED_POS[0] - TRANSIT_CLEAR_BLADE_CENTRE_X`: the asset's authored
#: inserted pose is 0.75 m and the *derived* seated plane is 0.676 m, so taking
#: the authored one gives 603 mm and a stroke 74 mm too long.
SEATING_STROKE_M = 0.529


def _literal(name: str, path: Path) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not a literal assignment in {path}")


def shipped_bay() -> dict[str, float]:
    """The destination bay's dimensions, read off the scene that builds it."""

    blade = _literal("BLADE_SIZE", ASSETS)
    guide_offset = float(_literal("GUIDE_CENTER_OFFSET_Y", ASSETS))
    guide_thickness = float(_literal("GUIDE_THICKNESS_Y_M", WORKCELL_CHECK))
    return {
        "module_length_m": float(blade[0]),
        "module_width_m": float(blade[1]),
        "module_height_m": float(blade[2]),
        # The face the module runs against, half a guide thickness inboard of the
        # guide body's centre. Reading the centre as the face turns a 0.75 mm
        # channel into a 9.75 mm one.
        "channel_inner_face_half_width_m": abs(guide_offset) - 0.5 * guide_thickness,
        "channel_height_m": SLOT_LIP_BOTTOM_Z - SLOT_FLOOR_TOP_Z,
        "seating_stroke_m": SEATING_STROKE_M,
        "seating_tolerance_rad": float(_literal("INSERTION_ORIENTATION_TOLERANCE_RAD", INSERTION)),
        "lateral_seating_tolerance_m": float(_literal("INSERTION_LATERAL_TOLERANCE_M", INSERTION)),
        "relief_per_side_m": float(_literal("DESTINATION_RELIEF_M", WORKCELL_CHECK)),
        "delivered_attitude_rad": float(_literal("DELIVERED_ATTITUDE_RAD", WORKCELL_CHECK)),
    }


def _print(verdict: ChannelVerdict) -> None:
    mm = 1000.0
    mrad = 1000.0
    print("the module")
    print(f"  length                        {verdict.module_length_m * mm:8.1f} mm")
    print(f"  section                       {verdict.module_width_m * mm:8.1f} x {verdict.module_height_m * mm:.1f} mm")
    print()
    print("the channel")
    print(f"  lateral clearance per side    {verdict.lateral_clearance_per_side_m * mm:8.3f} mm")
    print(f"  vertical clearance per side   {verdict.vertical_clearance_per_side_m * mm:8.3f} mm")
    print(f"  seating stroke                {verdict.seating_stroke_m * mm:8.1f} mm")
    print()
    print("how square it can hold a module resting in it, from 2c/L")
    print(f"  about the vertical axis, yaw  {verdict.resting_yaw_rad * mrad:8.2f} mrad")
    print(f"  about the lateral axis, pitch {verdict.resting_pitch_rad * mrad:8.2f} mrad")
    print(f"  so the gate must accept       {verdict.attitude_the_gate_must_accept_rad * mrad:8.2f} mrad"
          f"   (the looser axis: {verdict.limiting_axis})")
    print(f"  and an offset of              {verdict.offset_the_gate_must_accept_m * mm:8.3f} mm")
    print()
    print("what the acceptance criterion demands")
    print(f"  seated attitude               {verdict.seating_tolerance_rad * mrad:8.2f} mrad")
    print(f"  seated lateral offset         {verdict.lateral_seating_tolerance_m * mm:8.3f} mm")
    print()
    attitude = "fits" if verdict.attitude_is_compatible else "IMPOSSIBLE"
    offset = "fits" if verdict.offset_is_compatible else "IMPOSSIBLE"
    print(f"  attitude margin               {verdict.attitude_margin_rad * mrad:+8.2f} mrad   {attitude}")
    print(f"  offset margin                 {verdict.offset_margin_m * mm:+8.3f} mm     {offset}")
    print()
    if verdict.compatible:
        print("  VERDICT: this channel can hold this module inside its own acceptance criterion.")
    else:
        print("  VERDICT: this channel cannot. A module it is holding may be outside the criterion")
        print("           it uses to decide the module is seated, and no controller fixes that,")
        print("           because the module is already at rest when the gate is read.")
        print("           Narrow the channel, or loosen the criterion, and say which.")
    if verdict.regime is not None:
        print()
        print("getting it in, at the delivered attitude")
        print(f"  hand-over attitude            {(verdict.delivered_attitude_rad or 0.0) * mrad:8.2f} mrad")
        print(f"  2c/theta engagement limit     {(verdict.engagement_depth_limit_m or 0.0) * mm:8.1f} mm")
        print(f"  needs a correcting lead-in    {'yes' if verdict.correcting_lead_in_required else 'no':>8s}")
        print(f"  interface regime              {verdict.regime}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bay = shipped_bay()
    parser.add_argument("--module_length_mm", type=float, default=bay["module_length_m"] * 1000.0)
    parser.add_argument("--module_width_mm", type=float, default=bay["module_width_m"] * 1000.0)
    parser.add_argument("--module_height_mm", type=float, default=bay["module_height_m"] * 1000.0)
    parser.add_argument(
        "--lateral_clearance_mm",
        type=float,
        default=None,
        help="Clearance per side across the channel. Default: the shipped bay's own, from assets.py.",
    )
    parser.add_argument(
        "--vertical_clearance_mm",
        type=float,
        default=None,
        help="Clearance per side between floor and lip. Default: the shipped bay's own.",
    )
    parser.add_argument(
        "--relief_mm",
        type=float,
        default=bay["relief_per_side_m"] * 1000.0,
        help="Per-side relief added to both axes, as the chain passes it. 0 is the design point.",
    )
    parser.add_argument("--seating_stroke_mm", type=float, default=bay["seating_stroke_m"] * 1000.0)
    parser.add_argument(
        "--seated_attitude_tolerance_mrad",
        type=float,
        default=bay["seating_tolerance_rad"] * 1000.0,
        help="What the acceptance test allows. Default: INSERTION_ORIENTATION_TOLERANCE_RAD.",
    )
    parser.add_argument(
        "--seated_offset_tolerance_mm",
        type=float,
        default=bay["lateral_seating_tolerance_m"] * 1000.0,
        help="What the acceptance test allows. Default: INSERTION_LATERAL_TOLERANCE_M.",
    )
    parser.add_argument(
        "--delivered_attitude_mrad",
        type=float,
        default=bay["delivered_attitude_rad"] * 1000.0,
        help="Measured hand-over attitude, for the entry half. Pass 0 to ask the channel question alone.",
    )
    parser.add_argument(
        "--pad_half_bearing_offset_mm",
        type=float,
        default=15.0,
        help="Offset at which a pad still keeps half its face on the capture feature.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write the verdict as evidence JSON.")
    args = parser.parse_args()

    half_width = bay["channel_inner_face_half_width_m"]
    height = bay["channel_height_m"]
    if args.lateral_clearance_mm is not None:
        half_width = args.lateral_clearance_mm / 1000.0 + 0.5 * args.module_width_mm / 1000.0
    if args.vertical_clearance_mm is not None:
        height = 2.0 * args.vertical_clearance_mm / 1000.0 + args.module_height_mm / 1000.0

    delivered = args.delivered_attitude_mrad / 1000.0
    verdict = channel_verdict(
        module_length_m=args.module_length_mm / 1000.0,
        module_width_m=args.module_width_mm / 1000.0,
        module_height_m=args.module_height_mm / 1000.0,
        channel_inner_face_half_width_m=half_width,
        channel_height_m=height,
        relief_per_side_m=args.relief_mm / 1000.0,
        seating_stroke_m=args.seating_stroke_mm / 1000.0,
        seating_tolerance_rad=args.seated_attitude_tolerance_mrad / 1000.0,
        lateral_seating_tolerance_m=args.seated_offset_tolerance_mm / 1000.0,
        delivered_attitude_rad=delivered if delivered > 0.0 else None,
        pad_half_bearing_offset_m=args.pad_half_bearing_offset_mm / 1000.0 if delivered > 0.0 else None,
    )
    _print(verdict)

    if args.report is not None:
        report = {
            "title": "Whether a channel can hold its module inside its own acceptance criterion",
            "evidence_type": "closed_form_geometry_no_simulator",
            "generated_utc": datetime.now(UTC).isoformat(),
            "question": (
                "How square can this channel hold a module resting in it, how square does the "
                "acceptance test demand it be, and are those compatible?"
            ),
            "read_from": [
                "src/zero_g_blade_swap/tasks/blade_swap/assets.py",
                "src/zero_g_blade_swap/tasks/blade_swap/mdp/insertion.py",
            ],
            "verdict": verdict.describe(),
            "scope_and_limitations": [
                "Closed form. No simulator, no policy, no checkpoint.",
                "Quasi-static wedging geometry for a rigid part in a rectangular channel. It bounds "
                "the attitude a module AT REST can take; it says nothing about what a module in "
                "motion does, and this repository has measured that the two differ -- a module is "
                "squared during the stroke rather than carrying its hand-over attitude through it.",
                "The bound is confirmed in functional form and not in coefficient. A swept "
                "measurement over eight clearances fits attitude = 3.609 * relief + 6.217 mrad with "
                "R^2 = 0.99979 against the law's 2/L = 4.444 per mm, so it holds as an upper limit at "
                "six of eight points and is 2.2 to 2.5% optimistic at the two tightest -- which is "
                "the wrong way round, because the design point is at the tight end.",
                "Lateral and vertical are separate limits, not components of one. The looser governs "
                "the acceptance criterion because a resting module may take either.",
            ],
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0 if verdict.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
