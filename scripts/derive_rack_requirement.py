"""Compute a rack requirement from a manipulator's measured performance.

The command-line face of `zero_g_blade_swap.servicing_design`. It takes three
measured numbers about an arm and a gripper, plus the module they have to
service, and prints the rack those imply: the two-sided lateral clearance bound,
the channel that maximises the smaller margin, the family of module
cross-sections that channel accepts, and how accurately the rail has to index.

No simulator, no checkpoint, no asset file, and no number from this repository's
own workcell unless you pass one. The defaults are this project's measured
values, so running it with no arguments reproduces the shipped rack -- which is
the demonstration, not the purpose.

    python scripts/derive_rack_requirement.py --delivered_attitude_rad 0.020

`tests/test_servicing_design.py` asserts the library this wraps lands on the
same numbers `check_workcell_geometry.py` derives for the shipped workcell, over
the whole 36-cell section grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402
from zero_g_blade_swap.servicing_design import ManipulatorPerformance, rack_requirement  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    measured = parser.add_argument_group("measured manipulator performance")
    measured.add_argument(
        "--delivered_attitude_rad",
        type=float,
        default=0.046,
        help="What the manipulator actually hands the insertion over at, not what it is specified to.",
    )
    measured.add_argument(
        "--seating_tolerance_rad",
        type=float,
        default=0.0523599,
        help="The attitude a seated module is accepted at; a property of the interface, not of the arm.",
    )
    measured.add_argument(
        "--pad_half_bearing_offset_m",
        type=float,
        default=0.015,
        help="Offset at which a gripper pad still keeps half its face on the capture feature.",
    )
    module = parser.add_argument_group("the module being serviced")
    module.add_argument("--module_length_m", type=float, default=0.450)
    module.add_argument("--module_width_m", type=float, default=0.130)
    module.add_argument("--module_height_m", type=float, default=0.020)
    module.add_argument("--channel_height_m", type=float, default=0.036)
    module.add_argument(
        "--destination_relief_per_side_m",
        type=float,
        default=0.0046125,
        help="Channel relief on the destination side, credited to the entry criterion.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write the requirement as JSON as well.")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    performance = ManipulatorPerformance(
        delivered_attitude_rad=arguments.delivered_attitude_rad,
        seating_tolerance_rad=arguments.seating_tolerance_rad,
        pad_half_bearing_offset_m=arguments.pad_half_bearing_offset_m,
    )
    requirement = rack_requirement(
        performance,
        module_length_m=arguments.module_length_m,
        module_width_m=arguments.module_width_m,
        module_height_m=arguments.module_height_m,
        channel_height_m=arguments.channel_height_m,
        destination_relief_per_side_m=arguments.destination_relief_per_side_m,
    )
    window = requirement.lateral_clearance_window_m

    print("measured manipulator performance")
    print(f"  delivered attitude          {performance.delivered_attitude_rad * 1000:8.2f} mrad")
    print(f"  seating tolerance           {performance.seating_tolerance_rad * 1000:8.2f} mrad")
    print(f"  pad half bearing offset     {performance.pad_half_bearing_offset_m * 1000:8.3f} mm")
    print()
    print("rack requirement")
    print(
        f"  lateral clearance per side  {window['lower_bound_m'] * 1000:8.3f} to "
        f"{window['upper_bound_m'] * 1000:.3f} mm"
    )
    print(f"  design point                {window['equal_margin_design_point_m'] * 1000:8.3f} mm (equal margin)")
    print(f"  channel inner face          {requirement.channel_inner_face_half_width_m * 1000:8.3f} mm from centre")
    print(f"  rail indexing accuracy      {requirement.rail_indexing_bound_m * 1000:8.3f} mm, geometric bound")
    print()
    shipped = requirement.shipped_section
    verdict = "accepted" if shipped.accepted else f"rejected on {shipped.limiting_criterion}"
    print(f"the module as given: {verdict}")
    print(f"  entry margin                {shipped.entry_margin_m * 1000:8.3f} mm")
    print(f"  grip margin                 {shipped.grip_margin_m * 1000:8.3f} mm")
    print()
    print(f"cross-sections this rack accepts ({len(requirement.accepted_sections)} of 36 evaluated)")
    for section in requirement.accepted_sections:
        print(
            f"  {section.width_m * 1000:5.0f} x {section.height_m * 1000:4.0f} mm   "
            f"entry {section.entry_margin_m * 1000:7.3f} mm   grip {section.grip_margin_m * 1000:7.3f} mm"
        )

    if arguments.report is not None:
        payload = requirement.describe()
        payload["source_revision"] = git_source_revision(ROOT)
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
