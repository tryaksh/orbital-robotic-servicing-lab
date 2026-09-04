"""The rack requirement as a function of the manipulator, which is the whole claim.

The design derivation is easy to state and hard to believe until it is swept:
change one measured number about the arm and watch the rack change. This emits
that sweep as data a figure can be drawn from, with no simulator, no checkpoint
and no asset file.

The independent variable is the attitude the manipulator hands the insertion over
at, because it is the one this project measured most carefully and the one a
servicer can actually improve. Everything else is held at the shipped workcell's
values, so the curve passes through the rack that exists.

Four dependent quantities, and they do not move together:

* the **lateral clearance window** -- lower bound from what the channel must
  admit, upper bound from what a resting module may not exceed. The window
  *widens* as the arm improves, because only its lower bound moves;
* the **equal-margin design point** inside that window;
* the **rail indexing bound**, which follows from the pad's reach once the
  channel is placed. Note the sign: a better arm earns a *looser* rail, because
  the channel it needs is tighter and a tighter channel leaves the pads more of
  their reach;
* how many of thirty-six candidate module cross-sections the rack accepts, which
  is the sentence a designer actually wants.

The last column is `requires_a_correcting_lead_in`, and it is where the sweep
earns its keep: below about 42 mrad this bay no longer needs a funnel at all,
which is a design decision -- a part deleted -- falling out of an arm's measured
performance.

**Read the rail column with the caveat in `servicing_design`'s own docstring.**
It is a static bound, and this repository has measured that the rail axis fails a
*settling* condition at a threshold three times looser. The number is safe; it is
not the reason the axis binds.

    python scripts/sweep_rack_requirement.py --report evidence/<name>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402
from zero_g_blade_swap.servicing_design import (  # noqa: E402
    ManipulatorPerformance,
    rack_requirement,
    requires_a_correcting_lead_in,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--attitudes_mrad",
        type=float,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 35, 40, 46, 50, 52],
        help="Hand-over attitudes to sweep. 46 is this project's measured value.",
    )
    parser.add_argument("--seating_tolerance_rad", type=float, default=0.0523599)
    parser.add_argument("--pad_half_bearing_offset_m", type=float, default=0.015)
    parser.add_argument("--module_length_m", type=float, default=0.450)
    parser.add_argument("--module_width_m", type=float, default=0.130)
    parser.add_argument("--module_height_m", type=float, default=0.020)
    parser.add_argument("--channel_height_m", type=float, default=0.036)
    parser.add_argument("--seating_stroke_m", type=float, default=0.529)
    parser.add_argument("--destination_relief_per_side_m", type=float, default=0.0046125)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="Also write the rows as CSV, for plotting.")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    rows = []
    for attitude_mrad in arguments.attitudes_mrad:
        attitude = attitude_mrad / 1000.0
        if attitude >= arguments.seating_tolerance_rad:
            # A manipulator that hands over outside the interface's own acceptance
            # is not a rack problem, and the library refuses it. Record the refusal
            # rather than dropping the point, because where the curve ends is part
            # of the answer.
            rows.append(
                {
                    "delivered_attitude_mrad": attitude_mrad,
                    "admissible": False,
                    "why": "hands over outside the interface's seating tolerance; no clearance fixes that",
                }
            )
            continue
        performance = ManipulatorPerformance(
            delivered_attitude_rad=attitude,
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
        lead_in = requires_a_correcting_lead_in(
            performance,
            seating_stroke_m=arguments.seating_stroke_m,
            clearance_per_side_m=window["equal_margin_design_point_m"],
        )
        rows.append(
            {
                "delivered_attitude_mrad": attitude_mrad,
                "admissible": True,
                "clearance_lower_mm": 1000.0 * window["lower_bound_m"],
                "clearance_upper_mm": 1000.0 * window["upper_bound_m"],
                "clearance_window_width_mm": 1000.0 * window["window_width_m"],
                "design_point_mm": 1000.0 * window["equal_margin_design_point_m"],
                "rail_indexing_bound_mm": 1000.0 * requirement.rail_indexing_bound_m,
                "accepted_sections_of_36": len(requirement.accepted_sections),
                "requires_a_correcting_lead_in": bool(lead_in["required"]),
                "engagement_depth_limit_mm": 1000.0 * float(lead_in["engagement_depth_limit_m"]),
            }
        )

    header = (
        f"{'attitude':>9} {'clearance window mm':>21} {'width':>7} {'design pt':>10} "
        f"{'rail mm':>8} {'sections':>9} {'lead-in':>8}"
    )
    print(header)
    for row in rows:
        if not row["admissible"]:
            print(f"{row['delivered_attitude_mrad']:>7.0f}   -- {row['why']}")
            continue
        print(
            f"{row['delivered_attitude_mrad']:>7.0f}   "
            f"{row['clearance_lower_mm']:8.3f} to {row['clearance_upper_mm']:6.3f} "
            f"{row['clearance_window_width_mm']:7.3f} {row['design_point_mm']:10.3f} "
            f"{row['rail_indexing_bound_mm']:8.3f} {row['accepted_sections_of_36']:>9} "
            f"{'yes' if row['requires_a_correcting_lead_in'] else 'no':>8}"
        )

    if arguments.report is not None:
        revision = git_source_revision(ROOT)
        if revision.get("available") and revision.get("dirty"):
            raise SystemExit("refusing to write evidence from a dirty tracked worktree; commit first")
        payload = {
            "title": "The rack requirement as a function of the manipulator's delivered attitude",
            "evidence_type": "closed_form_design_sweep",
            "what_this_is": (
                "One measured number about the arm is swept and the rack it implies is recomputed. No "
                "simulator, no checkpoint, no asset file. The shipped workcell is the 46 mrad row."
            ),
            "held_fixed": {
                "seating_tolerance_rad": arguments.seating_tolerance_rad,
                "pad_half_bearing_offset_m": arguments.pad_half_bearing_offset_m,
                "module_m": {
                    "length": arguments.module_length_m,
                    "width": arguments.module_width_m,
                    "height": arguments.module_height_m,
                },
                "channel_height_m": arguments.channel_height_m,
                "seating_stroke_m": arguments.seating_stroke_m,
                "destination_relief_per_side_m": arguments.destination_relief_per_side_m,
            },
            "scope": [
                "Closed form only. Nothing here was simulated and nothing was measured on hardware.",
                (
                    "Every bound swept here is static. The rail column in particular is a bound on a pad "
                    "sliding off the pin, and this repository has measured that the rail axis actually "
                    "fails a settling condition at a threshold about three times looser."
                ),
            ],
            "rows": rows,
            "source_revision": revision,
        }
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {arguments.report}")

    if arguments.csv is not None:
        usable = [row for row in rows if row["admissible"]]
        arguments.csv.parent.mkdir(parents=True, exist_ok=True)
        with arguments.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(usable[0]))
            writer.writeheader()
            writer.writerows(usable)
        print(f"wrote {arguments.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
