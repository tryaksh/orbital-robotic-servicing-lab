"""Score each analytical criterion against the failure it predicts, not the pooled rate.

**The boundary comparison has been asking the wrong question, and the nominal
point says so on its own.** At 64 environments the nominal chain scores 54.7%,
and 27 of its 29 failures reach the final phase with the form lock engaged and
miss the 2.5 mm seating gate by about a millimetre. So roughly two episodes in
five are already lost, at the design point, to a mode that no serviceability
criterion claims to predict. A boundary point has to move the pooled rate by
more than that before a Wilson interval can separate it -- which is why five of
seven points came back "mismatch" while their *mechanisms* were behaving exactly
as the geometry said they would.

Each criterion in `check_workcell_geometry.py` predicts a specific failure, and
the episode rows record enough to count it:

* the **grip** criterion -- a module in the corner of the *source* channel has
  to stay inside the offset at which a pad keeps half its face on the pin --
  predicts the grip being lost during the pull. That is an episode that times
  out in capture, seat or extract and never delivers the module at all.
* the **entry** criterion -- the module has to reach the seated plane at the
  attitude the transit hands over at -- predicts a jam. That is an episode that
  reaches the insert phase and stops there, short of the seated plane.
* neither criterion predicts an episode that arrives, seats, and misses the
  terminal gate. That is the chain's own precision, and it is the mode that
  dominates the nominal cohort.

So this scores each point on the rate of the mode its own criterion predicts,
beside the pooled rate the previous protocol used, and reports both. Nothing is
re-run and no tolerance moves: the same episodes are counted a second way.

**And a rate is not the only thing an episode records.** The grip criterion is a
bound on how far a pad may slide off the pin, so where it is violated the
episodes that fail should carry a *larger tool-to-pin offset* than the ones that
succeed -- a statement about a measured quantity, not about a count. That
signature is reported per point, and at 64 episodes it separates where the rate
does not: the two points the criterion flags are the only two where it is
positive.

The criteria themselves are recomputed from current source through
`check_workcell_geometry.py`, so this cannot drift away from the geometry it
claims to be testing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

# 0 capture, 1 seat, 2 extract, 3 transit, 4 insert, 5 done -- the driver's own
# ordering, repeated here because these rows are read without importing it.
PHASE_NAMES = ("capture", "seat", "extract", "transit", "insert", "done")
DELIVERY_PHASES = (0, 1, 2)
TRANSIT_PHASE = 3
INSERT_PHASE = 4
DONE_PHASE = 5


def wilson_95(successes: int, trials: int) -> dict[str, float]:
    """The interval this repository quotes everywhere else, at the same z."""

    if trials == 0:
        return {"low": 0.0, "high": 1.0}
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return {"low": round(max(0.0, centre - half), 6), "high": round(min(1.0, centre + half), 6)}


def separated(point: dict[str, object], nominal: dict[str, object]) -> bool:
    """A point's mode rate is separated when its interval clears nominal's.

    Same rule the pooled protocol uses, applied to the mode instead of to the
    pooled rate: the point's lower bound above nominal's upper bound. The
    direction is reversed relative to a success rate because a *failure* mode
    going up is the loss.
    """

    return float(point["wilson_95"]["low"]) > float(nominal["wilson_95"]["high"])  # type: ignore[index]


def decompose(rows: np.ndarray, fields: list[str]) -> dict[str, object]:
    """Split one point's episodes into the three modes and count each."""

    column = {name: rows[:, fields.index(name)] for name in fields}
    success = column["success"] > 0.5
    timed_out = column["timed_out_in_phase"].astype(int)
    reached = column["reached_phase"].astype(int)

    # Partition on where the episode stopped, not on how far it got, so a
    # transit timeout cannot be filed as a jam. No point measured so far
    # produces one, and that is exactly why it would go unnoticed.
    lost_before_delivery = np.isin(timed_out, DELIVERY_PHASES)
    lost_in_transit = timed_out == TRANSIT_PHASE
    jammed_in_the_bay = (~success) & ~(lost_before_delivery | lost_in_transit) & (reached <= INSERT_PHASE)
    missed_the_gate = (
        (~success) & ~(lost_before_delivery | lost_in_transit | jammed_in_the_bay) & (reached >= DONE_PHASE)
    )

    episodes = int(len(success))
    counted = int(
        success.sum()
        + lost_before_delivery.sum()
        + lost_in_transit.sum()
        + jammed_in_the_bay.sum()
        + missed_the_gate.sum()
    )
    if counted != episodes:
        raise RuntimeError(f"the modes and the successes do not partition the cohort: {counted} of {episodes}")

    grip_error = column["grip_error_m"]
    return {
        "episodes": episodes,
        "successes": int(success.sum()),
        "success_rate": round(float(success.mean()), 6),
        "modes": {
            "lost_before_delivery": {
                "what_it_is": "timed out in capture, seat or extract; the module never reached the destination",
                "predicted_by": "grip",
                "count": int(lost_before_delivery.sum()),
                "rate": round(float(lost_before_delivery.mean()), 6),
                "wilson_95": wilson_95(int(lost_before_delivery.sum()), episodes),
                "timed_out_phase_counts": {
                    PHASE_NAMES[phase]: int((timed_out == phase).sum()) for phase in DELIVERY_PHASES
                },
            },
            "lost_in_transit": {
                "what_it_is": "timed out carrying the module between bays; predicted by neither criterion",
                "predicted_by": None,
                "count": int(lost_in_transit.sum()),
                "rate": round(float(lost_in_transit.mean()), 6),
                "wilson_95": wilson_95(int(lost_in_transit.sum()), episodes),
            },
            "jammed_in_the_bay": {
                "what_it_is": "reached the insert phase and stopped there, short of the seated plane",
                "predicted_by": "entry",
                "count": int(jammed_in_the_bay.sum()),
                "rate": round(float(jammed_in_the_bay.mean()), 6),
                "wilson_95": wilson_95(int(jammed_in_the_bay.sum()), episodes),
            },
            "missed_the_terminal_gate": {
                "what_it_is": "arrived and seated, then missed the 2.5 mm / 52.4 mrad terminal gate",
                "predicted_by": None,
                "count": int(missed_the_gate.sum()),
                "rate": round(float(missed_the_gate.mean()), 6),
                "wilson_95": wilson_95(int(missed_the_gate.sum()), episodes),
            },
        },
        "median_grip_error_m": {
            "successful": round(float(np.median(grip_error[success])), 6) if success.any() else None,
            "failing": round(float(np.median(grip_error[~success])), 6) if (~success).any() else None,
        },
        # The grip criterion's own mechanism, as a measured quantity rather than
        # a count: how much further off the pin a failing episode's pads sat.
        "grip_signature_mm": (
            round(1000.0 * float(np.median(grip_error[~success]) - np.median(grip_error[success])), 4)
            if success.any() and (~success).any()
            else None
        ),
    }


def load_geometry(script: Path):
    """Recompute the criteria from current source rather than from a report."""

    spec = importlib.util.spec_from_file_location("workcell_geometry", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["workcell_geometry"] = module
    spec.loader.exec_module(module)
    return module


def criteria_for(module, lateral_m: float, vertical_m: float) -> dict[str, object]:
    """Apply both closed-form criteria to one (lateral, vertical) half-gap pair.

    Every sweep point reduces to such a pair: a module section changes it by
    changing the module, a rack clearance changes it by moving the guides, and
    the two are the same question asked of the same channel.
    """

    envelope = module.section_envelope()
    needed = float(envelope["required_half_gap_m"])
    relief = float(envelope["destination_relief_per_side_m"])
    pad_bound = float(envelope["pad_half_bearing_offset_m"])

    corner = float(math.hypot(lateral_m, vertical_m)) if min(lateral_m, vertical_m) > 0.0 else float("inf")
    entry_margin = min(lateral_m, vertical_m) + relief - needed
    grip_margin = pad_bound - corner
    return {
        "lateral_half_gap_m": round(lateral_m, 6),
        "vertical_half_gap_m": round(vertical_m, 6),
        "required_half_gap_m": needed,
        "destination_relief_per_side_m": relief,
        "pad_half_bearing_offset_m": pad_bound,
        "channel_corner_m": None if corner == float("inf") else round(corner, 6),
        "entry_margin_m": round(entry_margin, 6),
        "grip_margin_m": None if corner == float("inf") else round(grip_margin, 6),
        "entry_admissible": bool(entry_margin >= -1.0e-6),
        "grip_admissible": bool(corner <= pad_bound + 1.0e-6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, default=ROOT / "artifacts" / "robustness64")
    parser.add_argument("--nominal", default="nominal")
    parser.add_argument(
        "--compare_dir",
        type=Path,
        default=None,
        help=(
            "A second sweep directory holding the same point names under one changed flag. Points "
            "found there are decomposed the same way and reported beside the primary arm, which is "
            "how the clearance re-measurement keeps its losing arm."
        ),
    )
    parser.add_argument("--compare_label", default="compared_arm")
    parser.add_argument("--report", type=Path, default=ROOT / "evidence" / "boundary_failure_modes_v1.json")
    arguments = parser.parse_args()

    module = load_geometry(ROOT / "scripts" / "check_workcell_geometry.py")
    guide_inner_face = float(module.section_envelope()["guide_inner_face_half_width_m"])
    channel_height = float(module.section_envelope()["channel_height_m"])
    shipped = [float(value) for value in module._literal("BLADE_SIZE")]
    shipped_width, shipped_height = shipped[1], shipped[2]

    def section_gaps(width_m: float, height_m: float) -> tuple[float, float]:
        return guide_inner_face - 0.5 * width_m, 0.5 * (channel_height - height_m)

    def clearance_gaps(clearance_m: float) -> tuple[float, float]:
        # Moving the guides changes the lateral gap and leaves the channel's own
        # height, and therefore the vertical gap, exactly where it was.
        return clearance_m, 0.5 * (channel_height - shipped_height)

    nominal_lateral, nominal_vertical = section_gaps(shipped_width, shipped_height)

    # Each sweep point, and the (lateral, vertical) pair it presents to the
    # channel. The two base points do not change the channel at all -- that is
    # the finding they carry, and it is stated rather than encoded.
    layout: dict[str, dict[str, object]] = {
        "nominal": {"gaps": (nominal_lateral, nominal_vertical), "dimension": "reference"},
        "section_120x16": {"gaps": section_gaps(0.120, 0.016), "dimension": "module_section"},
        "section_140x26": {"gaps": section_gaps(0.140, 0.026), "dimension": "module_section"},
        "rack_lat_6mm": {"gaps": clearance_gaps(0.006), "dimension": "rack_clearance"},
        "rack_lat_16mm": {"gaps": clearance_gaps(0.016), "dimension": "rack_clearance"},
        "base_x_-0.70": {"gaps": (nominal_lateral, nominal_vertical), "dimension": "base_offset"},
        "base_y_+10mm": {"gaps": (nominal_lateral, nominal_vertical), "dimension": "base_offset"},
    }

    def read_points(directory: Path) -> dict[str, dict[str, object]]:
        """Decompose every point present, whether or not the layout names it.

        The base_y ladder adds rungs that did not exist when this was written,
        and a rung is still a measurement. An unnamed point inherits the nominal
        channel, because a point that does not change the channel is exactly
        what those are.
        """

        found: dict[str, dict[str, object]] = {}
        for archive in sorted(directory.glob("*.npz")):
            name = archive.stem
            entry = layout.get(name)
            if entry is None:
                if not name.startswith("base_"):
                    continue
                entry = {"gaps": (nominal_lateral, nominal_vertical), "dimension": "base_offset"}
            loaded = np.load(archive, allow_pickle=True)
            fields = [str(value) for value in loaded["fields"]]
            record = decompose(loaded["rows"], fields)
            lateral, vertical = entry["gaps"]  # type: ignore[misc]
            record["dimension"] = entry["dimension"]
            record["analytical"] = criteria_for(module, float(lateral), float(vertical))
            found[name] = record
        return found

    points = read_points(arguments.sweep_dir)
    if arguments.nominal not in points:
        raise FileNotFoundError(f"{arguments.nominal}.npz is not in {arguments.sweep_dir}")

    nominal = points[arguments.nominal]
    verdicts: dict[str, dict[str, object]] = {}
    for name, record in points.items():
        if name == arguments.nominal:
            continue
        analytical = record["analytical"]
        rows = []
        for criterion, mode in (("entry", "jammed_in_the_bay"), ("grip", "lost_before_delivery")):
            admissible = bool(analytical[f"{criterion}_admissible"])  # type: ignore[index]
            point_mode = record["modes"][mode]  # type: ignore[index]
            nominal_mode = nominal["modes"][mode]  # type: ignore[index]
            raised = separated(point_mode, nominal_mode)
            rows.append(
                {
                    "criterion": criterion,
                    "margin_m": analytical[f"{criterion}_margin_m"],  # type: ignore[index]
                    "analytically_admissible": admissible,
                    "predicted_mode": mode,
                    "mode_rate": point_mode["rate"],
                    "mode_wilson_95": point_mode["wilson_95"],
                    "nominal_mode_rate": nominal_mode["rate"],
                    "nominal_mode_wilson_95": nominal_mode["wilson_95"],
                    "mode_separated_from_nominal": raised,
                    "grip_signature_mm": record["grip_signature_mm"],
                    "signature_present": (
                        None
                        if criterion != "grip" or record["grip_signature_mm"] is None
                        else bool(float(record["grip_signature_mm"]) > 0.0)
                    ),
                    # An inadmissible point supports the criterion by showing the
                    # mode; an admissible one supports it by not showing it.
                    "supports_the_criterion": (admissible and not raised) or (not admissible and raised),
                }
            )
        verdicts[name] = {
            "dimension": record["dimension"],
            "pooled_success_rate": record["success_rate"],
            "criteria": rows,
        }

    compared: dict[str, dict[str, object]] = {}
    if arguments.compare_dir is not None:
        for name, record in read_points(arguments.compare_dir).items():
            primary = points.get(name)
            record["primary_arm_success_rate"] = primary["success_rate"] if primary else None
            if primary is not None:
                record["mode_deltas"] = {
                    mode: round(
                        float(record["modes"][mode]["rate"]) - float(primary["modes"][mode]["rate"]), 6
                    )
                    for mode in record["modes"]
                }
            compared[name] = record

    report = {
        "title": "Serviceability criteria scored against the failure each one predicts",
        "evidence_type": "analytical_criteria_against_simulated_failure_modes",
        "what_this_is": (
            "The same 64-environment sweep episodes, counted a second way. Each closed-form criterion "
            "names a failure mode; the mode's rate at each point is compared with its rate at nominal "
            "on the same Wilson rule the pooled protocol uses. Nothing was re-run and no tolerance moved."
        ),
        "why": (
            "At nominal, 27 of 29 failures reach the final phase and miss the terminal gate, so about "
            "two episodes in five are lost at the design point to a mode no serviceability criterion "
            "claims to predict. A pooled-rate comparison asks every boundary point to clear that noise "
            "floor before it can separate."
        ),
        "scope": [
            "Simulation only. No result here was produced on real hardware.",
            "This is a second reading of existing episodes, not a new measurement.",
            (
                "The two base-offset points do not change the channel, so neither channel criterion "
                "applies to them; they are reported to show what mode they do produce."
            ),
            (
                "A mode rate at 64 episodes has a wide interval. A point whose mode moves in the "
                "predicted direction without separating is reported as unseparated, not as support."
            ),
        ],
        "mode_definitions": {
            "lost_before_delivery": "timed out in capture, seat or extract; predicted by the grip criterion",
            "jammed_in_the_bay": "reached insert and stopped short of the seated plane; predicted by the entry criterion",
            "lost_in_transit": "timed out carrying the module between bays; predicted by neither",
            "missed_the_terminal_gate": "arrived and seated, then missed the terminal gate; predicted by neither",
        },
        "nominal_point": arguments.nominal,
        "points": points,
        "verdicts": verdicts,
        "compared_arm": {
            "label": arguments.compare_label,
            "directory": str(arguments.compare_dir) if arguments.compare_dir else None,
            "points": compared,
        },
        "source_revision": git_source_revision(ROOT),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'point':<16} {'rate':>6} {'lost':>6} {'jam':>6} {'gate':>6}   analytical")
    for name, record in points.items():
        modes = record["modes"]
        analytical = record["analytical"]
        marks = []
        if not analytical["entry_admissible"]:
            marks.append(f"entry short {abs(float(analytical['entry_margin_m'])) * 1000:.2f} mm")
        if not analytical["grip_admissible"]:
            marks.append(f"grip over {abs(float(analytical['grip_margin_m'])) * 1000:.2f} mm")
        print(
            f"{name:<16} {record['success_rate']:6.3f} "
            f"{modes['lost_before_delivery']['rate']:6.3f} "
            f"{modes['jammed_in_the_bay']['rate']:6.3f} "
            f"{modes['missed_the_terminal_gate']['rate']:6.3f}   "
            f"{'; '.join(marks) if marks else 'admissible'}"
        )
    print()
    print(f"{'point':<16} {'grip signature mm':>18}   (failing minus successful tool-to-pin offset)")
    for name, record in points.items():
        signature = record["grip_signature_mm"]
        flagged = "" if record["analytical"]["grip_admissible"] else "   <- grip criterion violated"
        print(f"{name:<16} {float(signature):>18.2f}{flagged}" if signature is not None else f"{name:<16}{'n/a':>19}")
    print()
    for name, verdict in verdicts.items():
        for row in verdict["criteria"]:
            state = "supports" if row["supports_the_criterion"] else "does not support"
            print(
                f"{name:<16} {row['criterion']:<6} admissible={str(row['analytically_admissible']):<5} "
                f"mode {row['mode_rate']:.3f} vs nominal {row['nominal_mode_rate']:.3f} "
                f"separated={str(row['mode_separated_from_nominal']):<5} -> {state}"
            )
    if compared:
        print()
        print(f"{arguments.compare_label}:")
        print(f"{'point':<16} {'rate':>6} {'lost':>6} {'jam':>6} {'gate':>6}   (delta on the primary arm)")
        for name, record in compared.items():
            modes = record["modes"]
            deltas = record.get("mode_deltas", {})
            print(
                f"{name:<16} {record['success_rate']:6.3f} "
                f"{modes['lost_before_delivery']['rate']:6.3f} "
                f"{modes['jammed_in_the_bay']['rate']:6.3f} "
                f"{modes['missed_the_terminal_gate']['rate']:6.3f}   "
                f"jam {deltas.get('jammed_in_the_bay', 0.0):+.3f}  lost {deltas.get('lost_before_delivery', 0.0):+.3f}"
            )
    print(f"\nwrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
