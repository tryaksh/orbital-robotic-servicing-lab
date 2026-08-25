"""Why the insert skill does not seat: it is attitude, not creep.

The published account of this skill was that it *creeps* -- still moving at
3.65 mm/s when the clock stops, against 120 mm/s of available authority -- and
that the objective paid for creeping because progress is potential-based and
dawdling was nearly free. The fix that followed from that reading was a time
cost sized against the failure penalty, and it was recorded as untested rather
than refuted because the run had only reached 300 of ~1400 epochs.

It is now trained to 1,400 epochs and it is refuted. The time cost is being
*paid*, not avoided: every episode still spends its whole clock, and the module
still stops the same distance short.

What the same episodes show instead is a geometric impossibility. A rigid part of
length ``L`` entering a channel with ``c`` of relief per side fits only while its
tilt stays under ``2c/L``. For the shipped relief that is
``SERVICE_DELIVERED_ATTITUDE_RAD`` = 20.5 mrad. The module arrives at a median of
84.6 mrad, and the *fifth percentile* is 56.1 -- so not one episode in five
hundred ends inside the angle at which it could enter at all. It is not creeping
toward a seat it might reach. It is wedged at an attitude the channel cannot
accept, and the remaining axial command does nothing.

The reset is not the cause: ``evidence/insert_reset_bank.json`` reports
``attitude_residual_rad`` of 0.0 at every station, so the module starts square
and the episode takes it to 84.6 mrad. Nor is the grip: tool-to-handle holds at
12.2 mm with a p95 of 12.48, which is the pin's own 12 mm feed rather than slip.

CPU only. Reads the ``.npz`` rows two evaluations already wrote.

Usage::

    python scripts/report_insert_attitude.py
"""

from __future__ import annotations

import glob
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "insert_attitude_diagnosis.json"

#: The angle the destination channel admits, 2c/L for the shipped relief. Kept
#: as a literal here so this script needs no simulator import; the source of
#: truth is ``assets.SERVICE_DELIVERED_ATTITUDE_RAD``.
CHANNEL_ADMITS_RAD = 0.0205


def _rows(paths: list[str]) -> tuple[dict[str, int], np.ndarray]:
    stacked, index = [], {}
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        fields = [str(name) for name in archive["fields"]]
        index = {name: position for position, name in enumerate(fields)}
        stacked.append(archive["rows"])
    return index, np.vstack(stacked)


def _summary(paths: list[str], label: str) -> dict:
    index, data = _rows(paths)

    def quantiles(field: str, scale: float) -> dict:
        values = data[:, index[field]] * scale
        return {
            "p5": round(float(np.percentile(values, 5)), 3),
            "median": round(float(np.median(values)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
        }

    return {
        "label": label,
        "episodes": int(len(data)),
        "success_rate": round(float(data[:, index["success"]].mean()), 6),
        "orientation_error_mrad": quantiles("orientation_error_rad", 1000.0),
        "axial_short_mm": quantiles("axial_error_m", 1000.0),
        "lateral_error_mm": quantiles("lateral_error_m", 1000.0),
        "terminal_speed_mm_s": quantiles("blade_linear_velocity_mps", 1000.0),
        "tool_to_handle_mm": quantiles("tool_to_handle_error_m", 1000.0),
        "control_steps_median": float(np.median(data[:, index["control_steps"]])),
        "fraction_dead_within_10_steps": round(
            float((data[:, index["control_steps"]] <= 10).mean()), 4
        ),
        "fraction_inside_channel_admittance": round(
            float((data[:, index["orientation_error_rad"]] < CHANNEL_ADMITS_RAD).mean()), 4
        ),
    }


def main() -> int:
    arms = []
    v20 = sorted(glob.glob(str(ROOT / "artifacts/certify_skills/insert_v20chain_s0_seed*.npz")))
    if v20:
        arms.append(_summary(v20, "v20chain, stage 0, the published 0.00% (time cost at -0.10)"))
    for pattern, label in (
        ("artifacts/insert_v21time/s0_seed1070.npz",
         "v21time ep1400, stage 0, time cost at -0.40 trained to convergence"),
        ("artifacts/insert_v21time/s0_seed1070_lock.npz",
         "v21time ep1400, stage 0, form lock ENGAGED as the chain runs it"),
    ):
        path = ROOT / pattern
        if path.is_file():
            arms.append(_summary([str(path)], label))

    if not arms:
        raise SystemExit("no evaluation rows found; run the stage-0 evaluations first")

    report = {
        "title": "Why the insert skill does not seat: attitude, not creep",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "The insert skill certifies at 0.00% and stops ~204 mm short. The published reading "
            "was that it creeps because the objective made dawdling nearly free. Is that right?"
        ),
        "answer": (
            "No. The time cost trained to convergence changes nothing, and the episodes it "
            "produces are wedged rather than slow: the module ends at a median 84.6 mrad against "
            "a channel that admits 20.5 mrad, with a 5th percentile of 56.1 mrad. Not one episode "
            "in 512 ends inside the angle at which the module could enter the channel at all."
        ),
        "channel_admittance_mrad": CHANNEL_ADMITS_RAD * 1000.0,
        "channel_admittance_derivation": (
            "2c/L, for c = SERVICE_DESTINATION_CHANNEL_RELIEF_M per side and L = BLADE_LENGTH_M. "
            "The same inequality docs/service_interface_spec.md section 6 derives."
        ),
        "arms": arms,
        "what_this_refutes": [
            "That the insert skill is creeping toward a seat it might reach with more time "
            "pressure. The time cost at -0.40, trained to 1,400 epochs, leaves the median "
            "shortfall at 202.2 mm against 203.6 mm without it, and every episode still spends "
            "its whole clock. The cost is paid, not avoided.",
            "That switching the form lock on is a free fix. Engaged on this task the module is "
            "flung: 100% of episodes dead inside ten control steps, orientation 313.6 mrad, "
            "terminal speed 589.9 mm/s. The lock records its transform before the reset writes "
            "the module along the stroke and then fights the difference at up to its 1 kN cap.",
        ],
        "what_it_is_not": [
            "Not the reset. evidence/insert_reset_bank.json reports attitude_residual_rad 0.0 at "
            "every station, so the module starts square and the episode takes it to 84.6 mrad.",
            "Not grip slip. Tool-to-handle holds at 12.2 mm with a p95 of 12.48, which is the "
            "pin's own measured 12 mm feed rather than the pads losing it.",
        ],
        "mechanism": (
            "Two flat pads on a pin cannot resist a moment about the closing axis -- the same "
            "limit this project measured four independent ways and the reason the chain carries "
            "the module on a form lock. With the lock off, contact with the lead-in rotates the "
            "module and nothing opposes it. The objective did not object, because "
            "insertion_misalignment_penalty normalised orientation by 0.15 rad: the SEATED "
            "tolerance, which applies once the channel is already holding the module square. At "
            "that scale an 84.6 mrad module costs 0.08 a step, against the 0.50 the same episode "
            "pays for 7.1 mm of lateral. The objective ranked a fatal attitude below a survivable "
            "offset."
        ),
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "One PPO training seed. The evaluation seeds are held out.",
            "The lock-on arm is a diagnostic of the engagement path, not a certification: it is "
            "one seed and 257 episodes, and it reproduces a failure already recorded.",
            "Stage 0 only. The published skill certifications pool three curriculum stages.",
        ],
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for arm in arms:
        print(
            f"  {arm['label'][:58]:<58} orient median {arm['orientation_error_mrad']['median']:7.1f} mrad"
            f"   short {arm['axial_short_mm']['median']:7.1f} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
