"""Why the insert skill does not seat: attitude it cannot deliver, not creep.

Two results, one solid and one a correction.

**The time cost is refuted.** The published account of this skill was that it
*creeps* -- still moving at 3.65 mm/s when the clock stops, against 120 mm/s of
authority -- and the fix that followed was a time cost sized to make a full clock
cost 12, below the 15 that failing costs. Trained to convergence at 1,400 epochs
it moves the median shortfall by 1.4 mm and every episode still spends its whole
clock. The cost is paid, not avoided.

**What the episodes show instead is attitude.** The module ends at a median of
84.6 mrad against ``INSERTION_ORIENTATION_TOLERANCE_RAD`` = 52.4 mrad, the angle
a seated module must be inside for the insertion to count. Essentially every
episode is outside it, by about 1.6x. The reset is not the cause --
``evidence/insert_reset_bank.json`` reports ``attitude_residual_rad`` of 0.0 at
every station, so the module starts square -- and nor is the grip, which holds
at 12.2 mm with a p95 of 12.48, the pin's own measured feed.

**The correction.** An earlier version of this report compared 84.6 mrad against
``SERVICE_DELIVERED_ATTITUDE_RAD`` (20.5 mrad) and called that the angle at which
the module can enter the channel at all, concluding the objective had been
scaling orientation against the wrong tolerance. That was wrong twice over:
20.5 mrad is what a module *settles* at after the lead-ins have worked on it --
the arm delivers 63 mrad at the mouth -- and the objective's original 0.15 rad
scale is calibrated about right against the 52.4 mrad it should be. The tighter
scale was trained for 400 epochs and moved orientation from 84.61 to 84.58 mrad.
Reverted; see ``evidence/RETRACTED.md``.

That null result is the useful part. A 7x stronger angular penalty that does not
move the angle says attitude is not the policy's to give: two flat pads on a pin
cannot resist a moment about the closing axis, which this project has measured
four independent ways, and the chain reaches 46 mrad only because it carries the
module on a form lock. The blocker is the load path, not the reward -- NEXT_WORK
T9 rather than T2.

CPU only. Reads the ``.npz`` rows the evaluations already wrote.

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

#: The orientation a seated module must be inside for the insertion to count.
#: ``mdp.insertion.INSERTION_ORIENTATION_TOLERANCE_RAD``, kept as a literal so
#: this script needs no simulator import.
#:
#: **Not** ``SERVICE_DELIVERED_ATTITUDE_RAD`` (20.5 mrad). An earlier version of
#: this report compared against that and called it the angle at which the module
#: can enter at all. It is not: it is what a module *settles* at after the
#: lead-ins have worked on it, and the arm delivers 63 mrad at the mouth. See
#: evidence/RETRACTED.md.
SUCCESS_ORIENTATION_RAD = 0.0523599


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
        "fraction_inside_success_orientation": round(
            float((data[:, index["orientation_error_rad"]] < SUCCESS_ORIENTATION_RAD).mean()), 4
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
        ("artifacts/insert_attitude/probe_ep400.npz",
         "v22attitude ep400, stage 0, orientation penalty 7x stronger (RETRACTED scale)"),
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
            "No. The time cost trained to convergence changes nothing. What the episodes show "
            "is attitude: the module ends at a median 84.6 mrad against the 52.4 mrad a seated "
            "module must be inside, so essentially every episode is outside the orientation "
            "tolerance by about 1.6x. A 7x stronger angular penalty, trained 400 epochs, moved "
            "that by 0.03 mrad -- so the angle is not the policy's to give through pad contact."
        ),
        "success_orientation_tolerance_mrad": SUCCESS_ORIENTATION_RAD * 1000.0,
        "comparison_note": (
            "Compared against INSERTION_ORIENTATION_TOLERANCE_RAD, the success criterion. An "
            "earlier version of this report compared against SERVICE_DELIVERED_ATTITUDE_RAD "
            "(20.5 mrad) and called it an entry limit; that is the SETTLED attitude, not an entry "
            "limit, and the claim is retracted in evidence/RETRACTED.md."
        ),
        "arms": arms,
        "what_this_refutes": [
            "That the insert skill is creeping toward a seat it might reach with more time "
            "pressure. The time cost at -0.40, trained to 1,400 epochs, leaves the median "
            "shortfall at 202.2 mm against 203.6 mm without it, and every episode still spends "
            "its whole clock. The cost is paid, not avoided.",
            "That the objective was mis-scaled. Normalising orientation by 20.5 mrad instead of "
            "0.15 rad -- a 7x stronger angular penalty -- moved the measured orientation from "
            "84.61 to 84.58 mrad over 400 epochs. Retracted; see evidence/RETRACTED.md.",
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
            "module and nothing opposes it. The chain reaches 46 mrad at the same seating phase, "
            "and the only difference is that the chain carries the module on the form lock while "
            "the skill trains without one. Strengthening the objective does not substitute for "
            "that, which is what the 400-epoch null result measured."
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
