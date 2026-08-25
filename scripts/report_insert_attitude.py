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
        ("artifacts/insert_attitude/lock_defer5.npz",
         "v21time ep1400, stage 0, form lock engaged after 5 control steps"),
        ("artifacts/insert_attitude/lock_defer20.npz",
         "v21time ep1400, stage 0, form lock engaged after 20 control steps"),
        ("artifacts/insert_attitude/lock_mating3.npz",
         "v21time ep1400, stage 0, chain's MATING compliance, zero-shot (policy trained without it)"),
        ("artifacts/insert_attitude/v23lock_s0_seed1070.npz",
         "v23lock ep1400, stage 0, TRAINED on the chain's mating compliance"),
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
            "terminal speed 589.9 mm/s.",
            "That the fling is only an anchoring-timing problem. Deferring engagement to control "
            "step 20 removes the early deaths entirely -- 0% dead inside ten steps against 100% "
            "-- and the attitude is no better: 325.1 mrad, worse than the 84.6 mrad the same "
            "checkpoint reaches with the lock off. The timing was real and it was not the cause.",
        ],
        "what_it_is_not": [
            "Not the reset. evidence/insert_reset_bank.json reports attitude_residual_rad 0.0 at "
            "every station, so the module starts square and the episode takes it to 84.6 mrad.",
            "Not grip slip. Tool-to-handle holds at 12.2 mm with a p95 of 12.48, which is the "
            "pin's own measured 12 mm feed rather than the pads losing it.",
        ],
        "why_the_lock_cannot_simply_be_switched_on": (
            "The insert task's reset places the module INSIDE the destination channel, anywhere "
            "along a 436 mm stroke, so the rails are already constraining it when the latch "
            "engages. A restoring wrench on a module the rails hold fights the rack rather than "
            "the drift it was designed for -- the same effect this project already measured on "
            "extraction, where a latch engaged on capture collapsed the pull from 465 mm to about "
            "25 mm. The chain never does this: it arms the lock only once the module is clear of "
            "the rails, and softens it to the remote-centre mating compliance at the mouth so the "
            "lead-ins can still walk the module. Reproducing that inside the skill is the work, "
            "and it is docs/NEXT_WORK.md T9."
        ),
        "what_it_took_to_reach_the_chains_load_path": [
            "Anchor after the reset settles. engage_after_steps, added here, defaulting to 0.",
            "joint_mode 'fixed', not 'compliant'. With 'compliant' the load path is the explicit "
            "wrench and the mating joint is never installed, so softening re-anchors a transform "
            "engagement set one line earlier -- measured byte-identical to not softening at all.",
            "replicate_physics off. PhysX copies only the first environment's procedurally "
            "authored joint, so envs 1..N get the prim and no usable joint and the run dies with "
            "'Fixed release latch is missing'. configure_base_rail records the same thing. The "
            "skill tasks run replication ON for throughput, which is the structural reason the "
            "chain's load path was not reachable from them.",
        ],
        "zero_shot_on_the_mating_compliance": (
            "With all three in place the episodes survive -- 0% dead inside ten control steps, "
            "against 100% on the transit lock -- and the module gets 20 mm further in, 182.2 mm "
            "short against 202.2 with no lock. Attitude is worse, 113.3 mrad against 84.6, which "
            "is expected and is not a result: this policy was trained on pad contact alone and is "
            "being evaluated on a load path it has never seen. The measurement that matters is a "
            "policy TRAINED under it, which is T9."
        ),
        "mechanism": (
            "Two flat pads on a pin cannot resist a moment about the closing axis -- the same "
            "limit this project measured four independent ways and the reason the chain carries "
            "the module on a form lock. With the lock off, contact with the lead-in rotates the "
            "module and nothing opposes it. The chain reaches 46 mrad at the same seating phase, "
            "and the only difference is that the chain carries the module on the form lock while "
            "the skill trains without one. Strengthening the objective does not substitute for "
            "that, which is what the 400-epoch null result measured."
        ),
        "the_result_once_the_load_path_matched": {
            "what_changed": (
                "Trained on the chain's mating compliance, the median shortfall falls from "
                "202.2 mm to 98.6 mm and 35.5% of episodes now reach the seated plane within the "
                "12 mm depth tolerance, against essentially none before. Lateral improves from "
                "7.10 mm to 4.51 mm. The load path was the blocker for DEPTH, and T9 is confirmed "
                "in that respect."
            ),
            "what_did_not": (
                "Success is still 0.00%. Orientation is the sole remaining failure and it is not "
                "the policy's either."
            ),
            "why_orientation_cannot_be_met_in_this_rack": (
                "Among the 91 episodes that reach seated depth, orientation FLOORS at 56.033 "
                "mrad: p5 is 56.035 and the minimum is 56.033, with a tail to 86.5. A floor is "
                "a surface the module cannot get past, and 2c/L on the channel's unrelieved "
                "lateral throat is 56.396 mrad -- the floor is 0.994 of it. "
                "INSERTION_ORIENTATION_TOLERANCE_RAD is 52.36 mrad, so the angle at which this "
                "throat holds a module is 4.04 mrad OUTSIDE the angle its own acceptance "
                "criterion demands, and a module that merely rests in it cannot pass. The chain "
                "reaches 46 mrad because its form lock holds the module squarer than the rack "
                "does, not because the rack squares it."
            ),
            "correction_to_the_mechanism_named_here": (
                "An earlier version of this report called the band the module 'resting "
                "corner-to-corner against the channel walls at the largest angle the clearance "
                "permits'. The walls are nowhere near it: the destination bay is relieved, so "
                "they admit 76.90 mrad of yaw, and these runs went through play.py "
                "--latch_enabled, which applied the relief a second time and opened them to "
                "97.40. What holds the module is the lead-in throat at the mouth, which is "
                "authored from the rail face and does not move with the relief. Same constant, "
                "so the conclusion is unchanged. evidence/destination_channel_geometry.json and "
                "evidence/RETRACTED.md."
            ),
            "the_rack_change_this_implies_and_what_it_did": (
                "Changed 2026-08-25. The lateral clearance window is 10.35 mm (the lead-ins must "
                "admit the 46 mrad the transit delivers) to 11.78 mm (a resting module must be "
                "inside 52.36 mrad), and GUIDE_CENTER_OFFSET_Y sat above both at 12.689 mm. It "
                "is derived now at the equal-margin point, 11.065 mm, because both values this "
                "project has used sat ON a bound. Tested by moving it: same checkpoint, same "
                "seed, same 256 episodes, the attitude floor goes 56.03 -> 45.75 mrad and "
                "success 0.00% -> 18.85%. evidence/insert_attitude_wall_moved.json."
            ),
        },
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
