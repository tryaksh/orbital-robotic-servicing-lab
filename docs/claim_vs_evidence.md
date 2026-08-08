# Claim versus evidence

One page for a skeptical reader. Every row separates what was *measured* from
what is *not* established. All results are simulation results.

## The problem this addresses

Compute in orbit is limited less by launch mass than by what happens after
launch. A satellite or orbital compute platform cannot be serviced: a failed
accelerator card, storage module, or power board is a permanent capacity loss
for the life of the vehicle. Terrestrial data centers absorb hardware failure by
having a technician swap a hot-plug blade in minutes. Orbital platforms have no
technician.

Robotic module replacement is therefore a precondition for orbital compute that
scales past single-shot hardware. The hard part is not the arm; it is
contact-rich insertion under uncertain payload mass, contact friction, mounting
compliance, and pose error, with no operator to recover a jam.

This repository studies the narrowest useful slice of that problem: **learned
insertion of an already-secured replacement blade into a rack in microgravity**.

## What is measured

| Claim | Evidence | Scope and caveats |
| --- | --- | --- |
| A PPO policy inserts a secured blade with no rack contact, from three reset distances | 9,086 / 9,086 deterministic episodes, seeds 1060/2060/3060, Wilson 95% lower bound 0.9996 | `evidence/rigid_grasp_l0_ep700_certification.json`. Simulation only. Held-out *evaluation* seeds, one training seed |
| The policy still succeeds when the rack side rails are physically collidable and initial pose error is doubled | 9,014 / 9,014 deterministic episodes, seeds 1061/2061/3061, Wilson 95% lower bound 0.9996 | `evidence/rigid_grasp_l1_ep1200_certification.json`. Fine-tuned 500 PPO epochs from the Level-0 checkpoint |
| Contact fine-tuning improved precision, it did not merely preserve success | Stage-0 terminal axial error fell from 4.15 mm mean (L0) to 1.65 mm (L1) | Same reports. Distribution is bounded by the success criterion, see below |
| Terminal-state evidence cannot be corrupted by the simulator's automatic reset | `TerminalMetricsMixin` snapshots each episode inside `_reset_idx`; unit tests assert the captured value differs from the post-reset value | `tests/test_terminal_metrics.py` |
| No episode ended in numerical or physics instability | Zero non-finite and zero mount-instability terminations across 18,100 episodes | Categorization prefers instability over success when both fire in one control step |
| Simulated cycle time is measured, not estimated | Median 7.9 s at full reset distance, 1.2 s at near distance, at 30 Hz control | Simulated time. Does not include perception, approach, grasp, or extraction |
| Training is reproducible on consumer hardware | 512 environments on a 12 GB laptop GPU; 500 PPO epochs in about 22 minutes at 5,000-8,500 environment-steps/s | Isaac Sim 5.1 publishes a 16 GB VRAM minimum; this runs under it via benchmark-driven environment counts |

## What is not established

| Not claimed | Why |
| --- | --- |
| Learned grasping | The blade is held by a PhysX fixed joint standing in for an already-secured grasp. The real Robotiq pad/handle contact task failed its axial pull gate. The near-zero tool-to-handle error in the reports is a property of that joint, not of a grip |
| Sim2Real transfer | No real UR10e, hardware-in-the-loop rig, wrist force/torque sensor, calibrated camera, orbital acceleration data, or radiation dataset has been used |
| Accuracy independent of the success criterion | Because every certification episode succeeded, the terminal error distribution is bounded by the success box. It shows where inside tolerance the policy lands, not error it was free to exceed |
| Robustness to payload mass or rail stiction | Level 2 (5-15 kg mass, 1.5 mm clearance) is in training. Level 3 stiction reaches valid geometry but cannot settle below velocity limits and is documented as blocked, not hidden |
| Cross-seed training repeatability | Each promoted policy comes from one training seed. The three certification seeds vary *evaluation* initial conditions only |
| Perception | The policy consumes ground-truth blade pose. The vision student is scaffolding and has not been trained from the promoted policy |
| Industrial fidelity | Rack, blade, and rail are primitive proxies with no connector, latch, cable, chamfer, measured tolerance, or force-displacement curve |
| Full blade swap | Extraction, stow, acquisition, and verification are an eight-phase scaffold, not a converged policy |

## Why the numbers should be believed

- **Held-out.** Evaluation seeds never appear in training. Each level was
  certified on three of them.
- **Reset-safe.** Isaac Lab resets a finished environment inside `step`. The
  original evaluator read pose error afterwards, which measures the *next*
  episode. Metrics are now captured before that reset, and a unit test fails if
  the ordering regresses.
- **Interval, not point estimate.** A 100% sample has zero observed variance, so
  the report gives a Wilson 95% interval rather than implying certainty.
- **Failure-preferring categorization.** If an instability and a geometric
  success fire in the same control step, the episode is counted as the
  instability.
- **Pooled from raw rows.** Percentiles are recomputed over the pooled episode
  table, not averaged across runs.
- **Gate is in code.** `scripts/aggregate_evaluation.py` exits non-zero when any
  stage, the pooled rate, any randomized-parameter bucket, or the instability
  count fails. Thresholds are arguments, recorded in the report.

## Honest one-line summary

A reinforcement-learning policy trained in NVIDIA Isaac Lab performs
zero-gravity robotic insertion of a server blade into a rack at 100% success
over 18,100 held-out simulated episodes across two contact-robustness levels,
with reset-safe terminal-state evidence and confidence intervals — a simulation
result on primitive geometry, not a validated flight or hardware capability.
