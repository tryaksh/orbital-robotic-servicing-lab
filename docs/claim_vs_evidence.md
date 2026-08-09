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
| The policy still succeeds at 1.5 mm side clearance with payload mass randomized over 5-15 kg | 9,021 / 9,021 deterministic episodes, seeds 1062/2062/3062, and 100% in each of the low, mid, and high mass bands over an observed 5.00-14.97 kg | `evidence/rigid_grasp_l2_ep1800_certification.json`. **The mass half of this is close to vacuous**: a sweep to 1-50 kg is also 100%, because zero-g quasi-static motion makes the task nearly mass-insensitive. Read this as tight-clearance robustness, not payload robustness |
| Progressive fine-tuning improved precision and speed, it did not merely preserve success | Stage-0 terminal axial error fell from 4.15 mm mean (L0) to 1.65 mm (L1); full-distance median cycle time fell from 7.90 s (L0) to 7.20 s (L2) despite tighter clearance and a threefold mass range | Same reports. Distribution is bounded by the success criterion, see below |
| Terminal-state evidence cannot be corrupted by the simulator's automatic reset | `TerminalMetricsMixin` snapshots each episode inside `_reset_idx`; unit tests assert the captured value differs from the post-reset value | `tests/test_terminal_metrics.py` |
| No episode ended in numerical or physics instability | Zero non-finite and zero mount-instability terminations across 27,121 episodes | Categorization prefers instability over success when both fire in one control step |
| Simulated cycle time is measured, not estimated | Median 7.2 s at full reset distance, 1.2 s at near distance, at 30 Hz control | Simulated time. Does not include perception, approach, grasp, or extraction |
| Training is reproducible on consumer hardware | 512 environments on a 12 GB laptop GPU; 500 PPO epochs in about 22 minutes at 5,000-8,500 environment-steps/s | Isaac Sim 5.1 publishes a 16 GB VRAM minimum; this runs under it via benchmark-driven environment counts |
| The operating envelope is characterized, not just the operating point | Success degrades monotonically from 100% at the trained pose error to 97.0% at 3×, 62.4% at 6×, and 21.2% at 12×, with the half-success point near 7× | `evidence/rigid_grasp_l2_envelope_pose_error.json`. 500 episodes per point, one axis varied at a time |
| The policy degrades safely rather than diverging | Zero instability and zero non-finite terminations at every sweep point, including 12× pose error where it fails four episodes in five | It stops completing insertions; it never blows up numerically |
| Insertion contact load is measured, not assumed | Peak contact force over 4,513 successful Level-2 episodes: mean 6.73 N, p95 16.56 N, max 66.36 N; impulse p95 16.29 N·s | `evidence/rigid_grasp_l2_contact_forces.json`. Simulated contact against primitive geometry: a relative damage proxy, not an absolute force budget |
| Contact load depends strongly on approach length, and success rate hides it | Worst-case peak force rises about sevenfold from the near start (9.75 N) to the full start (66.36 N), while success stays 100% at both | Nothing in the reward, terminations, or action space bounds contact force today, so the policy has no reason to prefer a gentle insertion |
| The dominant failure mode is identified | Lateral divergence: terminal lateral error p95 goes from 0.0 mm at 3× to 60.6 mm at 4×, tripping the 60 mm failure predicate. Timeouts stay a small minority | This contradicted the prediction from margin analysis, which expected orientation to fail first. Orientation rises in failing episodes but is a symptom |

## What is not established

| Not claimed | Why |
| --- | --- |
| Learned grasping | The blade is held by a PhysX fixed joint standing in for an already-secured grasp. The real Robotiq pad/handle contact task failed its axial pull gate. The near-zero tool-to-handle error in the reports is a property of that joint, not of a grip |
| Sim2Real transfer | No real UR10e, hardware-in-the-loop rig, wrist force/torque sensor, calibrated camera, orbital acceleration data, or radiation dataset has been used |
| Accuracy independent of the success criterion | Because every certification episode succeeded, the terminal error distribution is bounded by the success box. It shows where inside tolerance the policy lands, not error it was free to exceed |
| Robustness to rail stiction or mount compliance | Level 3 stiction reaches valid geometry but cannot settle below velocity limits and is documented as blocked, not hidden. Level 4 floating-mount wobble is blocked behind it |
| Payload-mass robustness in any meaningful sense | The task is nearly mass-insensitive in this regime, so the mass sweep is flat. A real mass axis needs faster motion, real grasp friction where weight sets slip margin, or gravity |
| Damage safety | Contact force is now measured but still not *limited*. There is no force budget, no abort-and-retry, and no connector model, so nothing here shows the insertion would not bend a real pin |
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
- **Stress runs cannot pose as certification.** Evaluating outside the trained
  distribution flips the report to `simulation_capability_envelope` and marks
  the gate non-applicable, so a deliberately degraded result can never be
  mistaken for a promotion.
- **A failed prediction is recorded, not quietly dropped.** Margin analysis said
  orientation would break first. The sweep showed lateral divergence. Both are
  in `docs/status.md`.

## Honest one-line summary

A reinforcement-learning policy trained in NVIDIA Isaac Lab performs
zero-gravity robotic insertion of a server blade into a rack at 100% success
over 27,121 held-out simulated episodes across three contact-robustness levels,
up to 1.5 mm side clearance with payload mass randomized over 5-15 kg, with
reset-safe terminal-state evidence and confidence intervals — a simulation
result on primitive geometry, not a validated flight or hardware capability.
