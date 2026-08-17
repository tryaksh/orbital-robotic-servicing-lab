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
| Force feedback reduces accumulated contact load, measured against a matched control | Contact impulse fell 59% at the mean (7.55 to 3.06 N·s), 89% at the median (6.26 to 0.70), and 40% at p95 (16.55 to 9.94), with mean cycle time unchanged at 3.87 versus 3.89 s | `evidence/force_feedback_certification.json` against `evidence/force_feedback_control_certification.json`. Both trained from scratch on the identical schedule, seed, reward, and PPO config, differing only in the observation, so the effect is not an artifact of retraining |
| Peak contact force is geometrically irreducible in this action space | Peak force did not move with force feedback: 6.31 to 6.24 N mean pooled, and 10.55 to 10.48 N mean at full reset distance. Three penalty strengths and full force sensing all failed to move it | Testing this further needs an admittance/impedance action space, roadmap item 7. The floor is consistent with a blade crossing 1.5 mm clearance under position-based differential IK |
| The simulated pad grasp holds zero axial load, and the reason is measured | PhysX reports 0.0 N between finger and blade across the full 0–0.8203 rad finger range; drive torque stays at 0.39 N·m of a 10 N·m limit; blade travel under load matches free-body force over mass to within 8% | `evidence/grasp_axial_pull_gate.json`. Replaces a pass/fail "failed its axial pull gate" with a cause: the handle is configured 0.179 m from the flange while the fingers only obstruct between 0.06 and 0.15 m |
| Force sensing does **not** extend the pose error this policy tolerates | Force-aware against a matched force-blind control, one configuration, one seed, one schedule, 33,500 held-out episodes across seven slot displacements: identical at and below the trained 4 mm (99.87% against 99.77%), and the force-aware arm is **worse** beyond it, 96.94/87.50/74.07% against 99.56/94.90/82.31% at 6/8/10 mm | `evidence/uncertain_insertion_*_certification.json` and `*_envelope.json`. This refutes the hypothesis the pivot was built to test. Diagnosis in `docs/status.md`: the lead-in flares catch 16.6 mm per side and already handle a 4 mm offset mechanically, and a position-controlled action space gives a policy no way to turn a force reading into compliance |
| A policy that can measure force spends its force budget | The force-aware arm uses about twice the peak contact force at every displacement, 27.65 N against 13.72 N at p95 at 4 mm, for the same success and the same 7.07 s median cycle time | Conditioned on a per-episode maximum allowable force, only the arm that can measure force knows how close it is to it. Its failures are force-limit aborts and timeouts while grinding |
| A module is captured, extracted, and re-inserted by three trained policies holding it through real pad-against-pin contact, with no fixed joint | Two workflows run end to end in one continuous episode: removal pulls a fully installed module 495 mm clear of the rack, installation seats one at 8.63 mm axial and 0.61 mm lateral error | This is a **capability** claim, not a reliability one. The per-workflow success rate is certified separately and is low; see the next two rows |
| A single-point tapered pin cannot hold the module's attitude, and that is the binding constraint on extraction | Extract v4: 0 successes in 9,078 held-out episodes, with grip *position* holding at 12.2 mm median for the whole pull and grip *attitude* at 0.299 rad against a 0.20 rad limit. Insert v5: of 2,860 failures, 93.0% are outside the grip-orientation tolerance at their terminal step, while 100% satisfy lateral alignment and blade orientation | `evidence/grapple_extract_v4_certification.json`, `evidence/grapple_insert_v5_certification.json`. The module itself stays straight (0.0043 rad of blade orientation error), so this is the wrist rotating relative to a module the rails hold still |
| **That rotation is not the "yaw" this project called it for three sessions** | Decomposed into the gripper's own axes: 0.198 rad about the closing axis, 0.199 about the transverse axis, 0.070 about the approach axis | `play.py --grip_axis_metrics`. Only the magnitude was ever recorded before, and a magnitude cannot say which axis a rotation is about. Two interface features were designed against the closing axis alone on the strength of that name |
| Two interface features were built against it and **both are net negatives** | Anti-yaw yoke, all three skills retrained and certified: capture 95.55% → 88.81%, insertion 95.57% → 28.70%, extraction 0.00% → 0.13%. Modelled latch, swept 10–160 N·m against an unchanged policy: targeted rotation moves 0.006 rad while extraction travel collapses from 458 mm to about 25 mm | `evidence/grapple_grasp_v4_certification.json`, `evidence/grapple_extract_v5_certification.json`, `artifacts/latch/`. Both are off by default and both stay implemented, because the measurement is the result. The latch sweep used an unchanged policy specifically so no difference could be a training artefact |
| **A module is removed from the rack by two chained policies, held by real contact throughout, and the result survives a settling re-check** | 569 / 576 chained removals on three held-out seeds, **98.78%**, Wilson 95% [97.51, 99.41], consistent at 98.44 / 99.48 / 98.44, zero instability and zero non-finite terminations. The promotion gate passes | `evidence/workflow_remove_retain_certification.json`, checked with `check_evidence_currency.py`. Took three fixes, none mechanical: a reward for arriving settled, an attitude penalty whose clamp was saturating at the angle the policy parked at, and a third gripper command that stops squeezing once the module is free. Extraction alone was 68.62% under a reset that killed 25% of its episodes before the policy acted; with that reset corrected it certifies at **99.02%** (`grapple_extract_v14reset_certification.json`) |
| ~~Extraction works one time in ten~~ **RETRACTED 2026-08-16: every extraction figure published before that date was measured under a criterion the code no longer contains** | `grapple_extract_v8_certification.json` (13:21 on 2026-08-15) reports 68.36%; the settled-enough velocity limits were derived and tightened at 14:58 the same day, from a chosen 0.10 m/s to a derived 0.0143 m/s. Re-read against the limit now in force, **0 of that run's 6,156 counted successes qualify**, and the fastest-settling of them is 3.1× over. Extract v6's 10.09% and v9's 67.55% share the defect, as does the chained removal's 14.06% | `evidence/grapple_extract_v10_certification.json` is the only extraction ever measured against the current criterion, at **0.00%**, with 8,988 of 9,010 episodes ending on grip loss. The retraction is arithmetic on the published runs' own recorded terminal velocities, not a re-run. The limits stay: they are derived from the 0.70 s the chain waits before re-checking, and a module declared removed at 0.10 m/s drifts 70 mm against a 20 mm tolerance |
| The extraction end pose is reachable, so neither the workcell nor the interface explains what is left | Converged IK holds the head-on attitude at the end pose to 0.0114 rad against a 0.20 rad tolerance, and moving the robot base back makes it worse | `scripts/calibrate_grasp_pose.py --robot_base_x`. The 0.10–0.26 rad residuals recorded earlier came from a 400-step servo that had not converged. What remains is the objective: attitude cost about 0.16 per step at the success limit against a progress term weighted 12 |
| One servicing skill passes its promotion gate, and the chain it sits in reaches 89% | Insert v6: 2,867 / 3,000 held-out episodes, 95.57%, gate passed. Chained installation on the promoted set: 515 / 576, **89.41%**, Wilson 95% [86.63, 91.67], zero instability and zero non-finite | `evidence/grapple_insert_v6_certification.json`, `evidence/workflow_install_promoted_certification.json`. The whole improvement over v5's 6.96% came from one change: the episode went from 12 s to 20 s, because successful insertions take a median of 13.43 s and the old episode ended before the median success happened. **The 84.38% and 86.28% figures earlier reports carry are superseded**: they describe the same policies under a 6 s capture budget that became 10 s in commit `ffac648`, hours after they were written |
| The remaining installation gap is a distribution problem, and it is now measurable rather than inferred | Insert v6 scores 95.57% from its own reset and **93.06%** (536 / 576, three held-out seeds) on the states a real capture hands it, against 90.45% for the same policy in the real chain. The state it takes over in matches the chain's hand-off at the median and the 95th percentile on grip error, grip attitude, finger angle, drive torque, module pose and all six arm joints | `evidence/insert_chain_handoff_gate.json` against `evidence/workflow_install_promoted_certification.json`. This is a **gate**, not a promotion: it was run before any training to test whether `Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0` reproduces the hand-off. Three earlier reconstructions of the same hand-off as a reset distribution scored 0.00%, 26.32% and 47.17% on this question |
| **One insert policy seats a module in either bay of a two-bay rack** | 98.87% in bay 1 and **98.34% in bay 2**, pooled 98.60% over 3,004 episodes on three held-out seeds, zero instability and zero non-finite. Gated on the *worse* bay, not the pool | `evidence/grapple_insert_two_slot_certification.json`. The second bay is the certified one displaced part for part rather than re-authored, and the skill transferred almost immediately: 0 to 83% within 40 epochs of the curriculum unlocking it, while bay 1 never dropped below 97.9%. That is evidence for the construction, not for the policy |
| **The camera reports which bay holds the module, not only where it is** | 100% per-bay accuracy and 100% exact-match over the whole rack on 12,000 held-out frames, against a **66.6%** majority-class baseline, with module pose at 2.81 mm mean and 6.47 mm p95 | `evidence/module_pose_head_two_slot.json`. Read the 100% for what it is: the bays are 220 mm apart and this camera resolves 4 mm as 1.31 px, so the task is easy and the number says the construction is sound rather than that the perception is clever. The pose error is worse than the single-bay head's 1.75 mm, honestly so — the module now ranges over both bays |
| ~~The relocation, bay 1 to bay 2, is demonstrated~~ **NOT ESTABLISHED. Every skill it needs is certified and the chain does not complete** | Every episode times out inside the lateral transit. The mechanism is measured: with the tool sitting on its final waypoint the tool-to-module offset changes sign, −0.335 m to **+0.305 m**, so the module has swung end-for-end about the single-point pin. The transit commanded nothing on its three rotation channels. Holding the attitude takes grip error through the flight from 24 mm to **11 mm** and the retreat leg then completes for all 64 environments; the 220 mm lateral crossing, flown with the arm folded back near its own base, does not | `docs/status.md`. This is the same unsolved problem the `full` round trip has always had — *"the grip degrades from 15 mm to 35 mm during the return whatever speed it is flown at"* — and the relocation is simply the first task that has to carry a module 734 mm through free space. Three transit corrections are in and measured; one rate-limiting attempt is refuted. **No relocation success rate is claimed** |
| Capture's skill number and the number the chain needs of it are different questions, and both are measured | Capture alone is **88.78%** over 9,011 episodes (100% / 87.12% / 79.22% by reset distance) and **fails** its 95% gate, with 1,008 of 1,011 failures being refusals rather than timeouts. Chained installation, driven by the same checkpoint, overruns its capture phase **once in 192 episodes** | `evidence/grapple_grasp_v5_certification.json`. The skill task ends an episode when `capture_failed` fires; the chain has no such term, hands over on a 10 mm grip held 0.30 s, and otherwise lets the capture keep closing for its full 10 s — which this page already records it doing, "closing to a 9-to-12 mm median if simply allowed to finish". Adding such a termination to the chained-insert task was separately measured at 95.31% → 69.27%. Neither number may be quoted as the other, which is what rule 9 exists for. The earlier 96.10% is retracted |
| The skills the demonstration loads are certified as the versions it loads | Three checkpoints, three held-out seeds each, reports named for the version and carrying its SHA-256 | Written after a whole session quoted grasp v3 / extract v4 / insert v5 using v2 / v2 / v3 numbers. Two of the three stale figures pointed the wrong way |
| The dominant failure mode is identified | Lateral divergence: terminal lateral error p95 goes from 0.0 mm at 3× to 60.6 mm at 4×, tripping the 60 mm failure predicate. Timeouts stay a small minority | This contradicted the prediction from margin analysis, which expected orientation to fail first. Orientation rises in failing episodes but is a symptom |

## What is not established

| Not claimed | Why |
| --- | --- |
| Learned grasping | The blade is held by a PhysX fixed joint standing in for an already-secured grasp. The handle is configured past the fingertips, so the fingers transmit 0 N and the contact task has never grasped anything. The tool-to-handle error in the reports is exactly 0.0000 m because the joint welds the blade to the frame the metric compares against; it is a tautology, not a grip audit |
| Sim2Real transfer | No real UR10e, hardware-in-the-loop rig, wrist force/torque sensor, calibrated camera, orbital acceleration data, or radiation dataset has been used |
| Accuracy independent of the success criterion | Because every certification episode succeeded, the terminal error distribution is bounded by the success box. It shows where inside tolerance the policy lands, not error it was free to exceed |
| Robustness to rail stiction or mount compliance | Level 3 stiction reaches valid geometry but cannot settle below velocity limits and is documented as blocked, not hidden. Level 4 floating-mount wobble is blocked behind it |
| Payload-mass robustness in any meaningful sense | The task is nearly mass-insensitive in this regime, so the mass sweep is flat. A real mass axis needs faster motion, real grasp friction where weight sets slip margin, or gravity |
| Damage safety | Accumulated contact load is now reduced by a large, measured margin, but peak force is not, and there is still no connector model, force-displacement curve, or hardware measurement. Nothing here shows the insertion would not bend a real pin |
| That force can be regulated without force feedback | Two penalty strengths, the stronger charging the same order as the success reward, changed mean contact by 2.6% and impulse not at all. Adding force to the observation cut impulse 59% with everything else held fixed. Force control needs force sensing |
| That peak contact force can be regulated at all in this action space | Nothing tried has moved it: two penalty strengths, and full force feedback with a matched control. Position-based differential IK through a 1.5 mm clearance slot appears to have a hard floor |
| That force sensing helps under pose uncertainty | Measured and refuted on this task, see above. The result is specific and its cause is identified: force was made *observable* while the action space stayed stiff, so the only thing a policy could do with it was push harder. Whether force sensing helps with an admittance or impedance action space is untested, and is now the main open question |
| Cross-seed training repeatability | Each promoted policy comes from one training seed. The three certification seeds vary *evaluation* initial conditions only |
| Perception | The policy consumes ground-truth blade pose. The vision task is scaffolding and no policy has been trained on it |
| Anything under pose uncertainty | Every result on this page comes from a task where the policy is *told* its exact pose error. With a rigid known object on a constrained axis and full observability, that is motion planning and force control; the pivot recorded in `docs/status.md` exists because RL cannot demonstrate its value there |
| Industrial fidelity | Rack, blade, and rail are primitive proxies with no connector, latch, cable, chamfer, measured tolerance, or force-displacement curve |
| A reliable servicing workflow | The eight-phase scaffold was deleted on 2026-08-10 because four of its five stages had no physics content, and is not coming back. The three head-on grapple-pin skills were restored on 2026-08-11 and do chain, but none of the three passes its promotion gate and the chain's own certified success rate is low. Capability is demonstrated; reliability is not claimed |
| A remove-and-replace round trip | Removal works and installation works. Carrying the module between them does not, and the cause is the interface rather than the controller: the pin does not constrain yaw once the rails release the module, and slowing the return fourfold makes it worse |
| That the skills' failures are a training problem | Measured otherwise. Two of the three are blocked by a geometric property of the interface, not by policy quality or episode budget. See the grip-orientation rows above |

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
- **The headline experiment of the current pivot returned a negative result and
  is published as one.** Force feedback was predicted to extend the tolerable
  pose error and did the opposite. The prediction, the curve, the contact-force
  measurement that explains the direction, and the methodological miss that made
  a flat curve foreseeable are all in `docs/status.md`.
- **A failed experiment is published with its diagnosis.** Force-penalty shaping
  did not reduce contact load at either strength tried. The result, the
  arithmetic showing why the first attempt was too weak, and the two hypotheses
  that survive the second attempt are all in `docs/status.md`.
- **The follow-up carried a control.** The force-feedback policy is compared
  against a policy trained from scratch on the identical schedule with the
  observation left alone, so its impulse reduction cannot be an artifact of
  retraining rather than of sensing.
- **A convenient assumption was measured and destroyed.** The contact-grasp task
  was assumed to be a nearly working grasp that needed tuning. Measuring it
  showed the fingers transmit 0 N at any commanded closure, which moved
  grasping from a training problem to a geometry bug and invalidated the
  project's own prior description of that failure.

## Honest one-line summary

A reinforcement-learning policy trained in NVIDIA Isaac Lab performs
zero-gravity robotic insertion of a server blade into a rack at 100% success
over 27,121 held-out simulated episodes across three contact-robustness levels,
up to 1.5 mm side clearance with payload mass randomized over 5-15 kg, with
reset-safe terminal-state evidence and confidence intervals; adding wrist-force
feedback and retraining against a matched control cut accumulated contact
impulse by 59% at no cost in cycle time — a simulation result on primitive
geometry, not a validated flight or hardware capability.
