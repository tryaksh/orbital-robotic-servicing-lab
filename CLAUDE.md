# Agent handover

Act as the senior robotics simulation engineer who owns this repository.
Preserve its evidence-first approach: diagnose one physical or learning
bottleneck at a time, require deterministic held-out evaluation before
promotion, and never describe a smoke test or an attractive render as Sim2Real
validation.

## Mission

Answer one engineering question with measurements: **what service interface does
a 6-axis manipulator need in order to capture, extract, and insert a modular
compute unit in microgravity, and what loads does that impose?**

The deliverable is `docs/service_interface_spec.md`, a module-side specification
where every dimension traces to a measurement, plus grasp/extract/insert skills
certified under randomized pose, payload, and friction with bounded contact
loads.

Scope was narrowed deliberately on 2026-08-10. The earlier framing carried a
five-stage chain from compute fault through isolation, replacement, hardware
verification, and restored compute. Four of those five stages have no physics
content; in simulation they are a state machine, and they invited a mock-up
reading of work whose value is measured contact mechanics. **Do not reintroduce
them.** Likewise, lead with the mechanism rather than the setting: the envelope
sweep already showed payload mass is close to vacuous in this regime, so
"orbital data centre" claims more than the evidence supports, while the
zero-gravity mechanisms this project actually measured do not:

- You cannot grasp a free-floating mass by squeezing it. Asymmetric contact
  timing becomes net momentum with nothing to absorb it.
- A wedge converts closing force into payload thrust, so capturing harder makes
  capture worse. That is why capture and hold are separate commands.

Position this as research into **design-for-serviceability and contact-rich
field servicing**, not as a flight-ready system. The value is the disciplined
workflow: GPU-parallel RL, physics-gap diagnosis, curriculum design, measurable
promotion gates, and an honest Sim2Real plan. Several results here are negative
or retracted, and they stay in the record.

## Current state in one paragraph

Levels 0, 1 and 2 of the secured-grasp insertion curriculum are promoted at 100%
over 27,121 held-out episodes, but those policies hold the module with a PhysX
fixed joint and that is not learned grasping. The real work is the head-on
grapple pin, where the module is held by pad-against-pin contact with no fixed
joint: the pin holds **69 N** of axial pull against the 66.4 N the insertion
reaction demands, where flat pads on a smooth post held about 6 N. On that
interface three skills are certified on three held-out seeds each — **capture
95.55%**, **insert 95.57%**, and **extract 28.48%**, the last having gone
0.00% → 10.09% → 28.48% on 2026-08-15 without a single hardware change. They
chain into two servicing workflows that run end to end in one continuous
episode: **installation certifies at 86.28%** and **removal at 14.06%**, both up
from a chain that composed to zero four sessions ago. Nothing passes the 95%
gate, the remaining failure is entirely grip attitude — 6,387 of extract's 6,438
failures end at the 0.350 rad limit while grip *position* holds at 13.0 mm — and
there is one live contradiction to resolve first: **extract v7 is the better
skill and the worse chain component**, scoring 28.48% alone but dropping the
removal chain from 14.06% to 3.30%, so the chain currently loads v6.

## What 2026-08-15 established, and what it demolished

Read this before planning anything. Four hypotheses were tested and **three were
refuted**, which is the session's real output.

**The anti-yaw yoke is a net negative and is off.** All three skills were
fine-tuned onto it and certified: capture 95.55% → 88.81%, insert 95.57% →
28.70%, extract 0.00% → 0.13%. It costs 6.7 points of capture and 67 of
insertion to buy 0.13 of extraction.

**And it was aimed at the wrong axis.** `play.py --grip_axis_metrics` now
decomposes the capture attitude into the gripper's own axes; only its magnitude
was ever recorded before, and a magnitude cannot say which axis a rotation is
about. Measured: **0.198 rad about the closing axis, 0.199 about the transverse
axis, 0.070 about the approach axis.** The yoke's walls oppose the closing axis
and nothing else. Three sessions called this failure "yaw" and designed against
that name without measuring it.

**A modelled latch is also a net negative and is off.** `mdp.GrappleLatch`
engages on a qualifying capture and applies a rated restoring torque, which is
what flight hardware does — the SSRMS latching end effector rigidizes a grapple
fixture, Dextre's tool changeout mechanism carries a powered socket drive. Swept
from 10 to 160 N·m against the *unchanged* extract v4 policy, it moves the
rotation it targets by 0.006 rad and collapses extraction travel from 458 mm to
about 25 mm, because a restoring torque on a module the rails still hold jams it
in the rails.

**The workcell is not the problem either.** Extraction ends with the tool
0.336 m horizontally from the base and 0.570 m above it, folded back over the
shoulder, and two handovers flagged that this was never checked kinematically.
Checked with 2,000 servo steps instead of 400, and the base swept along x, the
arm holds the head-on attitude there to **0.0114 rad** — seventeen times inside
the 0.20 rad tolerance — and moving the base back makes it *worse*. The
0.10–0.26 rad residuals recorded earlier were an under-converged servo.

**What was true: the clock, and then the objective.** Extract certified on a
15 s episode whose median cycle time was 15.000 s — every episode ran out the
clock. Lengthening it to 25 s *and* fine-tuning took extraction from 0.00% to
10.09% and removal from 0.00% to 14.06%. Successful extractions take 18.23 s,
so 15 s made them impossible by construction. Lengthening the clock **without**
fine-tuning was measured first and does not work: it converts 449 timeouts into
512 lost grips at a hard 478 mm ceiling.

Then the arithmetic that explains the rest. `grip_retention_penalty` charges
attitude about **0.16 per step** at the 0.20 rad success limit, against a
progress term weighted **12**. The policy was not failing to control attitude;
it was correctly trading an almost-free quantity for a well-paid one and taking
a one-off −15 at the end. The extract task now charges about 3.6 per step, via
parameters on the shared function so **insertion keeps the defaults its own
certification was produced under**.

## Next action, decided 2026-08-15

**1. Resolve why the better skill is the worse chain component.** Extract v7
certifies at 28.48% alone and drops the removal chain to 3.30%, where v6 at
10.09% alone gives 14.06%. In the chain 557 of 576 failures are timeouts;
running alone, 51 in 9,002 are. Installation is bit-identical at 86.28% because
it never calls extract, so the difference is extract and nothing else. The
hypothesis to test first is that a large attitude penalty makes standing still
cheaper than pulling when the episode starts outside the policy's comfortable
region, and a chained extract always starts wherever capture's servoing left the
arm rather than on the nominal reset. That is the same class of defect as the
0.0005 rad reset noise that made the first chained extract reverse into the
rack, and the fix shape is the same: train across the states the predecessor
actually produces. `run_workflow_demo.py --episodes` with per-phase reporting is
the instrument; do not widen the workflow episode.

**2. Then the action space, not the interface.** The arm commands relative
Cartesian pose through damped-least-squares IK, and this repository has now
measured twice that a position-controlled action space cannot convert a sensed
quantity into compliance — once on contact force (`docs/status.md`, force
feedback moved impulse and not peak force) and once on attitude, where charging
it properly was worth 18 points and the residual failure is still 99% attitude.
Roadmap item 7, an admittance or impedance action space, is the change both
results point at. **Do not build a third passive interface feature; two are
already measured as harmful.**

**3. Raise capture, which is the install chain's remaining bottleneck.** Capture
certifies at 95.55% with its worst stage at 92.61%, and 29 of the install
chain's 79 failures are capture overrunning its own 6 s budget. It has never had
the IndustReal sampling-based curriculum the roadmap points at:
`InsertionSuccessRateCurriculum` ramps *into* the hard stages through mixtures,
which is the pathology that produced the hollow 99.3% in `docs/status.md`.
`BeliefSamplingCurriculum` in `mdp/uncertainty.py` is already IndustReal's SBC
written against the slot displacement; write a sibling over the reset-noise
envelope. Change the curriculum on its own, never alongside a physics change.

**4. Only after both chains certify, perception.** The requirement is written
into `docs/service_interface_spec.md` section 7: 4 mm laterally. The camera has
to be fixed first — `docs/perception_plan.md` measures it resolving 4 mm as
**0.13 pixels** — and the fix is a narrower field of view, not more pixels.

Adopt established formulations rather than inventing reward terms. IndustReal's
sampling-based curriculum samples the whole initial-state range from the first
step and raises only its *easy* bound as success improves. FORGE (arXiv
2408.04587) conditions the policy on a per-episode maximum allowable force
rather than the two fixed penalty profiles this repository already measured as
ineffective.

Must not be deleted: the grapple pin geometry, the yoke and `mdp.GrappleLatch`
(both off, both measured, both worth keeping as evidence),
`docs/service_interface_spec.md`, everything in `evidence/` including the
negative results and the two inconclusive yaw probes, the contact-force
machinery in `mdp/insertion.py`, the evaluator and its promotion gate,
`TwoStageRobotiqAction` (including its per-environment `hold_latch`) and
`hold_two_stage_grip` in `mdp/grapple.py`, the per-phase budget in
`run_workflow_demo.py`, `scripts/check_evidence_currency.py`, and the
visual-randomization modules hanging off `vision_insertion_env_cfg.py`.

What the head-on grapple pin established, and why it stays: it is the first
interface here to form a real grip, holding **69 N** of axial pull where flat
pads on a smooth post held about 6 N. The step that got there was recognising
that **capture and hold are two different commands** — a wedge converts closing
force into thrust along the pull axis, so a firm capture drives the payload away
before it is taken. Capture at 0.48 rad, firm to 0.68 once loaded. The window is
narrow and asymmetric: 0.44 holds 63 N, 0.52 holds 68 N, 0.56 collapses to 26 N.
Bias low.

## Operating rules

- Preserve exact zero gravity and the 30 Hz policy / 120 Hz physics timing
  unless an experiment explicitly tests a change.
- Never resume a checkpoint after changing action or observation dimensions.
  Resuming across a *physics* or *reward* change is allowed and is how the
  L0→L1→L2 lineage and extract v6 and v7 were produced; say so when you do.
- Change one failure category per experiment and save a JSON report.
- A scripted controller is allowed only as a physics feasibility test;
  demonstrations must use a checkpoint.
- Never call a fixed joint, compliant spring, or scripted action a learned
  grasp. `GrappleLatch` is a modelled mechanism and is labelled as one.
- Never weaken a success threshold to make a gate pass. Extract's 854-per-1000
  failures at the 0.350 rad limit are a standing temptation to loosen that
  predicate; do not.
- **Measure the axis before designing against it.** Two interface features were
  built against a rotation nobody had decomposed, and it turned out to be split
  evenly across two axes with only one of them addressed. `--grip_axis_metrics`
  exists so this cannot recur.
- **Before believing a servo, converge it.** The extraction end pose was
  declared kinematically marginal on a 400-step IK residual; at 2,000 steps it
  converges to 0.0114 rad. An under-converged solver is not a reachability
  oracle.
- Never quote a success rate without checking that the certification in
  `evidence/` names the same policy version the demo loads. Run
  `scripts/check_evidence_currency.py` on the workflow reports; it compares
  SHA-256, not filenames, and it caught a chain running insert v5 on the night
  it was written.
- A recorded video is a demonstration. A pooled multi-seed report is evidence.
  Never let the first stand in for the second, in any document or commit message.
- A chained workflow gives every phase the episode length its own skill was
  certified on. If a skill's clock changes, change it on the task and let
  `PHASE_BUDGET_S` read it; never widen the workflow episode to make a phase fit.
- Before publishing a probe, check it can move the thing it measures. The yaw
  gate reported an identical number with and without the feature it was built to
  test, because the rails were holding the module still.
- Do not edit `src/` or `scripts/` while an evaluation sweep is running. Every
  `play.py` launch re-imports the package, so a broken edit fails every remaining
  run in the sweep rather than one.
- Do not start perception while the chained workflow is uncertified.
- Keep `.deps`, logs, datasets, checkpoints, artifacts, and videos out of Git.
- Do not reintroduce the eight-phase swap task.
  `tests/test_configuration_contract.py` fails if the swap state machine returns.

## Where to read, by task

Read the entry below that matches the work. Do not ingest `.deps`, `logs`,
checkpoints, or every task file.

| Working on | Read |
| --- | --- |
| Any result, claim, or limitation | `docs/status.md` |
| Explaining the project to a reader | `docs/claim_vs_evidence.md` |
| What to do next, prior art | `docs/roadmap.md` |
| Code and data flow | `docs/architecture.md` |
| Physics gaps, missing measurements | `docs/sim2real_matrix.md` |
| Public claim and commands | `README.md` |
| The design deliverable | `docs/service_interface_spec.md` |
| Insertion task physics | `src/zero_g_blade_swap/tasks/blade_swap/rigid_grasp_insertion_env_cfg.py` |
| Force penalties, force feedback | `src/zero_g_blade_swap/tasks/blade_swap/force_limited_insertion_env_cfg.py` |
| Camera, lighting, rack materials | `src/zero_g_blade_swap/tasks/blade_swap/vision_insertion_env_cfg.py` |
| Pose uncertainty, force threshold, SBC | `src/zero_g_blade_swap/tasks/blade_swap/mdp/uncertainty.py` |
| The pose-belief task and its ablation | `src/zero_g_blade_swap/tasks/blade_swap/uncertain_insertion_env_cfg.py` |
| Perception, before writing any of it | `docs/perception_plan.md` |
| Grasp physics before any grasp PPO | `scripts/grasp_diagnostics.py`, `evidence/grapple_pin_axial_pull_gate.json` |
| Gripper geometry, ever | `scripts/measure_gripper_envelope.py`, `evidence/gripper_collision_envelope.json` |
| Head-on capture task, and the three skills | `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py` |
| Pin, yoke and latch, and their derivation | `src/zero_g_blade_swap/grapple_geometry.py`, `tests/test_grapple_geometry.py`, `tests/test_yoke_asset.py` |
| The chained workflow and how it is judged | `scripts/run_workflow_demo.py`, `scripts/certify_workflow.sh` |
| Re-certifying a skill after a retrain | `scripts/certify_demo_policies.sh` |
| Auditing that a quoted number describes the loaded policy | `scripts/check_evidence_currency.py` |
| Is a pose reachable | `scripts/calibrate_grasp_pose.py --robot_base_x`, and converge it |
| Measuring a new interface before training on it | `scripts/run_yoke_gates.sh`, `scripts/run_latch_rating_sweep.sh` |
| Capture/hold gripper action, the latch, grip metrics | `src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py` |
| Rewards, terminations, curriculum | `src/zero_g_blade_swap/tasks/blade_swap/mdp/insertion.py` |
| Evaluation statistics and gates | `src/zero_g_blade_swap/evaluation.py`, `scripts/aggregate_evaluation.py` |
| Training and playback entry points | `scripts/train.py`, `scripts/play.py` |

## Evaluation contract

`ManagerBasedRLEnv.step` resets terminated environments before returning, so
reading pose or velocity error after `step` measures the *next* episode.
`TerminalMetricsMixin` intercepts `_reset_idx` and snapshots each finished
episode while the scene still holds its terminal state. All insertion tasks are
registered on `TerminalMetricsManagerBasedRLEnv`; the hook is inert unless an
evaluator installs it, so training is unaffected.

Certification is one `play.py` run per curriculum stage and seed writing
`--episode_metrics`, then `scripts/aggregate_evaluation.py` pooling those raw
rows into a single gated report under `evidence/`. Reports align runs by column
name, so a task may record extra columns — randomized blade mass, contact force,
the grip-attitude decomposition — without invalidating earlier runs.

Promotion gate: at least the stated success rate pooled *and* in every
curriculum stage, at least 80% in every randomized-parameter bucket, zero
instability terminations, and zero non-finite terminal metrics.
