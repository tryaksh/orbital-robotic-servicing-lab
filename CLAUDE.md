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

Levels 0, 1, and 2 of the secured-grasp insertion curriculum are promoted, each
on three held-out evaluation seeds, at 100% success over roughly 9,000 episodes
per level (27,121 total) with terminal metrics captured before Isaac Lab's
automatic reset. Level 2 covers 1.5 mm side clearance and 5–15 kg payload mass.
Envelope sweeps past the trained range show initial pose error is the binding
axis (half-success near 7× trained noise, failing by lateral divergence, never by
numerical instability), while blade mass is not a meaningful axis in this regime,
which weakens the Level-2 mass claim. Level 3 stiction is physically blocked.
Those policies hold the blade with a PhysX fixed joint; that is not learned
grasping. Contact force is measured per episode (Level-2 peak p95 16.6 N, max
66.4 N). Reward shaping at two strengths failed to constrain it; adding contact
force to the observation and retraining against a matched control cut contact
impulse 59% at the mean and 89% at the median while leaving peak force and cycle
time unchanged, so sensing binds sustained rubbing and peak force is
geometrically irreducible under position-based IK. Force sensing did **not** buy
robustness to pose error: that ablation was run, refuted, and its cause found —
the rack's 16.6 mm lead-in flares were doing the alignment mechanically. On the
physical side, the head-on grapple pin holds 69 N of axial pull against the
66.4 N the insertion reaction demands, where flat pads on a smooth post held
about 6 N, and three skills — capture, extract, insert — train against it and
chain into two servicing workflows that run end to end in one continuous episode
holding the module by pad-against-pin contact with no fixed joint. **Everything
in that last sentence is capability, and as of 2026-08-11 all of it is
certified and none of it passes its gate.**

## What the 2026-08-11 certification session established

Read this before planning anything. It overturned two things the previous
handover asserted.

**The demonstration was quoting the wrong policies.** `evidence/` named grasp v2,
extract v2 and insert v3 while `run_workflow_demo.py` loaded v3, v4 and v5. All
three are now certified as the versions the demonstration loads, on three
held-out seeds each, in files named for the version:

| Skill | Report | Episodes | Success |
| --- | --- | ---: | ---: |
| Capture v3 | `evidence/grapple_grasp_v3_certification.json` | 9,020 | 95.55% |
| Extract v4 | `evidence/grapple_extract_v4_certification.json` | 9,078 | **0.00%** |
| Insert v5 | `evidence/grapple_insert_v5_certification.json` | 3,074 | 6.96% |
| **Insert v6** | `evidence/grapple_insert_v6_certification.json` | 3,000 | **95.57%, gate passes** |

**The chain is certified too**, across three seeds and 576 workflows each:
removal `evidence/workflow_remove_certification.json` at **0 / 576**, and
installation at **497 / 576, 86.28%**, Wilson 95% [83.23, 88.85]
(`evidence/workflow_install_v6insert_certification.json`; the 15.10% in
`workflow_install_certification.json` is the same chain before the insert clock
was fixed). Neither passes the 95% gate.

**One property of the interface explains almost all of it, and it is not a
training problem.** A single-point tapered pin clamped by flat pads cannot resist
rotation about the closing axis, because the pads' contact normals lie along that
axis and a normal force cannot oppose a moment about its own direction. Measured
three independent ways:

- extraction holds grip *position* at 12.2 mm for a whole 15 s pull and fails on
  grip *attitude* at 0.299 rad against a 0.20 rad limit, which is why it scores
  zero while travelling 458 of the required 495 mm;
- of insert v5's 2,860 failures, **93.01%** are outside that same tolerance at
  the step they end on, while 100% satisfy lateral alignment and blade
  orientation, and all 214 successes sit against the limit at a maximum of
  0.1945 rad;
- pooled over 576 chained removals the grip attitude is inside tolerance at the
  end in **3.8%** of episodes.

The module itself stays straight — its orientation error against the goal is
0.0043 rad — so this is the wrist rotating relative to a module the rails hold
still.

**The previous handover's reading of insert as "slow, not unreliable" was
right, and a mid-session reading that contradicted it was wrong.** Insert v5's
failures are 93% out of grip-orientation tolerance at the step they end on, which
looked like the pin's yaw capping the skill. Lengthening the episode from 12 s to
20 s and fine-tuning took it from 6.96% to **95.57%**, and the install chain from
15.10% to **86.28%**. Grip orientation was converging, slowly, and the old
episode cut it off before the median success had happened: successful insertions
take 13.43 s. The lesson is in `docs/status.md` and worth carrying: a
distribution of terminal states says which condition is unsatisfied when the
clock stops, not whether that condition was still moving.

Extraction is a different case and the yaw diagnosis there still stands: it
scores zero while holding grip *position* at 12.2 mm for a full 15 s pull, so
nothing about it is a clock problem.

**A per-phase budget now reconciles the chain with per-skill certification.**
`PHASE_BUDGET_S` in `run_workflow_demo.py` reads `episode_length_s` off the three
task configurations, so a phase that overruns its skill's own episode fails the
workflow. That is what turned "it completes in the chain" into 569 removals
overrunning extract's 15 s and 448 installations overrunning insert's 12 s. Do
not widen the workflow episode to make a number look better; the budget is
derived on purpose.

## The anti-yaw yoke: built, dimensioned, axially validated, yaw untested

The second-generation interface feature is implemented and off by default
(`GrapplePinBladeCfg.anti_yaw_yoke`). Two walls at a 15 mm half-gap, 34 mm long
from the collar face, with a 10 mm lead-in flare at 20 degrees giving 5.14 mm of
catch per side. Every dimension comes from a re-reading of
`evidence/gripper_collision_envelope.json` across the whole closure range: no
gripper body other than an inner finger reaches past 0.1245 m from the flange,
and the widest non-finger body reaches 17.5 mm against the fingers' 13.5 mm, so
there is a 37.6 mm band behind the collar where a 15 mm wall is safe and nowhere
else. `tests/test_grapple_geometry.py` defends all of it without a simulator.

**It does not cost the hold**: 67 N at the 0.48 rad capture command against the
66.4 N required, on the same grid the 69 N plain-pin figure was measured on, with
angular slip under axial pull falling from 0.1481 to 0.1312 rad at p95.
`evidence/grapple_pin_axial_pull_gate_yoked.json`.

**Whether it fixes yaw is unmeasured, and a static probe cannot measure it.** The
yaw probe reports 0.079 rad whatever load is applied, identically with and
without the yoke, and 200 N of lateral force moves the module 1.2 mm — because
the capture scene holds the module in its rails and the rails constrain it. Yaw
is a property of the interface *after* the rack lets go, so it has to be measured
on a moving extraction. The two probe files are named
`grapple_pin_yaw_probe_railed_*` and carry `gate.applies: false` so they cannot
be misread as evidence the yoke does nothing.

## Next action, decided 2026-08-11: turn the yoke on and re-measure

The order below is chosen so that each step's answer is readable off a number
that already exists.

**1. Raise the capture skill, which is now the chain's second bottleneck.**
With insert fixed, 29 of the install chain's 79 remaining failures are capture
overrunning its own 6 s budget, and capture v3 certifies at 95.55% with its worst
stage at 92.61%. It has never had the IndustReal sampling-based curriculum the
roadmap has been pointing at: `InsertionSuccessRateCurriculum` ramps *into* the
hard stages through mixtures, which is the pathology that produced the hollow
99.3% recorded in `docs/status.md`. This is the cheapest remaining point.

**2. Turn the yoke on and retrain capture.** Set `anti_yaw_yoke = True` on the
grapple-pin blade config. Changing the pin changes the contact, so the capture
skill must be retrained before anything downstream is judged; run
`scripts/grasp_diagnostics.py` first to confirm a grip still forms, then train
`Isaac-ZeroG-Blade-GrapplePin-Grasp-v0` and certify it. Watch the capture rate:
the yoke's 1.5 mm parallel clearance is the one thing that could make capture
harder, and the 5.14 mm lead-in exists to stop it.

**3. Retrain extract on the yoked pin and read the answer off it.** Extract is
the cleanest instrument this project has for yaw, because it currently scores
**0.00%** for that reason alone and its grip-position error is already fine.
Anything above zero is the yoke working. Certify on the same three seeds.

**4. Re-certify the chain, both workflows.** `scripts/certify_workflow.sh`. Then,
and only then, attempt `--workflow full` and record the round trip.

**5. Only after the chain is certified, perception.** The requirement is already
written into `docs/service_interface_spec.md` section 7: 4 mm laterally. The
camera has to be fixed first — `docs/perception_plan.md` measures it resolving
4 mm as **0.13 pixels** — and the fix is a narrower field of view, not more
pixels.

Adopt established formulations rather than inventing reward terms. IndustReal's
sampling-based curriculum samples the whole initial-state range from the first
step and raises only its *easy* bound as success improves, which is the direct
fix for the Grasp-v1 pathology in `docs/status.md`; the mixture ramp in
`InsertionSuccessRateCurriculum` does the opposite. FORGE (arXiv 2408.04587)
conditions the policy on a per-episode maximum allowable force rather than the
two fixed penalty profiles this repository already measured as ineffective.

Must not be deleted: the grapple pin geometry and the yoke,
`docs/service_interface_spec.md`, everything in `evidence/` including the
negative results and the two inconclusive yaw probes, the contact-force machinery
in `mdp/insertion.py`, the evaluator and its promotion gate,
`TwoStageRobotiqAction` (including its per-environment `hold_latch`) and
`hold_two_stage_grip` in `mdp/grapple.py`, the per-phase budget in
`run_workflow_demo.py`, and the visual-randomization modules hanging off
`vision_insertion_env_cfg.py`.

What the head-on grapple pin established, and why it stays: it is the first
interface here to form a real grip, holding **69 N** of axial pull against the
66.4 N the insertion contact reaction demands, where flat pads on a smooth post
held about 6 N. The step that got there was recognising that **capture and hold
are two different commands** — a wedge converts closing force into thrust along
the pull axis, so a firm capture drives the payload away before it is taken.
Capture at 0.48 rad, firm to 0.68 once loaded. The window is narrow and
asymmetric: 0.44 holds 63 N, 0.52 holds 68 N, 0.56 collapses to 26 N. Bias low.

## Operating rules

- Preserve exact zero gravity and the 30 Hz policy / 120 Hz physics timing
  unless an experiment explicitly tests a change.
- Never resume a checkpoint after changing action or observation dimensions.
- Change one failure category per experiment and save a JSON report.
- A scripted controller is allowed only as a physics feasibility test;
  demonstrations must use a checkpoint.
- Never call a fixed joint, compliant spring, or scripted action a learned
  grasp.
- Never weaken a success threshold to make a gate pass.
- Never quote a success rate without checking that the certification in
  `evidence/` names the same policy version the demo loads. This rule exists
  because grasp v3, extract v4 and insert v5 were quoted using v2/v2/v3 numbers.
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
| Pin and yoke dimensions, and their derivation | `src/zero_g_blade_swap/grapple_geometry.py`, `tests/test_grapple_geometry.py` |
| The chained workflow and how it is judged | `scripts/run_workflow_demo.py`, `scripts/certify_workflow.sh` |
| Re-certifying a skill after a retrain | `scripts/certify_demo_policies.sh` |
| Measuring a new interface before training on it | `scripts/run_yoke_gates.sh` |
| Capture/hold gripper action | `src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py` |
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
name, so a task may record extra columns such as randomized blade mass without
invalidating earlier runs.

Promotion gate: at least the stated success rate pooled *and* in every
curriculum stage, at least 80% in every randomized-parameter bucket, zero
instability terminations, and zero non-finite terminal metrics.
