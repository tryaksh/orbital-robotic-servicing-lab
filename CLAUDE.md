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
automatic reset. Level 2 covers 1.5 mm side clearance and 5–15 kg payload mass. Envelope sweeps
past the trained range show initial pose error is the binding axis (half-success
near 7× trained noise, failing by lateral divergence, never by numerical
instability), while blade mass is not a meaningful axis in this regime, which
weakens the Level-2 mass claim. Level 3 stiction is physically blocked. The
blade is held by a PhysX fixed joint standing in for an already-secured grasp;
that is not learned grasping. Contact force is measured per episode (Level-2 peak
p95 16.6 N, max 66.4 N, rising about sevenfold with approach length while
success stays 100%). Reward shaping at two strengths failed to constrain it, but
adding contact force to the observation and retraining from scratch against a
matched control cut contact impulse 59% at the mean and 89% at the median while
leaving peak force and cycle time unchanged: sensing binds sustained rubbing,
and peak force is geometrically irreducible under position-based IK, so the
remaining lever is an admittance action space. Learned grasping is blocked by a
measured bug, not by training: the handle is configured 0.179 m from the wrist
flange while the fingers only reach it between about 0.06 and 0.15 m, so they
close past it and hold 0 N of the 66.4 N required. The 2F-85 has since been
measured from its collision meshes rather than its body origins: the pads close
along wrist x, span 105 to 162 mm from the flange, open to 87.08 mm at
`finger_joint` 0 and close at 106.2 mm/rad, so **zero is fully open** and the
contact task's finger commands are still inverted. A head-on tapered grapple pin
built on those measurements forms the project's first real grip and holds 59 N,
ten times the flat-pad grip but 10% short of the gate. Full numbers,
limitations, and the pre-existing `train.py --smoke` probe defect live in
`docs/status.md`.

## Next action, decided 2026-08-09

Build the **head-on grapple pin** capture interface, then the grasp, extract,
and insert skills that a replacement demonstration needs. The owner has given
standing authorization for long GPU training; do not ask before starting a run.

Why this and not more gripper tuning. A parallel-jaw friction grip cannot hold
this blade, and that is structural rather than a tuning failure. The gripper
approaches downward along -z while the blade extracts sideways along -x, so the
pull axis is the one direction nothing constrains: the rails must leave it free,
and flat pads on a smooth post can only resist it by friction. Measured
consequence, with the rails solid and closure swept from 0.62 to 0.77 rad: grip
torque *falls* as the fingers close tighter, because they shove the post along
x and then close on air. Axial capacity is about 6 N against the 66.4 N the
insertion contact reaction demands.

The head-on grapple pin is **built, calibrated, and past its gate**. It is the
first interface in this project's history to form a real grip: drive torque
saturates the 10 N-m limit in 363 of 363 environments, and it holds **69 N** of
axial pull within 2 mm of slip against the 66.4 N requirement, where flat pads
on a post held about 6 N.

The step that got there is worth remembering: **capture and hold are two
different commands.** A single closure caps out at 59 N, and capacity *falls* as
the fingers close harder, because the wedge converts closing force into thrust
along the pull axis and a firm capture drives the payload away before it has
been taken. Capturing at 0.48 rad and firming to 0.68 once the grip loads gives
69 N. The window is narrow and asymmetric: 0.44 holds 63 N, 0.52 holds 68 N,
0.56 collapses to 26 N. Bias low.

Full numbers, including why the flared head you would expect had to become a
tapered wedge, are in `docs/status.md`.

**Grip force was tested as the cause and refuted, before the real fix was
found.** The obvious reading was that the gripper is modelled at under half its
rated strength, so the drive was raised from 10 N-m to the 24.96 N-m that
produces Robotiq's rated 235 N at the measured transmission ratio. On a matched
grid that measured *worse*, 62 N against 66 N, and lost capture entirely above
0.65 rad. Same mechanism as the capture/hold split: a harder squeeze drives an
unconstrained payload instead of holding it. The change is reverted; the
constant and the reasoning stay in `assets.py` so nobody repeats it.

Two things the measurements still say, for whoever picks this up:

1. **The remaining give is rotational, not axial.** Slip decomposes into 0.7 mm
   of median axial movement against 0.054 rad of angular, and the blade only
   levers once the pull has dragged it clear of the rails. An interface feature
   opposing yaw would raise the margin further.
2. **The pull test is harsher than extraction.** It applies a constant force for
   1.5 s to a body the rails do not constrain along x, so the blade travels up
   to 0.29 m and leaves the channel mid-measurement. Any change here must be
   argued as a protocol correction and re-run against the earlier
   configurations, not adopted because it flatters the number.

The three skills are registered as separate gated tasks
(`Isaac-ZeroG-Blade-GrapplePin-Grasp-v0`, `-Extract-v0`, `-Insert-v0`) and all
three now pass `train.py --smoke`. `scripts/run_grapple_skills.sh` runs the whole
pipeline: smoke, train, evaluate on three held-out seeds across three curriculum
stages, and pool into one gated report under `evidence/`.

Extract ends with the blade fully clear of the slot mouth, about 495 mm of
travel; that was decided with the owner on 2026-08-09. Its final pose puts the
wrist about 200 mm in front of the robot's own base, folded, and that has never
been checked kinematically, so a failure to reach is the first thing to suspect
if extraction plateaus. Insert starts at the certified staging pose because that
is the arm pose that has been calibrated; chaining all three needs one more
calibration run.

Tools that now exist for this work:

- `scripts/measure_gripper_envelope.py` measures the 2F-85 from its collision
  meshes in the wrist frame. **Never infer pad locations from body origins**:
  every 2F-85 body in this asset is collapsed within 18 mm of the flange, and
  reading them has produced one retracted claim already.
- `scripts/calibrate_grasp_pose.py` servos the tool frame onto a target with the
  task's own differential IK. It now solves orientation as well as position,
  which a head-on pose needs. Four traps are fixed: the IK delta is applied in
  the robot *root* frame; the episode timeout must be disabled or the arm resets
  mid-solve; the fingers must be held open or they foul the interface they are
  being driven around; and the blade must be pinned, or the swinging arm shoves
  a free body in zero gravity and every stage returns the same answer.
- `scripts/grasp_diagnostics.py` is the gate. A grasp counts as formed only when
  drive torque rises off its 1e-5 N-m noise floor. Quote the finest force grid
  you ran: the reported capacity is the largest force below the *first*
  environment that slipped, so coarse grids flatter the result by 10%.

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
- Do not advance to Phase 3 while L2 is unpromoted and L3 settling is blocked.
- Keep `.deps`, logs, datasets, checkpoints, artifacts, and videos out of Git.

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
| Grasp physics before any grasp PPO | `scripts/grasp_diagnostics.py`, `evidence/grapple_pin_axial_pull_gate.json` |
| Gripper geometry, ever | `scripts/measure_gripper_envelope.py`, `evidence/gripper_collision_envelope.json` |
| Head-on capture task | `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py` |
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
