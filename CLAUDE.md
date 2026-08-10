# Agent handover

Act as the senior robotics simulation engineer who owns this repository.
Preserve its evidence-first approach: diagnose one physical or learning
bottleneck at a time, require deterministic held-out evaluation before
promotion, and never describe a smoke test or an attractive render as Sim2Real
validation.

## Mission

Build a portfolio-quality autonomy stack for robotic replacement of modular
compute hardware in microgravity. The active skill is inserting a replacement
server blade with a UR10e and Robotiq 2F-85 in NVIDIA Isaac Lab. The longer-term
system removes a failed module, stows it, acquires a replacement, inserts it
safely, and verifies completion under uncertain contact, payload, mounting,
illumination, and sensing.

Position this as research into **contact-rich orbital field servicing of modular
compute hardware**, not as a flight-ready space data center. The value is the
disciplined workflow: GPU-parallel RL, physics-gap diagnosis, curriculum design,
measurable promotion gates, perception/control separation, and an honest
Sim2Real plan.

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

The head-on grapple pin is now **built, calibrated, and measured**, and it is
the first interface in this project's history to form a real grip: drive torque
saturates the 10 N-m limit in 363 of 363 environments and the pin holds 59 N of
axial pull within 2 mm of slip, against about 6 N for flat pads on a post. It
still **fails the 66.4 N gate by roughly 10%**, so no skill has been trained on
it. Full numbers, including why the flared-head design had to become a tapered
wedge, are in `docs/status.md`.

**Grip force has been tested and refuted.** The obvious fix was that the
gripper is modelled at under half its rated strength, so the drive was raised
from 10 N-m to the 24.96 N-m that produces Robotiq's rated 235 N at the measured
transmission ratio. On a matched grid that measured *worse*, 62 N against 66 N,
and lost capture entirely above 0.65 rad of closure. A wedge converts closing
force into thrust along the pull axis, so in zero gravity a harder squeeze
drives an unconstrained payload instead of holding it. The change is reverted;
the constant and the reasoning stay in `assets.py` so nobody repeats it.

Remaining levers, in the order they are worth trying:

1. **Constrain rotation.** This is where the measurement points. The collar
   holds along the pull axis at 1.1 mm of median axial slip while the blade
   levers at 0.166 rad p95, and it only levers after the pull has dragged it
   clear of the rails. An interface feature that opposes yaw, or an extract
   skill that keeps the blade railed for longer, attacks the actual failure.
2. **Steepen the taper.** Nearly exhausted: capacity goes as the sine of the
   taper angle, and 24.2 degrees already puts the wedge's free end at 70 mm
   inside an 87.08 mm aperture, leaving 8.5 mm of approach clearance a side.
3. **Reconsider the pull-test protocol itself.** It applies a constant force for
   1.5 s to a body the rails do not constrain along x, so the blade travels up
   to 0.29 m and leaves the channel mid-measurement. That is arguably harsher
   than extraction, where the gripper and the rails share the load. Any change
   here must be argued as a protocol correction and re-run against the earlier
   configurations, not adopted because it flatters the number.

Then train and certify grasp, extract, and insert as separate gated skills.
Extract ends with the blade fully clear of the slot mouth, about 495 mm of
travel; that was decided with the owner on 2026-08-09.

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
