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

## Next action, decided 2026-08-10: pivot to pose uncertainty

**P0 is done.** The three head-on grapple-pin skills, the eight-phase swap task,
and their reward/termination/curriculum classes were deleted on 2026-08-10. The
capture scene, the pin geometry, the interface specification, every file in
`evidence/`, the contact-force machinery, and the evaluator all survive. The
visual-randomization machinery was repointed at the insertion scene as
`Isaac-ZeroG-Blade-Insertion-Vision-v0`, which is untrained scaffolding for P3.
`docs/status.md` records the smoke sweep and the two pre-existing failures.

**Next is P1: the pose-belief insertion task and its force-blind ablation.**
The full plan is `C:\Users\tryak\.claude\plans\with-this-literature-research-merry-gray.md`.

Why the direction changed. Every RL task in this repo trains against a task that
contains no uncertainty. The policy observes `insertion_goal_error`, derived
from `attached_blade_pose_world`, which is simulator ground truth; reset noise
randomizes the initial condition and the policy is then told the exact resulting
error. With a rigid known object on a constrained axis and full observability,
that is a motion-planning and force-control problem, and a scripted controller
would solve it. RL cannot demonstrate its value there, which is why three
hand-rolled skills cost a night of GPU and certified nothing.

The target result is now one falsifiable plot: **success rate against
pose-belief error, force-aware policy versus force-blind ablation.** That is the
axis IndustReal and FORGE are evaluated on, and the repo already owns both
halves it needs: a working `BladeContactWrenchObservation` and a certified
force-feedback task lineage.

Adopt established formulations rather than inventing reward terms. IndustReal
(sampling-based curriculum, SDF dense reward, simulation-aware policy update)
transfers contact-rich assembly at 83-99% over 600 trials on a UR10e, the same
arm as here. FORGE (arXiv 2408.04587) targets force-aware manipulation under
pose uncertainty directly. Note that the Grasp v1 failure recorded in
`docs/status.md`, where the policy succeeded at reset and never learned an
approach, is exactly the overfitting pathology IndustReal's sampling-based
curriculum exists to prevent.

What survived the prune and must not be deleted: the grapple pin geometry and
`docs/service_interface_spec.md`, everything in `evidence/` including the
negative results, the contact-force machinery in `mdp/insertion.py`, the
evaluator and its promotion gate, `TwoStageRobotiqAction` and
`hold_two_stage_grip` in `mdp/grapple.py` (they implement the capture/hold split
the 69 N result depends on), and the visual-randomization modules
(`OrbitalLightingRandomizer`, `RackMaterialRandomizer`,
`camera_rgb_with_radiation_noise`, `make_tiled_camera_cfg`, the student
scaffold), which now hang off `vision_insertion_env_cfg.py` instead of the
deleted swap task.

Adopt, do not invent. IndustReal's sampling-based curriculum samples the whole
initial-state range from the first step and raises only its *easy* bound as
success improves, which is the direct fix for the Grasp-v1 pathology below; the
mixture ramp in `InsertionSuccessRateCurriculum` does the opposite. FORGE
conditions the policy on a maximum allowable force `F_th` sampled per episode and
charges `-beta * max(0, ||F|| - F_th)`, rather than the two fixed penalty
profiles this repository already measured as ineffective.

What the head-on grapple pin established, and why it stays: it is the first
interface here to form a real grip, holding **69 N** of axial pull against the
66.4 N the insertion contact reaction demands, where flat pads on a smooth post
held about 6 N. The step that got there was recognising that **capture and hold
are two different commands** — a wedge converts closing force into thrust along
the pull axis, so a firm capture drives the payload away before it is taken.
Capture at 0.48 rad, firm to 0.68 once loaded. The window is narrow and
asymmetric: 0.44 holds 63 N, 0.52 holds 68 N, 0.56 collapses to 26 N. Bias low.

Its unfixed limitation: a single-point pin does not constrain yaw once the rails
release the blade, measured at 0.93 rad of blade rotation in failing
extractions. An anti-yaw feature is a legitimate second-generation result.


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
- Do not reintroduce the eight-phase swap task or the three grapple-pin skills.
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
| Grasp physics before any grasp PPO | `scripts/grasp_diagnostics.py`, `evidence/grapple_pin_axial_pull_gate.json` |
| Gripper geometry, ever | `scripts/measure_gripper_envelope.py`, `evidence/gripper_collision_envelope.json` |
| Head-on capture task | `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py` |
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
