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
built on those measurements forms the project's first real grip and holds 69 N
against the 66.4 N the insertion reaction demands, where flat pads on a smooth
post held about 6 N. Three skills — capture, extract, insert — now train against
that pin and chain into two workflows that each run end to end in one continuous
episode: removal, which pulls a fully installed module 495 mm clear of the rack,
and installation, which seats one at 8.63 mm axial and 0.61 mm lateral error.
Both hold the module by pad-against-pin contact alone, with no fixed joint.
**Neither the chain nor the retrained policies driving it are certified**, and
that is the first thing the next session fixes. Full numbers, limitations, and
the pre-existing `train.py --smoke` probe defect live in `docs/status.md`.

## Next action, decided 2026-08-11: certify what exists, then fix the interface

**The pose-uncertainty pivot ran and its hypothesis was refuted.** Force-aware
and force-blind policies are indistinguishable up to 4 mm of pose-belief error,
and the force-aware arm is *worse* past it at roughly twice the contact force.
The cause is worth more than the hypothesis was: the rack's lead-in flares,
16.6 mm per side, do the final alignment mechanically. Remove them and **both**
policies score 0% even at zero belief error. That is not a failed experiment, it
is a module-side design requirement — the lead-in is load-bearing, and its
dimension sets the pose tolerance the whole system must hold. It belongs in
`docs/service_interface_spec.md`, which is the actual deliverable.
`Isaac-ZeroG-Blade-Insertion-Uncertain-v0` and `-UncertainBlind-v0` stay for the
record. Do not re-run that sweep expecting a different answer.

Chaining the three skills exposed four defects that no single-skill certification
could have caught, all the same shape: a skill that works alone assumes something
its neighbour does not provide. They are written up in `docs/status.md`. The one
worth internalising is that extraction had trained with ±0.0005 rad of reset
noise and had therefore seen exactly one arm configuration in its entire life.

### Do these in order

**1. Certify the policies the demo actually runs.** `evidence/` holds
certifications for grasp v2, extract v2 and insert v3. `run_workflow_demo.py`
loads grasp **v3**, extract **v4**, insert **v5**. Every figure currently quoted
about the demo describes a superseded policy, and the retrains changed reset
noise 40×. Run `play.py` per skill per seed with `--episode_metrics`, pool with
`scripts/aggregate_evaluation.py`, supersede the stale files. Do this before
touching geometry — it is also the before-baseline step 4 is measured against.

**2. Certify the chain itself, not only its parts.** Nothing in `evidence/`
covers a chained run; the two workflow videos are n=1. Extend
`run_workflow_demo.py` to run headless across many seeds and emit the same
episode rows the evaluator already pools, then gate it and report a success rate
with a Wilson interval. Until that report exists the videos are demonstrations,
not evidence, and every document must describe them that way.

**3. Fix insert's clock, not its policy.** Insert scores 6.5% at full distance
alone, and 473 of 479 failures are timeouts with the median module 11.29 mm from
a 12 mm tolerance and 0.62 mm laterally against 2.5 mm. Successful insertions
take 11.77 s against a 12 s episode. It is slow, not unreliable. Lengthen the
episode, retrain, re-certify. Per-skill certification and the chain currently
disagree about what the task is, because the chain grants 45 s — reconcile them
rather than quoting whichever number reads better.

**4. Design an anti-yaw grapple pin and measure it.** A single-point pin does not
constrain yaw once the rails release the module: 0.93 rad in failing extractions,
and the return leg degrades the grip from 15 mm to 35 mm. Slowing the replay
fourfold makes it *worse*, so this is rotation under sustained load, not an
acceleration artefact. It is the one thing blocking a full remove-and-replace
round trip, and a keyway or bearing flats is a legitimate second-generation
interface result. Re-run `scripts/grasp_diagnostics.py` and the axial pull gate
on the new geometry, expect to retrain capture because changing the pin changes
the contact, then re-certify the chain and record the round trip.

**5. Only then, perception readiness.** The pose envelope already implies the
requirement: the skills tolerate about 4 mm of pose error before the flares stop
catching, so that is the accuracy perception must deliver. Write it into the spec
as a stated requirement rather than rediscovering it later.
`docs/perception_plan.md` carries a blocking finding — the authored camera
resolves 4 mm as **0.13 pixels** — so the camera has to be fixed before a single
image is collected. The full original plan is
`C:\Users\tryak\.claude\plans\with-this-literature-research-merry-gray.md`.

Adopt established formulations rather than inventing reward terms. IndustReal
(sampling-based curriculum, SDF dense reward, simulation-aware policy update)
transfers contact-rich assembly at 83-99% over 600 trials on a UR10e, the same
arm as here. Its curriculum samples the whole initial-state range from the first
step and raises only its *easy* bound as success improves, which is the direct
fix for the Grasp-v1 pathology in `docs/status.md`; the mixture ramp in
`InsertionSuccessRateCurriculum` does the opposite. FORGE (arXiv 2408.04587)
conditions the policy on a per-episode maximum allowable force rather than the
two fixed penalty profiles this repository already measured as ineffective.

Must not be deleted. The three grapple-pin skills were pruned on 2026-08-10 and
restored on 2026-08-11 once it was clear the uncertainty pivot had no
demonstrable artefact; they are now the only thing in the repo that grasps
anything. Also keep the grapple pin geometry and
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
- Never quote a success rate without checking that the certification in
  `evidence/` names the same policy version the demo loads. This rule exists
  because grasp v3, extract v4 and insert v5 were quoted using v2/v2/v3 numbers.
- A recorded video is a demonstration. A pooled multi-seed report is evidence.
  Never let the first stand in for the second, in any document or commit message.
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
