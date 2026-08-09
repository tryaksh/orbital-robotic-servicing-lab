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
measured bug, not by training: the tool frame the IK drives sits 165.6 mm from
the physical Robotiq pads, so the pads never reach the handle and hold 0 N of
the 66.4 N required. Full numbers, limitations, and the
pre-existing `train.py --smoke` probe defect live in `docs/status.md`.

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
| Grasp physics before any grasp PPO | `scripts/grasp_diagnostics.py`, `evidence/grasp_axial_pull_gate.json` |
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
