# Claude Opus Handover

Act as the senior robotics simulation engineer taking ownership of this repository. Preserve its evidence-first approach: diagnose one physical or learning bottleneck at a time, require deterministic held-out evaluation before promotion, and never describe a smoke test or attractive render as Sim2Real validation.

## Mission and positioning

Build a portfolio-quality autonomy stack for robotic replacement of modular compute hardware in microgravity. The immediate task is inserting a replacement server blade with a UR10e/Robotiq 2F-85 in NVIDIA Isaac Lab. The longer-term system should remove a failed module, stow it, acquire a replacement, insert it safely, and verify completion under uncertain contact, payload, mounting, illumination, and sensing.

Position this as research into **contact-rich orbital field servicing of modular compute hardware**, not as a flight-ready space data center. The interview value is the disciplined workflow: GPU-parallel RL, physics-gap diagnosis, curriculum design, measurable promotion gates, perception/control separation, and an honest Sim2Real validation plan.

## Read only what is needed

Start with:

1. `README.md` — public claim and commands.
2. `docs/architecture.md` — code/data flow.
3. `docs/sim2real_matrix.md` — modeled gaps and missing measurements.
4. `src/zero_g_blade_swap/tasks/blade_swap/rigid_grasp_insertion_env_cfg.py`.
5. `src/zero_g_blade_swap/tasks/blade_swap/mdp/insertion.py`.
6. `scripts/train.py` and `scripts/play.py`.

Do not ingest `.deps`, `logs`, checkpoints, or all task files unless the current diagnosis requires them.

## Verified state on 2026-08-07

Stack: native Windows 11, Isaac Sim 5.1, Isaac Lab v2.3.2 at `37ddf626871758333d6ed89cf64ad702aef127d0`, bundled Python 3.11, RL-Games PPO, RTX 5070 Ti Laptop GPU 12 GB.

The promoted checkpoint is local and intentionally ignored by Git:

```text
logs/rl_games/zero_g_blade_insertion_rigid_grasp/rigid_grasp_l0_fresh_seed60/nn/
last_zero_g_blade_insertion_rigid_grasp_ep_700_rew_74.81321.pth
```

SHA-256: `1635C0DA6464A34DE5D5D423D45D272AD0E19D808EA35B8502068F68B5043332`.

Deterministic evaluation on unseen seed 1060, 128 parallel environments:

| Reset distance | Result |
| --- | ---: |
| Near / curriculum stage 0 | 1,021 / 1,021 (100%) |
| Medium / stage 1 | 1,006 / 1,006 (100%) |
| Full / stage 2 | 1,001 / 1,001 (100%) |
| Total | 3,028 / 3,028, zero timeout/failure/non-finite termination |

This promotes **Level-0 secured-grasp insertion on one held-out seed**. It does not prove rail-contact robustness, learned grasping, perception, cross-seed repeatability, or real transfer. Repeat the three evaluations on at least two more unseen seeds before making a release artifact. `play.py` currently reports final live-state errors after automatic resets; fix it to accumulate terminal metrics before using those error fields as evidence.

Static validation: Ruff passed and 36/36 tests passed. GPU physics smoke passed for Levels 0, 1, and 2.

## What is implemented

- `ManagerBasedRLEnvCfg`, zero gravity, 120 Hz PhysX, 30 Hz control, differential IK, GPU-parallel PPO.
- Primitive rack/blade/caddy geometry and UR10e/Robotiq asset.
- A real PhysX fixed joint representing an **already-secured** blade with effectively zero settled tool/handle gap. Redundant handle/finger collision is disabled to prevent contradictory contact impulses.
- Anti-stall rewards: measured potential reduction is positive, standing still is negative, unfinished distance receives a timeout cost.
- A three-distance success curriculum and deterministic JSON evaluation.
- Five cumulative secured-grasp profiles:
  - L0 collision-free insertion — trained and promoted on seed 1060.
  - L1 wide side rails and larger pose error — physics smoke passed, untrained.
  - L2 tight 1.5 mm side clearance plus 5–15 kg mass — physics smoke passed, untrained.
  - L3 randomized side-rail friction and 10–120 N breakaway/viscous stiction — implemented but blocked.
  - L4 compliant floating mount and wrench pulses — implemented but blocked behind L3.
- Full-swap teacher/vision scaffolding, tiled RGB, orbital lighting/material/noise randomization, data collection and BC hooks. These are integration scaffolds, not converged policies.
- Sustained environment-only benchmarks previously passed at 1024 state and 256 vision environments; full PPO memory differs.

## Known failures and limitations

- L3 reaches valid insertion geometry but cannot consistently settle below velocity thresholds under the sampled high stiction. Do not hide this by loosening success thresholds or running long PPO. Inspect the force model and contact energy first.
- The visible lower shelf collider is disabled in the rigid-grasp task. Tight floor contact plus randomized friction caused non-physical lateral ejection. Side rails remain physical. Re-enable only after geometry/contact calibration.
- PhysX logs a startup warning that the fixed joint connects disjoint transforms and will snap them together. The settled gap is effectively zero, but the authored/reset joint frames should eventually be made consistent.
- The fixed joint is a task abstraction. The real Robotiq pad/handle contact task failed its axial pull gate and must not be called learned grasping.
- Primitive blade/rack geometry has no connector, latch, cable, chamfer, measured tolerance, or force-displacement curve.
- No wrist force/torque sensing, force limit, damage proxy, real UR10e, HIL rig, orbital acceleration data, calibrated camera, or radiation dataset exists.
- Visual noise and lighting ranges are engineering priors. The vision student has not been trained from the promoted insertion policy.
- Only one PPO training seed and one held-out evaluation seed support the new Level-0 result.
- Isaac Sim 5.1's published VRAM minimum exceeds this laptop's 12 GB; use benchmark-driven environment counts.

## Recommended next work, in order

1. **Make evaluation release-grade.** Capture terminal pose/velocity metrics before reset, evaluate seeds 1060/2060/3060, report Wilson confidence intervals, cycle-time distribution, and categorized failures.
2. **Promote L1 then L2.** Fine-tune the epoch-700 checkpoint sequentially, never skipping held-out gates. Require at least 90% overall and 80% in every mass/contact bucket.
3. **Resolve L3 physically before training.** Plot rail force, blade velocity, contact impulse, and action versus time. Check whether the stiction implementation injects energy or chatters at zero velocity. Prefer a continuous, measured friction/connector force curve and force/admittance-limited insertion over arbitrary reward changes.
4. **Add industrially meaningful hardware proxies.** Replace primitive rail/handle geometry with measured, non-proprietary CAD; add chamfers, latch/connector engagement, wrench sensing, peak force/impulse limits, and abort/recovery behavior.
5. **Learn grasp/extraction as a separate skill.** Validate force closure and axial pull with real pad collision before PPO. Do not remove the fixed joint from insertion until that gate passes.
6. **Compose skills under a task manager.** Prefer separately validated reach/grasp/extract/stow/insert policies over one monolithic sparse-reward policy.
7. **Perception and adaptation.** Collect RGB/proprio/action data from the robust state policy, behavior-clone a vision policy, then fine-tune with asymmetric PPO. Keep ground-truth blade pose out of the deployable actor. Consider a real-world residual/adaptation layer only after safe hardware instrumentation exists.
8. **Portfolio release.** Publish a short side-by-side video, learning curve, held-out failure montage, benchmark JSON, model card, and a one-page “claim versus evidence” table. Put checkpoints/videos in GitHub Releases, not Git history.

## Research basis

The implementation deliberately follows established patterns rather than claiming novelty:

- NVIDIA's Isaac Lab gear-insertion workflow assumes the part is already grasped, trains insertion in simulation with domain randomization, and deploys through a separate robot interface: <https://isaac-sim.github.io/IsaacLab/develop/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html>.
- Isaac Lab supports manager-based modular rewards/observations/events and mass/material randomization with inertia recomputation: <https://isaac-sim.github.io/IsaacLab/v2.3.2/genindex.html>.
- NVIDIA recommends plausible randomization and curriculum rather than making the final distribution so broad that learning is impossible: <https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html>.
- OpenAI Dactyl demonstrated dynamics/appearance randomization and separated vision-based pose estimation from control, while explicitly noting that contact modeling remains difficult: <https://openai.com/index/learning-dexterity/> and <https://arxiv.org/abs/1808.00177>.
- OpenAI's ADR work explains why randomization ranges should expand with capability instead of being maximized at the start: <https://openai.com/index/solving-rubiks-cube/>.
- NVIDIA SPARR is relevant future work for contact-rich assembly: simulation base policy plus a real-world visual residual corrected real dynamics/state-estimation errors. Treat it as a roadmap option, not a completed feature: <https://research.nvidia.com/labs/srl/projects/sparr/>.

## Operating rules

- Preserve exact zero gravity and the 30 Hz policy / 120 Hz physics timing unless an experiment explicitly tests a change.
- Never resume a checkpoint after changing action or observation dimensions.
- Change one failure category per experiment and save a JSON report.
- A scripted controller is allowed only as a physics feasibility test; demonstrations must use a checkpoint.
- Never call a fixed joint, compliant spring, or scripted action a learned grasp.
- Do not advance to Phase 3 while L1/L2 are unpromoted and L3 settling is physically blocked.
- Keep `.deps`, logs, datasets, checkpoints, artifacts, and videos out of Git.

Begin by auditing the terminal-metric bug and producing a concise proposal for L1 evaluation/training. Do not redesign the entire repository in the first turn.
