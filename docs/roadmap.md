# Roadmap and research basis

## Recommended next work, in order

1. ~~**Make evaluation release-grade.**~~ Done 2026-08-08. `TerminalMetricsMixin`
   in `src/zero_g_blade_swap/evaluation.py` intercepts `_reset_idx`;
   `InsertionTerminalMetrics` records one row per completed episode;
   `scripts/aggregate_evaluation.py` pools runs and applies the gate.
2. ~~**Promote L1.**~~ Done 2026-08-08. See `docs/status.md`.
3. ~~**Promote L2.**~~ Done 2026-08-08. See `docs/status.md`.
4. **Resolve L3 physically before training.** Plot rail force, blade velocity,
   contact impulse, and action versus time. Check whether the stiction
   implementation injects energy or chatters at zero velocity. Prefer a
   continuous, measured friction/connector force curve and force/admittance-
   limited insertion over arbitrary reward changes. The Level-2 certification
   predicts where this breaks: terminal orientation error already consumes 97.8%
   of its tolerance, so orientation is the axis to instrument first.
5. **Add industrially meaningful hardware proxies.** Replace primitive
   rail/handle geometry with measured, non-proprietary CAD; add chamfers,
   latch/connector engagement, wrench sensing, peak force/impulse limits, and
   abort/recovery behavior.
6. **Learn grasp/extraction as a separate skill.** Validate force closure and
   axial pull with real pad collision before PPO. Do not remove the fixed joint
   from insertion until that gate passes.
7. **Compose skills under a task manager.** Prefer separately validated
   reach/grasp/extract/stow/insert policies over one monolithic sparse-reward
   policy.
8. **Perception and adaptation.** Collect RGB/proprio/action data from the
   robust state policy, behavior-clone a vision policy, then fine-tune with
   asymmetric PPO. Keep ground-truth blade pose out of the deployable actor.
   Consider a real-world residual/adaptation layer only after safe hardware
   instrumentation exists.
9. **Portfolio release.** Publish a short side-by-side video, learning curve,
   held-out failure montage, benchmark JSON, model card, and the one-page claim
   versus evidence table. Put checkpoints and videos in GitHub Releases, not Git
   history.

## Research basis

The implementation deliberately follows established patterns rather than
claiming novelty:

- NVIDIA's Isaac Lab gear-insertion workflow assumes the part is already
  grasped, trains insertion in simulation with domain randomization, and deploys
  through a separate robot interface:
  <https://isaac-sim.github.io/IsaacLab/develop/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html>.
- Isaac Lab supports manager-based modular rewards/observations/events and
  mass/material randomization with inertia recomputation:
  <https://isaac-sim.github.io/IsaacLab/v2.3.2/genindex.html>.
- NVIDIA recommends plausible randomization and curriculum rather than making
  the final distribution so broad that learning is impossible:
  <https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html>.
- OpenAI Dactyl demonstrated dynamics/appearance randomization and separated
  vision-based pose estimation from control, while explicitly noting that
  contact modeling remains difficult:
  <https://openai.com/index/learning-dexterity/> and
  <https://arxiv.org/abs/1808.00177>.
- OpenAI's ADR work explains why randomization ranges should expand with
  capability instead of being maximized at the start:
  <https://openai.com/index/solving-rubiks-cube/>.
- NVIDIA SPARR is relevant future work for contact-rich assembly: a simulation
  base policy plus a real-world visual residual corrected real dynamics and
  state-estimation errors. Treat it as a roadmap option, not a completed
  feature: <https://research.nvidia.com/labs/srl/projects/sparr/>.
