# Roadmap and research basis

## The current line of work, decided 2026-08-10

Items 1 to 13 below are the pre-pivot roadmap and are kept for provenance. The
active plan is different, because every task in that list trains against a
problem containing no uncertainty: the policy is told its exact pose error. The
staging now is

- **P0, done.** Delete the three grapple-pin skills and the eight-phase swap
  task; repoint the visual randomizers at the insertion scene. See
  `docs/status.md`.
- **P1, active.** One task on the certified force-feedback lineage where the
  actor sees `true_error + episode_bias + per_step_noise` instead of ground
  truth, with the critic keeping ground truth. Train a force-aware policy and a
  force-blind ablation from scratch on an identical schedule, then sweep the bias
  and publish success rate against pose-belief error for both.
- **P2.** Repeat P1's best configuration with the blade held by the physical
  grapple pin instead of the fixed joint, which connects the interface
  specification to the learning result.
- **P3.** Replace the injected belief with a pose regressed from the 64x64 tiled
  camera under randomized orbital lighting and rack materials. Do not start
  before P1 certifies.
- **P4.** Package: README leading with the curve, demo clips from a trained
  checkpoint only, evidence index.

Two established formulations are adopted rather than reinvented. IndustReal
(RSS 2023) transfers contact-rich assembly at 83-99% over 600 real trials on a
UR10e, the same arm as here; its sampling-based curriculum samples the whole
initial-state range from the first step and raises only the easy bound as
success improves, which is the direct fix for the Grasp-v1 pathology recorded in
`docs/status.md`. FORGE (arXiv 2408.04587) targets force-aware manipulation under
pose uncertainty, conditioning the policy on a per-episode maximum allowable
force and charging a hinge penalty above it, and injects the fixed part's pose
error as a per-episode constant, which is exactly the belief model P1 needs.

## Pre-pivot roadmap, kept for provenance

1. ~~**Make evaluation release-grade.**~~ Done 2026-08-08. `TerminalMetricsMixin`
   in `src/zero_g_blade_swap/evaluation.py` intercepts `_reset_idx`;
   `InsertionTerminalMetrics` records one row per completed episode;
   `scripts/aggregate_evaluation.py` pools runs and applies the gate.
2. ~~**Promote L1.**~~ Done 2026-08-08. See `docs/status.md`.
3. ~~**Promote L2.**~~ Done 2026-08-08. See `docs/status.md`.
4. ~~**Characterize the capability envelope.**~~ Done 2026-08-09. Pose error is
   the binding axis (half-success near 7× trained noise, failing by lateral
   divergence); blade mass is not a meaningful axis in this regime. See
   `docs/status.md`.
5. ~~**Constrain contact force by reward shaping.**~~ Tried 2026-08-09 and it did
   not work. A force budget and abort exist and hold 100% success, but two
   penalty strengths changed mean contact by 2.6% and impulse not at all. See
   `docs/status.md` for the arithmetic and the two surviving hypotheses.
6. ~~**Give the policy force feedback.**~~ Done 2026-08-09, and it worked on one
   axis of two. Seven contact-force values were added to the observation and the
   policy was retrained from scratch against a matched no-feedback control on an
   identical schedule. Contact impulse fell 59% at the mean and 89% at the
   median; peak contact force did not move. Sensing was the binding constraint
   on sustained rubbing, and peak force really is geometrically irreducible in
   this action space. See `docs/status.md`.
7. **Try an admittance or impedance action space.** This is the untested half of
   the item above and the only remaining lever on *peak* contact force, which
   force feedback left unchanged at about 10.5 N mean and 31 N p95 at full
   reset distance. Replace position-based differential IK with a controller
   whose commanded motion yields to measured force, and retrain; the action
   interface changes, so no checkpoint can be resumed. Compare against
   `evidence/force_feedback_certification.json` on identical axes.
8. **Build a head-on grapple pin, then learn capture, extraction, and insertion.**
   Decided 2026-08-09 after friction grasping was shown to fail structurally: a
   downward approach cannot resist a sideways pull except by friction, and the
   measured axial capacity is about 6 N against 66.4 N required. Align the
   approach with the extraction axis and capture a flared pin by form closure,
   the pattern Canadarm2 and NASA's ORU standard already use. Then certify
   capture, extraction, and insertion as three separately gated skills, which is
   the decomposition a replacement demonstration needs.

   - Superseded, but its measurements still hold. Reaching the handle was fixed
     on 2026-08-09: the tool frame was moved onto the pads, the finger command's
     inverted sign was corrected, and a raised grapple post replaced a 30 mm tab
     that the pads could never straddle. A grip does now form, at the full
     10 N·m drive limit. It then ejects the blade, which is what sent the design
     to form closure. Do not remove the fixed joint from insertion until a grasp
     gate passes on three held-out seeds.
     See `evidence/grasp_axial_pull_gate.json`.
9. **Resolve L3 physically before training.** Plot rail force, blade velocity,
   contact impulse, and action versus time. Check whether the stiction
   implementation injects energy or chatters at zero velocity. Prefer a
   continuous, measured friction/connector force curve and force/admittance-
   limited insertion over arbitrary reward changes. Instrument the lateral axis
   first: the envelope sweep showed lateral divergence, not orientation, is what
   actually terminates failing episodes, even though orientation consumes 97.8%
   of its tolerance at Level 2.
10. **Add industrially meaningful hardware proxies.** Replace primitive
   rail/handle geometry with measured, non-proprietary CAD; add chamfers,
   latch/connector engagement, wrench sensing, peak force/impulse limits, and
   abort/recovery behavior.
11. **Compose skills under a task manager.** Prefer separately validated
   reach/grasp/extract/stow/insert policies over one monolithic sparse-reward
   policy.
12. **Perception and adaptation.** Collect RGB/proprio/action data from the
   robust state policy, behavior-clone a vision policy, then fine-tune with
   asymmetric PPO. Keep ground-truth blade pose out of the deployable actor.
   Consider a real-world residual/adaptation layer only after safe hardware
   instrumentation exists.
13. **Portfolio release.** Publish a short side-by-side video, learning curve,
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
