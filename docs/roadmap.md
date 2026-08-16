# Roadmap and research basis

## The goal, decided 2026-08-15: one module, two slots

**The eventual capability is a relocation:**

    GRASP -> REMOVAL -> RELOCATION -> INSERT

A module is captured in slot 1, pulled clear of the rack, carried to an empty
slot 2 beside it, and seated there. That is the operation ISS performs as ORU
changeout, and it is the first thing this project would build that is *servicing*
rather than assembly. Everything demonstrated so far is half of it.

**It is gated, deliberately, on removal.** A relocation is the product of four
stages, and chained numbers here have consistently come in *below* the product of
their parts. At today's rates the relocation would complete well under half the
time, and a flagship demonstration that fails more often than it succeeds is
worse than a smaller one that works. So the order is fixed:

1. **Close the installation chain.** Capture + insert, camera in the loop, sits
   at 80.38%. Diagnosed: 77 of its 113 failures are captures overrunning their
   budget, caused by the skill certifying on 20 mm of grip error while the chain
   waits for 10 mm. Being retrained against the aligned criterion now.
2. **Make removal work in the chain.** Extract certifies at 68.36% alone and the
   chained removal at 14.06%. This is the single highest-value piece of work in
   the project and it is the gate on everything below.
3. **Only then, two slots.** Second slot geometry, a real lateral transit, and
   insertion retrained for a second goal pose.

**Do not start step 3 before step 2 certifies.** Building the scene first
produces a demonstration that fails three times in four and invites exactly the
reading the evidence does not deserve.

## The current line of work, decided 2026-08-10

Items 1 to 13 below are the pre-pivot roadmap and are kept for provenance. The
active plan is different, because every task in that list trains against a
problem containing no uncertainty: the policy is told its exact pose error. The
staging now is

- **P0, done, and half of it reversed.** The eight-phase swap task is deleted and
  stays deleted; `tests/test_configuration_contract.py` fails if it returns. The
  three grapple-pin skills went with it on 2026-08-10 and were **restored on
  2026-08-11**, because each had failed for a cause identified and corrected in
  the same session and then never retested, and because deleting them removed the
  only path to a servicing demonstration. Retested, they chain. The visual
  randomizers stay repointed at the insertion scene. See `docs/status.md`.
- **P1, measured 2026-08-10, hypothesis refuted.** Force sensing did not extend
  the tolerable pose error and is worse beyond the trained range. The diagnosis
  promotes item 7 below from an optimisation to a precondition: force has to be
  *actionable*, not merely observable. The immediate cheap probe is to re-evaluate
  both existing checkpoints with the lead-in flares disabled, which costs no
  training and settles whether the ramp was doing the alignment.
- **P1 build, for reference.** `Isaac-ZeroG-Blade-Insertion-Uncertain-v0` and its
  force-blind ablation. The slot physically moves by an amount the actor is never
  told, rather than a bias being added to a reported error, because on this
  workcell an injected bias is recoverable from the observed tool pose. The
  critic keeps ground truth. See `docs/status.md` for the mechanism and the two
  faults found while building it.
- **P2, and the whole of it is now evidence rather than capability.** The module
  is held by the physical grapple pin instead of the fixed joint, and capture,
  extract and insert chain into two servicing workflows that run end to end in
  one continuous episode. Every checkpoint the demonstration loads is certified
  as the version it loads, and both chains are certified across three held-out
  seeds with Wilson intervals. What is left is the success *rates*.

  **The last item on this list used to say "constrain yaw", and that was wrong
  in three separate ways, each of which cost a session.** The rotation was never
  decomposed and turned out to be split evenly across two axes; two mechanical
  features built against one of them were both measured as net negatives, the
  yoke costing insertion 67 points; and what actually moved extraction from
  0.00% was three things with nothing mechanical among them — an episode shorter
  than the median success, an attitude term two orders of magnitude below the
  progress term it competed with, and an action space that could rotate at
  0.24 rad/s while the module rotated at up to 0.767. **Do not add a fourth
  interface feature.** The remaining lever is roadmap item 7.
- **P3, and it is not the wall this file implied.** Replace the injected
  displacement with one regressed from the tiled camera under randomized orbital
  lighting and rack materials. The "blocking finding" that a 64x64 camera
  resolves a 4 mm displacement as 0.13 pixels was a **focal length**, derived in
  `docs/perception_plan.md` and never applied: 18 mm to 180 mm puts it at
  1.31 px, and that change is now in `scene_cfg.make_tiled_camera_cfg`.
  `scripts/check_camera_scale.py` renders a frame and gates on millimetres per
  pixel, on whether the interface is genuinely inside a 7-degree cone, and on
  the image carrying signal at all. Run that gate whenever the optics change;
  it costs one frame. What waits for a certified chain is *training* a
  perception policy, because a student distilled from an uncertified teacher
  inherits its failures. The accuracy it has to reach is not a guess: 4 mm
  laterally, written into `docs/service_interface_spec.md` and derived from the
  measured envelope.
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
7. **Try an admittance or impedance action space. Now the main open question.**
   The pose-belief ablation measured that force in the observation buys nothing
   under a stiff position-controlled arm, because no action yields to it; both
   2026 papers make force actionable instead (hybrid position/force selection in
   arXiv 2604.19677, commanded force direction in arXiv 2602.14174). This is also
   the only remaining lever on *peak* contact force, which
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
   - Closed 2026-08-11. The head-on pin passed its axial pull gate at 69 N
     against the 66.4 N required, and all three skills are trained on it and
     chained. Its one unfixed limitation is yaw: see item 14.
14. **Constrain yaw on the interface, not in the controller.** A single-point
   tapered pin clamped by flat pads cannot resist rotation about the closing
   axis, because the pads' contact normals lie along that axis and a normal force
   cannot oppose a moment about its own direction. Measured: 0.93 rad of module
   rotation in failing extractions, and a return leg that degrades the grip from
   15 mm to 35 mm and gets *worse* when replayed fourfold slower, so it is
   rotation under sustained load rather than an acceleration artefact. This is
   the only thing blocking a full remove-and-replace round trip, and it is a
   second-generation interface result rather than a bug fix. The measured
   gripper envelope says where the feature fits: no gripper body other than an
   inner finger reaches past 0.1245 m from the flange at any closure, so a wall
   narrower than the 17.5 mm knuckles has a 37.6 mm band to live in.
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
