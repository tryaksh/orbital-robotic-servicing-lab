# Agent handover

You own this repository. Act as the senior robotics simulation engineer who does.
Work in long autonomous blocks. Do not ask permission to start a run. Ask only if
you are genuinely stuck and no measurement can unstick you.

## The goal

A **relocation**: one module, two slots side by side.

```
CAPTURE  ->  EXTRACT  ->  TRANSIT ACROSS  ->  INSERT INTO SLOT 2
```

That is ORU changeout, and it is the deliverable. Everything else is a step
toward it. The demonstration has to be industrially credible and paper-worthy:
every skill certified, the full chain certified, every number naming its evidence
file.

## Where things stand

Deterministic evaluation, three held-out seeds, Wilson interval, terminal state
captured before auto-reset. Promoted: **capture v5, extract v13unsat, insert v6**.

| | Result | Gate 95% | Evidence |
| --- | ---: | --- | --- |
| Capture | 96.10% | pass | `grapple_grasp_v5_certification.json` |
| Extract | 99.02% | pass | `grapple_extract_v14reset_certification.json` |
| Insert, alone | 95.57% | pass | `grapple_insert_v6_certification.json` |
| **Removal chain** | **98.78%** | **pass** | `workflow_remove_retain_certification.json` |
| **Install chain** | **84.38%** | **short** | `workflow_install_final_certification.json` |
| Install, camera | 80.38% | — | `vision_workflow_camera_certification.json` |
| Install, oracle | 80.38% | — | `vision_workflow_oracle_certification.json` |
| Install, blind | 43.58% | — | `vision_workflow_blind_certification.json` |
| Insert **inside the chain** | ~80% | — | derived; this is the one gap |
| Interface axial hold | 69 N vs 66.4 N required | pass | `grapple_pin_axial_pull_gate.json` |
| Module pose from 64x64 RGB | 1.75 mm mean | — | `module_pose_head.json` |
| Onboard compute | 0.73 ms CPU, 2.2% of period | — | `inference_budget.json` |

Promoted checkpoints, under `logs/rl_games/zero_g_blade_insertion_contact/`:

```
grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth
grapple_extract_l0_seed70_v13unsat/nn/last_zero_g_blade_insertion_contact_ep_5700_rew__148.17932_.pth
grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth
```

## The plan

Work these in order. Each has a gate. **Do not start the next before the gate
passes.** Commit after each with the numbers in the message.

### 1. Insert, trained inside the chain — closes the install chain

The only gap. Insert scores 95.57% alone and ~80% on the states the chain hands
it. **Four attempts to reproduce that hand-off as a reset distribution have
failed** (see *Do not retry*). The hand-off is a trajectory and a controller
state, not a pose, so stop approximating it and train in place.

Build a training task that runs the real capture: reset the capture scene, step
the frozen capture policy until the workflow's hand-off condition fires, latch
the grip, then hand control to the learning policy for the insert phase.
`scripts/run_workflow_demo.py` already contains every piece — the phase machine,
the hand-off predicate, `hold_latch` — so lift the driver rather than rewriting
it.

- Fine-tune from insert v6. Reward change only, no dimension change.
- Gate: **install chain >= 95%**, `scripts/certify_workflow.sh install`.
- Then re-run all three vision arms, `scripts/certify_vision_workflow.sh`.
- Budget: ~1 session.

### 2. Second slot geometry

- Mirror `INSERTION_SLOT_LEFT/RIGHT_GUIDE_CFG` and the lips and lead-in flares at
  **y = -0.22 m**. Module is 0.16 m wide, so that leaves 60 mm between modules.
- **New registration.** Do not modify the single-slot tasks; that discipline is
  why the promoted L0/L1/L2 results have survived every change.
- Reachability is already verified: converged 6-DoF IK reaches y = -0.22 to
  0.0060 mm. Re-verify with `scripts/calibrate_grasp_pose.py --task
  Isaac-ZeroG-Blade-GrapplePin-Capture-v0 --steps 3000 --pin_blade --finger_joint
  0.02 --target_offset 0 -0.22 0`.
- Gate: the scene builds, `train.py --smoke` passes, IK converges.
- Budget: ~half a session.

### 3. Insert retrained for slot 2

A laterally offset slot is out of insert's distribution. Solve the slot-2 staging
arm pose with the converged calibrator, add it as a second curriculum entry,
fine-tune.

- Gate: **insert >= 95% on both slots**, `certify_demo_policies.sh Insert`.
- Budget: ~1 session.

### 4. Lateral transit

Today's transit retraces the extraction path backwards, deliberately: a direct
move takes the DLS IK through a near-singularity and swings the shoulder 74
degrees. A lateral move is a new motion.

- Record waypoints from the converged IK solve and follow them, which is what the
  current transit does and why it works. Do not fly it open-loop.
- **The module is free during transit.** Set `retain_latch` for the transit and
  clear it before insertion. See rule 1 — this is not optional, it is what took
  removal from 0.00% to 98.78%.
- Gate: module held to under 20 mm grip error across the whole transit.
- Budget: ~half a session.

### 5. Certify the relocation chain

`capture -> extract -> lateral transit -> insert(slot 2)`, one continuous
episode, real contact throughout.

- Gate: **>= 95%**, three held-out seeds, zero instability, zero non-finite.
- Chained numbers here have consistently landed *below* the product of their
  parts. If it does, trace the hand-offs with `--handoff_trace` before retraining
  anything.
- Budget: ~1 session.

### 6. Perception reads the rack

With two slots the camera must report *which* slot is occupied, not only where
the module is. That upgrades the claim from "locates a part" to "reads the state
of the rack" and it is the version worth demonstrating.

- Add a slot-occupancy output to the pose head. Re-run all three vision arms.
- Gate: camera arm within 10 points of oracle, blind arm clearly below both.
- Budget: ~1 session.

## Non-negotiable rules

1. **Any phase that waits must either command or retain.** An arm that stops
   commanding while gripping a module in zero gravity is not holding still, it is
   being pushed: the wedge turns closing force into thrust along the pull axis.
   Measured twice on two workflows — every chained removal fired its predicate and
   failed the 0.70 s re-check until `retain_latch` was added (0/570 -> 569/576),
   and a 2 s idle pause in the install chain costs 84.90% -> 21.35%, recovered to
   68.75% by retaining. **Capture gently, hold hard to move, retain once free.**
2. **Read constants, never restate them.** Six failures here came from one number
   living in two places: the action scales, the settling velocity limits, the
   hand-off grip tolerance, and the attitude penalty clamp.
3. **A number chosen for one purpose will silently decide what a policy can
   learn.** Before blaming a policy, check whether the objective still has a
   gradient where the policy sits. `grip_retention_penalty` saturated at 0.325 rad
   and the policy parked at 0.3538 — extraction went 0.00% -> 68.62% by unclamping
   it. Read the per-term tfevents under the run's `summaries/`.
4. **A skill must be trained across the states its predecessor actually
   produces.** The most repeated defect in this project.
5. **Before believing a reconstructed distribution, run the unchanged successor
   policy on it** and check it scores what it scores in the real chain. Five
   minutes. It would have saved two training runs.
6. **Before believing a probe, prove it moves what it measures.** A control that
   changes nothing means the probe is broken, not that the effect is absent.
7. **Before believing a solver, converge it.** A pose was called unreachable on a
   400-step IK residual; at 3,000 it converges to 0.0060 mm. Use the **Capture**
   task for pose calibration — Extract holds the arm still for its first second
   and every offset including zero reports unconverged.
8. **Never weaken a success threshold to make a gate pass.** Correcting a reset
   that produces *unwinnable* states is different, allowed, and must be stated
   with the measurement — as the extract reset fix was.
9. **A skill certification is not evidence about the chain.** Certify both.
10. **Never quote a rate without `scripts/check_evidence_currency.py`**, and check
    the report's timestamp against `git log -S` on the criterion it uses. Every
    extraction figure this project published was invalidated by a criterion change
    an hour after certification, and nobody noticed for a session.
11. Zero gravity, 30 Hz policy / 120 Hz physics. Never resume across an action or
    observation dimension change. Resuming across physics or reward changes is
    allowed and is how most policies here were made — say so when you do.
12. One Isaac process at a time. **Check for orphans before every launch**
    (`Get-Process kit`); one survived five hours and slowed everything ~40%.
13. 512 environments costs ~0.9 GB of 12 GB. 1024 is likely safe; verify with
    `nvidia-smi` in the first minutes.
14. `train.py`'s redirected stdout is block-buffered and lags minutes. Judge
    progress from `summaries/` tfevents and the `nn/` checkpoint mtime.
15. Keep `.deps`, logs, datasets, checkpoints, artifacts and videos out of Git.

## Do not retry

Each of these was built, measured, and refuted. Re-running them costs a session
and returns the same answer.

- **Any mechanical interface feature.** The anti-yaw yoke cost insertion 67 points
  to buy extraction 0.13, and was re-tested on 2026-08-16 against a policy whose
  attitude failure sits squarely on the axis it opposes — it moved that axis by
  0.0015 rad. The modelled latch jams the module and collapses travel from 458 mm
  to 25 mm. The compliance is the pads camming open under load; no passive
  geometry reaches it.
- **Reproducing the insert hand-off as a reset.** Per-joint noise box 0.00%,
  measured arm poses 26.32%, arm-and-module poses paired 47.17% — against ~80% in
  the real chain. Train in the chain instead.
- **The scripted realign in the install chain.** `ALIGN_STEPS` defaults to 0. It
  gives +2.4 points on the state chain and -7.5 on the camera arm, because it
  targets the orientation the episode started at and the vision profile displaces
  the module. Module-relative targeting removes the harm and the benefit together.
- **Raising the gripper drive to its rated 24.96 N-m.** Capacity *fell*, because a
  harder squeeze thrusts the payload.
- **Force sensing for pose robustness.** Refuted; worse beyond the trained range.
  Force has to be actionable, not merely observable — that is the action-space
  work, roadmap item 7.
- **Widening the extract reset noise past 0.020 rad.** Above that the scripted
  capture closes on nothing and episodes die on control step 1. Check the step-1
  death rate before touching it.
- **The eight-phase swap task.** Deleted; `tests/test_configuration_contract.py`
  fails if it returns.

## Known broken

- **The `full` round-trip workflow goes non-finite by control step 10** and has
  never been certified. Unrelated to removal, which is healthy. Diagnose before
  quoting anything from `--workflow full`.
- The contact task's finger commands are inverted, and the capture-in-slot task
  fails its own smoke contract. Both pre-existing, both recorded in
  `docs/status.md`, neither blocking. Do not "fix" them without re-certifying what
  they move.

## Where to read

| Working on | Read |
| --- | --- |
| Any result or limitation | `docs/status.md` |
| What to do next | this file, then `docs/roadmap.md` |
| Explaining the project | `docs/claim_vs_evidence.md`, `docs/portfolio.html` |
| The design deliverable | `docs/service_interface_spec.md` |
| Code and data flow | `docs/architecture.md` |
| Physics gaps | `docs/sim2real_matrix.md` |
| The three skills and their criteria | `tasks/blade_swap/grapple_pin_env_cfg.py` |
| Grip metrics, the retain state, derived limits | `mdp/grapple.py` |
| Camera, pose head, blind arm | `mdp/perception.py`, `vision_grapple_env_cfg.py` |
| The chain and how it is judged | `scripts/run_workflow_demo.py` |
| What a chain hands each skill | `--handoff_trace`, `scripts/analyse_handoff.py` |
| Per-reward-term training diagnosis | the run's `summaries/` tfevents |
| Is a pose reachable | `scripts/calibrate_grasp_pose.py`, Capture task, 3,000 steps |
| Gripper geometry, ever | `evidence/gripper_collision_envelope.json` |
| Pin and yoke dimensions | `src/zero_g_blade_swap/grapple_geometry.py` |

## Evaluation contract

`ManagerBasedRLEnv.step` resets terminated environments before returning, so
reading pose after `step` measures the *next* episode. `TerminalMetricsMixin`
intercepts `_reset_idx` and snapshots each finished episode while the scene still
holds its terminal state.

Certification is one `play.py` run per stage and seed writing
`--episode_metrics`, then `scripts/aggregate_evaluation.py` pooling those rows
into a gated report under `evidence/`. Reports align by column name, so a task
may record extra columns without invalidating earlier runs.

**Promotion gate:** at least the stated success rate pooled *and* in every stage,
at least 80% in every randomized-parameter bucket, zero instability terminations,
zero non-finite terminal metrics.

## Standing intent

This is heading for a research paper and an industrial demonstration. Capture
every learning in `docs/status.md` as it happens, keep negative and retracted
results in the record, and make sure every number in any document names the file
in `evidence/` it came from. Commit directly to main, tryaksh as author, no
Co-Authored-By trailers.
