# Agent handover

You own this repository. Act as the senior robotics simulation engineer who does.

**This project is finished as a demonstration.** It is not a to-do list any more.
What follows is the state it is in, the rules that produced it, and the things
that were tried and refuted so a later session does not spend a night rediscovering
them. If you are picking it up to extend it, read *Where it stops* before choosing
anything.

## What it is

A **design-for-serviceability** study: what does a modular compute unit have to
present, physically, for a 6-axis manipulator to swap it in microgravity, and what
loads does that impose. The design output is
[`docs/service_interface_spec.md`](docs/service_interface_spec.md).

Isaac Sim 5.1, Isaac Lab v2.3.2, RL-Games PPO, zero gravity, 30 Hz policy /
120 Hz physics, one RTX 5070 Ti Laptop GPU.

## The finding

**Attitude, not position, is the binding constraint** on every operation that
moves a payload through free space, and it binds in two independent places:

- **On the grip.** A parallel-jaw grip on a passive feature cannot resist a moment
  about its closing axis — the pads' normals lie along that axis, and a normal
  force cannot oppose a moment about its own direction. Only friction can, and it
  loses. The clearest single datum is a module carried between bays that swung
  *end-for-end* about its grip, the tool-to-module offset changing sign from
  −0.335 m to +0.305 m, while the grip error read a healthy 24 mm.
- **On the arm.** Near the folded configuration servicing requires, attitude and
  reach trade at about **7.5 metres per radian**. Drive orientation at full
  authority and the tool stops 88.7 mm short of the extraction pose; allow
  0.0114 rad and it reaches to 3.6 mm. The arm can go where the task needs it; it
  cannot go there pointing the right way and stay there.

## Certified state

Deterministic evaluation, three held-out seeds, Wilson intervals, terminal state
captured before Isaac Lab's auto-reset. Every number names its evidence file.

| | Result | Gate 95% | Evidence |
| --- | ---: | --- | --- |
| **Removal chain** | **98.78%** | **pass** | `workflow_remove_retain_certification.json` |
| **Installation chain** | **96.35%** | **pass** | `workflow_install_clock30retain_certification.json` |
| Extract | 99.02% | pass | `grapple_extract_v14reset_certification.json` |
| Insert, one bay | 98.27% | pass | `grapple_insert_v6clock30_certification.json` |
| **Insert, both bays**, gated on the worse | **98.34%** | **pass** | `grapple_insert_two_slot_certification.json` |
| Capture, alone | **88.78%** | **fail** | `grapple_grasp_v5_certification.json` |
| **Relocation chain** | **does not complete** | **fail** | `relocation_reach_boundary.json` |
| Interface axial hold | 69.0 N vs 66.4 N | pass | `grapple_pin_axial_pull_gate.json` |
| Pose + bay occupancy from 64×64 RGB | 2.81 mm mean, 100% occupancy | — | `module_pose_head_two_slot.json` |
| **Vision, two bays: oracle / camera / blind** | **88.72% / 84.90% / 34.03%** | **pass** | `vision_workflow_*_twoslot_certification.json`. Camera is 3.82 points behind oracle against a 10-point allowance and 50.87 above blind |
| Onboard compute | 0.73 ms CPU, 2.2% of period | — | `inference_budget.json` |

Both chains are 576 workflows, zero instability, zero non-finite, module held by
real pad-against-pin contact throughout with no fixed joint anywhere.

**Capture's 88.78% does not retract any chain number, and the distinction
matters.** The skill task ends an episode when its `capture_failed` predicate
fires; the chain has no such term, hands over on a 10 mm grip held 0.30 s, and
otherwise lets the capture finish — which it does, overrunning once in 192 chained
installations. Adding such a termination to the chained-insert task was separately
measured at 95.31% → 69.27%. So *capture reliably produces the grip the chain
needs* and *fails a fixed-episode predicate at the two widest resets*. Never quote
either as the other.

Promoted checkpoints, under `logs/rl_games/zero_g_blade_insertion_contact/`:

```
grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth
grapple_extract_l0_seed70_v13unsat/nn/last_zero_g_blade_insertion_contact_ep_5700_rew__148.17932_.pth
grapple_insert_l0_seed70_v10twoslot/nn/last_zero_g_blade_insertion_contact_ep_4400_rew__29.616938_.pth
```

The single-bay insert v6 at `grapple_insert_l0_seed70_v6/nn/..._ep_3200_...` is
superseded but kept: the install-chain certification was run with it, so that
report describes v6 and not v10twoslot. The two-bay vision arms used v10twoslot.

Pose heads under `checkpoints/`: `module_pose_head.pth` (one bay) and
`module_pose_head_two_slot.pth` (two bays, pose and occupancy).

## Where it stops

### The relocation, and it is a workcell problem

`capture → extract → transit across → insert(slot 2)` was the deliverable and it
does not complete. Every episode times out inside the transit, and the cause is
measured rather than guessed.

Driving position and the head-on capture attitude at full authority, the tool
parks at local x = −0.0258. Extraction must finish 88.7 mm past that; the
transit's retreat needs 167 mm past it. Two controls settle what that means:

- **attitude free** — every depth converges to 0.00 mm, including 64 mm beyond the
  retreat. The arm can go there;
- **lift** — the rack's lead-in flares are 50 mm plates and the module is 35 mm
  thick, so a 45 mm lift clears them and lets the module cross at the *extraction*
  depth rather than behind the flare plane. The parking point does not move, at
  0, 50, 100 or 200 mm of lift.

So no leg order and no lift fixes it. Fixing it means moving the robot base, which
needs the head-on spawn pose re-solved first — probed at −0.65 the arm starts
200 mm short of its own capture pose and shoves the module 137 mm — and then
invalidates capture, extraction, insertion, both chains and all three vision arms.
That is not a trade worth making against a certified removal-and-installation
demonstration, and it is why the relocation is reported blocked rather than fixed.

### Capture's own gate

88.78%, and it fails. See above for why that is a statement about the skill task's
termination and not about the chain.

### The camera arm is not deterministic, and nothing here explains why

The state pipeline is: the oracle and blind arms both reproduce their per-seed
rates *exactly* across sessions and code states. The camera arm does not — two
runs with identical seed, task and checkpoints diverge at the first episode, and
six runs span 80.73–86.46% with a 2.13-point standard deviation.

Two controls narrow the cause to one thing. The blind arm runs in the **same**
vision scene, with the camera built and rendering, and simply does not read the
frame — and it reproduces exactly. So it is not the camera's presence, the
Replicator randomizers, or the scene's physics. It is reading the rendered frame
and pushing it through the pose head.

**What is not explained is the size of the worst excursion.** The retracted
2026-08-17 run scored 25.00% where six repeats average 83.85% — **27.6 standard
deviations** out. That is not the tail of the observed spread; it is very likely a
second mechanism, and it is unidentified. Quote the camera number with that
attached, and replicate any camera measurement before believing it.

## Do not retry

Each was built, measured, and refuted. Re-running them costs a session and returns
the same answer.

- **Any additive anti-rotation feature on the tapered pin.** The anti-yaw yoke cost
  insertion 67 points to buy extraction 0.13; its code was deleted on 2026-08-18
  and a contract test keeps it out. The modelled latch jams the module in the rails
  and collapses travel from 458 mm to 25 mm. The compliance is the pads camming
  open under load; no passive geometry reaches it.
- **The keyed interface.** Flat keyed faces between two axial stops fix rotation
  decisively — seated grip 0.0194 → 0.0007 m, attitude 0.0637 → 0.0013 rad, 120 N
  held at a 0.34 m arm — and cannot be built on this gripper. The nose flange sits
  **45.0 mm inside the palm at every closure**, and no forward axial stop fits at
  any key height: 7.9 mm of room against the 43.5 mm a splay-proof stop needs.
  `evidence/grapple_pin_keyed_interference.json`. The branch `keyed-interface` is
  kept for its rotation measurements; nothing on it will be merged.
  **Corollary worth keeping:** the 2F-85's throat is cone-shaped and the taper's
  profile matches it, so the taper doing double duty as funnel and clamp is not a
  design smell — it is the only shape this hand admits.
- **Reproducing the insert hand-off as a reset.** Four attempts: 0.00%, 26.32%,
  47.17%. `InsertChain-v0` reproduces it at 93.06% by running the real capture.
- **Fine-tuning insert on the chain's own distribution.** 300 epochs took the
  install chain 89.41% → 88.37%. The hand-off costs the skill 2.5 points, so there
  was nothing there to win.
- **A capture-failure termination in the chained-insert task.** 69.27% with it
  against 95.31% without.
- **The scripted realign in the install chain.** `ALIGN_STEPS` defaults to 0:
  +2.4 points on the state chain, −7.5 on the camera arm.
- **Raising the gripper drive to its rated 24.96 N·m.** Capacity *fell*.
- **Force sensing for pose robustness.** Worse beyond the trained range. Force has
  to be actionable, not merely observable.
- **Widening the extract reset noise past 0.020 rad.** Above that the scripted
  capture closes on nothing and episodes die on control step 1.
- **Rate-limiting the relocation transit's final leg**, **commanding transit
  attitude at full authority** (starves translation), and **a relative lateral
  cross** (carries 93 mm of drift into a 72.5 mm channel).
- **Reading the transit's arrival in 3-D.** Per-axis is not a shortcut, it is the
  correct test: holding attitude moves the tool off the other two axes.
- **The eight-phase swap task** and **`--workflow full`.** Both deleted, both
  defended by contract tests.

## Non-negotiable rules

These are the ones that were paid for.

1. **Any phase that waits must either command or retain.** An arm that stops
   commanding while gripping a module in zero gravity is not holding still, it is
   being pushed. Measured twice: chained removal 0/570 → 569/576 when `retain_latch`
   was added, and a 2 s idle pause in the install chain costs 84.90% → 21.35%.
   **Capture gently, hold hard to move, retain once free.**
2. **Read constants, never restate them.** Six failures came from one number living
   in two places. It failed in the other direction too: `PALM_FACE_FROM_FLANGE_M`
   existed, said "nothing can sit closer to the flange than this", and the keyed
   redesign put a section 13 mm inside it because nobody read it and no test
   defended it. Two do now.
3. **A number chosen for one purpose will silently decide what a policy can
   learn.** `grip_retention_penalty` saturated at 0.325 rad and the policy parked
   at 0.3538; extraction went 0.00% → 68.62% by unclamping it. Read the per-term
   tfevents under the run's `summaries/`.
4. **A skill must be trained across the states its predecessor actually produces.**
5. **Before believing a reconstructed distribution, run the unchanged successor
   policy on it.** Five minutes. It would have saved two training runs.
6. **Before believing a probe, prove it moves what it measures.** Paid three times
   in one session: two versions of the pin-clearance check condemned the
   *certified* taper before the third was right, and `--robot_base_x` had never
   moved anything — two sweeps 300 mm apart returned byte-identical joint
   solutions.
7. **Before believing a solver, converge it — and know which side of a trade it
   converged to.** A pose called unreachable on a 400-step residual converges to
   0.0060 mm at 3,000. And where position and orientation trade steeply, "converged"
   depends on the authority each is driven with: the same pose reads 3.6 mm at
   0.0114 rad or 88.7 mm at 0.0002 rad. Use the **Capture** task for pose
   calibration.
8. **Never weaken a success threshold to make a gate pass.** Correcting a reset
   that produces *unwinnable* states is different, allowed, and must be stated with
   its measurement.
9. **A skill certification is not evidence about a chain.** Certify both.
10. **Never quote a rate without `scripts/check_evidence_currency.py`** and
    `check_criterion_currency.py`.
11. **Three seeds are three samples of the configuration, not of the run.** A
    single-seed sweep once reported a gate *pass* three seeds overturned; three
    seeds then reported a *failure* a re-run overturned, because one run in nine
    scored 25.00% where every repeat scores about 82%. Where a run can differ by
    56 points, replicate the run.
12. Zero gravity, 30 Hz policy / 120 Hz physics. Never resume across an action or
    observation dimension change; resuming across physics or reward changes is
    allowed and is how most policies here were made — say so when you do.
13. One Isaac process at a time. **Check for orphans before every launch**
    (`Get-Process kit`); one survived five hours and slowed everything ~40%.
14. 512 environments costs ~0.9 GB of 12 GB. `train.py`'s redirected stdout is
    block-buffered and lags minutes — judge progress from `summaries/` tfevents and
    the `nn/` checkpoint mtime.
15. Keep `.deps`, logs, datasets, checkpoints, artifacts and videos out of Git.

## Known broken, and left that way deliberately

- The contact task's finger commands are inverted: measured pad separation falls
  monotonically with the command, so `finger_joint` 0 is fully open, and that
  task's "pregrasp 0.80 / closed 0.68" pair opens the fingers by 14 mm. The
  grapple-pin tasks use the measured convention. Correcting the contact task
  changes the physics three promoted certifications were produced under, so it is
  recorded rather than fixed.
- The capture-in-slot task fails its own smoke contract, on a `contact_grasp` flag
  inconsistent with its parent's disabled handle collider. Pre-existing, not
  blocking, recorded in `docs/status.md`.

## Evaluation contract

`ManagerBasedRLEnv.step` resets terminated environments before returning, so
reading pose after `step` measures the *next* episode. `TerminalMetricsMixin`
intercepts `_reset_idx` and snapshots each finished episode while the scene still
holds its terminal state.

Certification is one `play.py` run per stage and seed writing `--episode_metrics`,
then `scripts/aggregate_evaluation.py` pooling those rows into a gated report under
`evidence/`. Reports align by column name, so a task may record extra columns
without invalidating earlier runs.

**Promotion gate:** at least the stated success rate pooled *and* in every stage,
at least 80% in every randomized-parameter bucket, zero instability terminations,
zero non-finite terminal metrics.

## Where to read

| Working on | Read |
| --- | --- |
| Any result or limitation | `docs/status.md` |
| Explaining the project | `README.md`, `docs/claim_vs_evidence.md`, `docs/portfolio.html` |
| What has been withdrawn and why | `evidence/RETRACTED.md` |
| The design deliverable | `docs/service_interface_spec.md` |
| Code and data flow | `docs/architecture.md` |
| Physics gaps | `docs/sim2real_matrix.md` |
| The three skills and their criteria | `tasks/blade_swap/grapple_pin_env_cfg.py` |
| Grip metrics, the retain state, derived limits | `mdp/grapple.py` |
| Camera, pose head, blind arm | `mdp/perception.py`, `vision_grapple_env_cfg.py` |
| The chain and how it is judged | `scripts/run_workflow_demo.py` |
| Per-reward-term training diagnosis | the run's `summaries/` tfevents |
| Is a pose reachable, and at what attitude | `scripts/calibrate_grasp_pose.py` — `--sweep_offset_x`, `--attitude_authority`, `--free_orientation` |
| Does a pin feature fit inside the gripper | `scripts/check_pin_gripper_clearance.py`, `scripts/measure_pin_design_window.py` |
| Gripper geometry, ever | `evidence/gripper_collision_envelope.json` |
| Pin dimensions | `src/zero_g_blade_swap/grapple_geometry.py` |

## Standing intent

This is a portfolio and paper artefact. Keep every learning in `docs/status.md` as
it happens, keep negative and retracted results in the record, and make sure every
number in any document names the file in `evidence/` it came from. Commit directly
to main, tryaksh as author, no Co-Authored-By trailers.
