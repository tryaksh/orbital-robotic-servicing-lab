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

## READ FIRST: the interface redesign is on branch `keyed-interface`, not on main

Main carries the **tapered** pin that every report in `evidence/` describes, and it
is verified — extraction runs 98.44% on seed 1070 stage 2 against the 99.02%
certified pooled. Every number below is current for main.

A keyed-flats redesign was built and measured on 2026-08-17 and **reverted from
main** because it took extraction to 0.00% and the axial half was unfinished. It
lives on `keyed-interface` with three corrections already in it. **It solves the
rotational failure that has blocked the relocation for three sessions**, and it is
the most promising open lead in the project:

| | Tapered (main) | Keyed (branch) |
| --- | ---: | ---: |
| Seated grip offset | 0.0194 m | **0.0007 m** |
| Seated grip attitude | 0.0637 rad | **0.0013 rad** |
| Rotation in extraction failures | 0.30+ rad | **0.10 rad** |
| Lateral load held, no slip | — | **120 N at 0.34 m, 40 N·m** |
| Axial pull gate | **69.0 N, passes** | 19–52 N, fails 66.36 N |
| Extraction | **99.02%** | 0.00% |

**What it needs, and it is named by the research rather than guessed.** Space
grippers use finger V-grooves and flight grapple fixtures use cone-shaped aligning
bodies or three lead-in ramps *to guide a part into place*. The taper was doing
double duty — funnel and clamp. Keyed flats are a better clamp and no funnel at
all, and a 30 mm key cannot catch a placement that a 0.020 rad reset draw puts
36 mm out. **The next attempt is a lead-in on the keyed pocket, not more tuning of
it.** Evidence kept on main: `grapple_pin_yaw_gate_keyed.json`,
`grapple_pin_axial_pull_gate_keyed.json`.

The measurements below are the tapered interface's:

| | Tapered pin | Keyed pin |
| --- | ---: | ---: |
| Seated grip offset | 0.0194 m | **0.0007 m** |
| Seated grip attitude | 0.0637 rad | **0.0013 rad** |
| Lateral load held, no slip | — | **120 N at 0.34 m arm, 40 N·m** |
| Axial pull gate | 69.0 N | **19–52 N — FAILS the 66.36 N requirement** |
| Extraction, unchanged policy | 99.02% | **0.00%** |

## Where things stand

Deterministic evaluation, three held-out seeds, Wilson interval, terminal state
captured before auto-reset. Promoted: **capture v5, extract v13unsat, insert
v10twoslot**.

| | Result | Gate 95% | Evidence |
| --- | ---: | --- | --- |
| Capture, alone | **88.78%** | **fail** | `grapple_grasp_v5_certification.json`, re-run 2026-08-17 under the 10 mm tolerance. 100% / 87.12% / 79.22% by reset distance; the gate needs 95% in each. **The old 96.10% is retracted.** Failures are refusals, not timeouts: 1,008 `capture_failed` to 3 `time_out` |
| Extract | 99.02% | pass | `grapple_extract_v14reset_certification.json` |
| Insert, alone, one bay | **98.27%** | pass | `grapple_insert_v6clock30_certification.json`, under the 30 s budget. The 95.57% figure describes the 20 s task |
| **Insert, both bays** | **98.34%** worse bay | **pass** | `grapple_insert_two_slot_certification.json`. Bay 1 98.87%, bay 2 98.34%, pooled 98.60% over 3,004 episodes. **This is the promoted insert** |
| **Removal chain** | **98.78%** | **pass** | `workflow_remove_retain_certification.json` |
| **Install chain** | **96.35%** | **pass** | `workflow_install_clock30retain_certification.json`, Wilson [94.49%, 97.60%], 555/576 |
| **Relocation chain** | **does not complete** | **fail** | Every episode times out in the transit. Diagnosed, not guessed — see item 4 below and `docs/status.md` |
| Install, camera / oracle / blind, one bay | 80.38% / 80.38% / 43.58% | — | `vision_workflow_{camera,oracle,blind}_certification.json` |
| **Install, oracle / camera / blind, two bays** | **88.72% / 65.10% / 34.03%** | **fail** | `vision_workflow_*_twoslot_certification.json`, 576 workflows each. Camera is 23.6 points behind oracle against a 10-point gate — but **on one seed of three**: 86.46% / **25.00%** / 83.85% while oracle is flat at 90.62 / 89.58 / 85.94. A collapse on one randomization draw, not a degradation |
| Insert **inside the chain** | **90.45%** | — | measured, not derived: `workflow_install_promoted_certification.json` predicate-fired column |
| Insert on the **reproduced** hand-off | **93.06%** | — | `insert_chain_handoff_gate.json` |
| Interface axial hold | 69 N vs 66.4 N required | pass | `grapple_pin_axial_pull_gate.json` |
| Module pose from 64x64 RGB, one bay | 1.75 mm mean | — | `module_pose_head.json` |
| **Module pose and bay occupancy, two bays** | **2.81 mm mean, occupancy 100%** | — | `module_pose_head_two_slot.json`. Occupancy exact-match 100% against a 66.6% majority-class baseline. Read it as the bays being 220 mm apart on a camera that resolves 4 mm as 1.31 px — the task is easy and the construction is sound |
| Onboard compute | 0.73 ms CPU, 2.2% of period | — | `inference_budget.json` |

**Capture's 88.78% does not retract any chain number, and the distinction
matters.** The skill task ends an episode when its `capture_failed` predicate
fires; the chain has no such term, hands over on a 10 mm grip held 0.30 s, and
otherwise lets the capture keep closing for its whole 10 s budget — which it does,
overrunning once in 192 chained installations. Adding such a termination to the
chained-insert task was separately measured at 95.31% → 69.27%. So *capture
reliably produces the grip the chain needs*, and *fails a fixed-episode predicate
at the two widest resets*. Never quote either as the other.

Promoted checkpoints, under `logs/rl_games/zero_g_blade_insertion_contact/`:

```
grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth
grapple_extract_l0_seed70_v13unsat/nn/last_zero_g_blade_insertion_contact_ep_5700_rew__148.17932_.pth
grapple_insert_l0_seed70_v10twoslot/nn/last_zero_g_blade_insertion_contact_ep_4400_rew__29.616938_.pth
```

The single-bay insert v6 at `grapple_insert_l0_seed70_v6/nn/..._ep_3200_...` is
superseded but kept: it is what the install-chain and vision certifications were
run with, so those reports describe it and not v10twoslot.

Pose heads, under `checkpoints/`: `module_pose_head.pth` (one bay, pose only) and
`module_pose_head_two_slot.pth` (two bays, pose and occupancy).

## The plan

Work these in order. Each has a gate. **Do not start the next before the gate
passes.** Commit after each with the numbers in the message.

### 1. Insert, trained inside the chain — DONE as an experiment, and it was the wrong lever

**Built, gated, and measured. The task works and the hypothesis was wrong.**

`Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0` runs the frozen capture inside the
environment and hands over on the chain's own predicate. It is the first
construction here that **reproduces** the hand-off rather than approximating it —
grip error, grip attitude, finger angle, drive torque, module pose and all six
arm joints match the chain's trace at p50 and p95, and insert v6 unchanged scores
93.06% on it against 90.45% in the real chain (`insert_chain_handoff_gate.json`).

What that gate also showed is that **the premise of this item was wrong by ten
points**: the hand-off costs the skill about 2.5 points, not the ~15 the "~80%"
figure implied. So there was little to gain, and 300 epochs of fine-tuning on it
moved the chain from 89.41% to **88.37%** — nothing.

**The two things that did move it were not training at all**:

| Change | Install chain, seed 4070 |
| --- | ---: |
| Baseline | 85.94% |
| Insert phase clock 20 s → 30 s | +2.6 pts on the insert phase |
| Retain instead of hold through the settling window | **90.10%** |

Both are in. Do not spend another session training insert on its distribution.

### 2. Second slot geometry — DONE, gate passed

Slot 2 at **y = -0.22 m**, built as a displacement of the certified slot part for
part, in `ZeroGTwoSlotGrapplePinSceneCfg`. Converged IK reaches its staging pose
to **0.0060 mm** at every stage (`artifacts/relocation/slot_two_pose.json`).

### 3. Insert retrained for both bays — DONE, gate passed

The smoke blocker is closed: the contact reward contract is **scoped** to the
family it was written for, not weakened. It is skipped where
`events.hold_gripper_closed.func is hold_two_stage_grip`, because on those tasks
the scripted capture is still closing through the action term's 1.0 s settling
window, so a zero action is not a stationary module and the progress term is paid
for motion the script caused. Where it is skipped it prints the number it would
have judged. Capture and the chained-insert task still run it.

Then trained: 1,200 epochs at 512 environments from insert v6, both bays 50/50.
**98.87% bay 1, 98.34% bay 2, pooled 98.60%** over 3,004 episodes and three
held-out seeds — gated on the worse bay, which is what the gate asked for.
`evidence/grapple_insert_two_slot_certification.json`.

The second bay was not hard: unlocked around epoch 3,250, it went 0 → 83% in 40
epochs and past 95% within 250, while bay 1 never dropped below 97.9%. That is
evidence for building the bay as a part-for-part displacement, not for the policy.

### 4. Lateral transit — MEASURED, gate FAILS, and the cause is now specific

**This is where the night stopped, and it is the one thing to work on next.**
Every episode times out inside the transit. Six instrumented runs; do not repeat
them. `scripts/run_relocation.sh trace` reproduces the whole diagnosis in about
four minutes at `EPISODES=64`, and the `[CHAIN]` progress line now reports the
follower's leg, its distance to that leg's waypoint, each conjunct of the arrival
test, the grip error, and the tool's position in the module's own frame.

What is **settled**:

- **The plan is right.** Environments that arrive land at tool x = 0.2482 against
  a planned 0.2475, module x = 0.5790 against a 0.5779 threshold. The 78.25 mm
  retreat derivation holds.
- **The module was flipping, not slipping.** With the tool exactly on its final
  waypoint the tool-to-module offset had gone from −0.335 m to **+0.305 m** — a
  sign change — while grip error stayed near 24 mm. The module had swung
  end-for-end about the pin, so pushing the tool forward drove the module's tail
  at the rack. The transit was commanding nothing on its three rotation channels.
- **Holding the attitude fixes the grip.** Bounded to a quarter of the rotation
  authority (`TRANSIT_ATTITUDE_AUTHORITY`), grip error through the flight is
  **11 mm** at the median and the module tracks its retreat waypoint exactly.
  Unbounded it also holds the module but starves translation, because a 0.1–0.3 rad
  attitude error against a 0.020 rad scale saturates the command permanently.
- Two more corrections are in and measured: legs finish along **the axis they were
  laid out along** (the 3-D test assumed nothing else moved the tool, and holding
  attitude does), and the cross leg and the arrival test now target **the bay in
  the rack's frame** instead of a displacement from wherever the episode started —
  the tool drifts about 93 mm laterally during capture and extraction, against a
  72.5 mm channel half-width.

**And the cause is now identified, by the converged IK calibrator.** The retreat
depth is **not reachable holding the head-on capture attitude**. Solving for the
tool at the retreat depth in the *first* bay — no lateral component at all — the
solver holds the attitude to 0.0001 rad and gives up **174 mm of position**
(`artifacts/relocation/cross_control_bay1.json`). It is not failing to converge; it
is converging to the nearest pose that keeps the attitude, and that pose is 174 mm
short.

That closes the story, because the transit's tool *does* reach that depth, to
within 4 mm of the same target — with its attitude unconstrained. **So the arm was
only ever reaching the retreat pose by rotating the wrist into a configuration the
pin cannot hold the module in.** The flip and the stalled cross leg are one
failure, not two: give up the attitude and the module flips; command it and the
legs cannot complete.

The lateral displacement is *not* the problem — the second bay's staging pose
converges to 0.0060 mm with the same solver and the same attitude, and the two-slot
insert certifies at 98.34% from it. **The depth is the problem, and the depth is
what `TRANSIT_RETREAT_M` exists to buy.**

Two ways out, both design changes rather than parameters:

1. **Cross before retreating fully** — move laterally while the module is still
   shallow in the first bay's rails, which constrain its attitude, and retreat only
   as far as the flare plane requires at the crossing y. The 78.25 mm figure is
   derived for a module turning *at* the extraction pose; a path that crosses
   earlier may need less. **Cheap, untried, do this first.**
2. **Change the workcell** — bay pitch, base position, or reach. This is the
   "workcell layout, not the interface" hypothesis `docs/status.md` has carried as
   its leading suspect since 2026-08-15, and it now has a direct measurement.

- Gate: module held under 20 mm grip error across the whole transit. The grip half
  is met at 11 mm; the traverse is not.

### 5. Certify the relocation chain — blocked on item 4

`capture -> extract -> lateral transit -> insert(slot 2)`, one continuous episode.
Do not start it before item 4's gate passes; a run today measures the transit
failure, not the chain. `scripts/run_relocation.sh relocate`, which now resolves
the two-bay insert checkpoint itself and refuses to run with the single-bay one.

- Gate: **>= 95%**, three held-out seeds, zero instability, zero non-finite.

### 6. Perception reads the rack — perception half DONE, arms measured on a two-bay install

The pose head has an occupancy branch: one independent logit per bay, because
during a relocation the module is in neither for the whole transit and a softmax
cannot say that. Trained on 60,000 two-bay frames under the same randomized
lighting, albedo, camera noise and unknown module displacement as the single-bay
head. Held out on 12,000: **occupancy 100% per bay and 100% exact-match over the
whole rack, against a 66.6% majority-class baseline**, with module pose at 2.81 mm
mean and 6.47 mm p95. `evidence/module_pose_head_two_slot.json`.

Read the 100% honestly: the bays are 220 mm apart and this camera resolves 4 mm as
1.31 px, so a module in bay 2 is in a visibly different part of the frame. The
number says the construction is sound, not that the perception is clever.

The three arms were run on `GrappleVisionTwoSlot-Install-v0` — installation into
bay 1 on a two-bay rack — rather than on the relocation, because the arms measure
what perception *costs* and that needs a manipulation task that completes. Item 4
is why the relocation does not.

**The gate fails, on one seed of three.** Oracle 88.72%, camera 65.10%, blind
34.03% over 576 workflows each. The blind half of the gate passes with a 31-point
margin. The camera half fails at 23.6 points behind oracle — but the per-seed row is
**86.46% / 25.00% / 83.85%** against an oracle that is flat at 90.62 / 89.58 /
85.94. Two seeds are inside the gate with room and one collapses.

So this is an estimator that fails on a randomization draw, not one that costs 23
points. It is not the occupancy readout (100% exact-match, and the failures are
insertion timeouts rather than wrong-bay attempts) and not the manipulation (oracle
is stable). The pose head's held-out **p95 is 6.47 mm against a 4 mm insertion
lateral tolerance** while its mean is 2.81 mm — an adequate typical accuracy with an
inadequate tail, which is the shape that produces exactly this.

**Next: find what seed 5070 draws, before training anything.**
`scripts/sweep_camera_calibration.py` and the collector's `--camera_offset_mm` /
`--camera_tilt_mrad` exist for this. If it is a lighting draw the fix is collection
coverage; if it is a module displacement the head extrapolates badly on, the fix is
the label range.

*Worth keeping: the same sweep at seed 4070 alone reported camera 84.90% against
oracle 90.63% and would have been written up as a pass. One seed would have
published a gate that three seeds refute.*

- Gate: camera arm within 10 points of oracle, blind arm clearly below both.

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
  measured arm poses 26.32%, arm-and-module poses paired 47.17%. **Closed:**
  `InsertChain-v0` reproduces it at 93.06% by running the real capture. Do not
  build a fifth reset. The pose bank, its generator and `reset_from_handoff_bank`
  were all deleted for this reason; the measurements are in `docs/status.md`.
- **Fine-tuning insert on the chain's own distribution.** 300 epochs at 512
  environments took the install chain 89.41% → 88.37%, and the training curve was
  flat from the first tenth. The hand-off costs the skill 2.5 points, so there
  was nothing there to win. The gap is the objective and the clock, not the
  starting state.
- **A capture-failure termination in the chained-insert task.** Refuted by its
  own gate before training: 69.27% with it against 95.31% without, because the
  chain carries no such term and overruns its capture once in 192 while the
  predicate was killing 52.
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
- **Rate-limiting the relocation transit's final leg.** Slowed to a third of
  command, the way the replayed transit is slowed: module x went −0.003 → −0.158
  and crossings fell 46 → 19, because the tool then lagged its own waypoint. The
  module was never being driven too fast.
- **Commanding the transit's attitude at full authority.** It holds the module
  beautifully and starves translation — a 0.1–0.3 rad grip attitude error against a
  0.020 rad rotation scale saturates the command permanently, and the tool sat at
  the retreat waypoint 1,450 steps later. Bounded to a quarter
  (`TRANSIT_ATTITUDE_AUTHORITY`) it works; unbounded it deadlocks.
- **Reading the relocation transit's arrival in 3-D.** Each planned leg moves along
  one axis and holding the module's attitude moves the tool off the other two: the
  tool sat 1 mm from its waypoint in x and 53 mm away in 3-D, and the follower never
  advanced. Per-axis is not a shortcut, it is the correct test.
- **A relative lateral cross.** `back_y + SECOND_SLOT_CENTER_Y` carries the ~93 mm
  of lateral drift capture and extraction leave into the second bay, whose channel
  half-width is 72.5 mm. Target the bay in the rack's frame; the same applies to the
  arrival test, which passed while the module sat outside the channel.
- **Trusting a single-seed vision sweep.** The two-bay arms at seed 4070 alone read
  camera 84.90% against oracle 90.63% and would have been published as a gate pass.
  Three seeds give camera 65.10% against 88.72%, because one seed collapses to
  25.00%.

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
