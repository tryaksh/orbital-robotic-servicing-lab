# Next session: the prompt to paste

> **HISTORICAL — this is a session prompt, not documentation.** It records how
> the work was directed on the date in its own text, and the numbers in it were
> current only then. Several are now retracted or superseded. It is kept because
> the record of how an autonomous session was steered is part of what this
> repository is showing. **For current numbers read `docs/status.md`,
> `README.md`, and `evidence/RETRACTED.md`.**


Copy everything between the rules. It assumes nothing that is not in the repo.

---

You own `D:\6axis-space-robotics`. Read `CLAUDE.md` first — it is the plan, not
background. Act as the senior robotics simulation engineer who owns this repo.

FULL AUTHORITY for a long unattended session. Start GPU work immediately. Do not
ask permission to launch a run.

**FIRST ACTIONS, in this order.** Check for orphaned Isaac processes
(`Get-Process kit`) and kill them; one survived five hours once and slowed
everything about 40%. Then:

```bash
python scripts/check_criterion_currency.py
```

Re-run whatever it flags STRONG, after confirming with `git log -S` on the
constant the report actually depends on — the last session found all ten flags
were a comment-only commit, a pure refactor, a default-off probe, and transit code
that touches only the two workflows that fly. A long flag list is the honest
output of that script, not a verdict. `evidence/RETRACTED.md` lists every
retraction and its replacement; read it before quoting any figure.

THE GOAL is the relocation demo: capture → extract → transit across → insert into
bay 2, one continuous episode, certified at ≥ 95% on three held-out seeds.

## Where the last session left it

**Done and certified.** The two-bay scene builds and smokes. One insert policy
seats a module in **either** bay — 98.87% in bay 1, **98.34% in bay 2**, pooled
98.60% over 3,004 episodes, gated on the worse bay
(`evidence/grapple_insert_two_slot_certification.json`). It is the promoted insert
now: `grapple_insert_l0_seed70_v10twoslot/nn/..._ep_4400_...`.

**Done and it failed.** Capture v5 re-certified under its own current 10 mm
tolerance is **88.78%** and fails its 95% gate — 100% / 87.12% / 79.22% by reset
distance, with 1,008 of 1,011 failures being refusals rather than timeouts. The
96.10% figure is retracted. This does **not** retract any chain number, and
`CLAUDE.md` explains exactly why in the paragraph under the results table: the
skill task ends an episode when `capture_failed` fires and the chain has no such
term. Do not conflate them, and do not "fix" capture on the strength of the skill
number without deciding which question you are answering.

**Done and measured.** The pose head has an occupancy branch. On 12,000 held-out
two-bay frames it reads which bay holds the module at 100%, exact-match over the
whole rack, against a 66.6% majority-class baseline, with pose at 2.81 mm mean
(`evidence/module_pose_head_two_slot.json`). The 100% is honest and easy: the bays
are 220 mm apart on a camera that resolves 4 mm as 1.31 px.

**Done and the gate failed.** The three vision arms on a two-bay installation:
oracle 88.72%, camera 65.10%, blind 34.03%, 576 workflows each
(`evidence/vision_workflow_*_twoslot_certification.json`). The blind half of the
gate passes with a 31-point margin. The camera half fails, and item 1 under
*outstanding* below is the reason it is interesting rather than just bad.

**Not done, and it is the one thing that matters.** The relocation chain does not
complete. Every episode times out inside the lateral transit.

## Item 4 is the whole job. Start here.

`scripts/run_relocation.sh trace` at `EPISODES=64` reproduces the entire diagnosis
in about four minutes. The `[CHAIN]` progress line reports the follower's leg, its
distance to that leg's waypoint, every conjunct of the arrival test, the grip
error, and the tool's position in the module's own frame. Use it; six runs went
into building it.

**Settled, and do not re-derive:**

- The transit plan is correct. Arriving environments land at tool x = 0.2482
  against a planned 0.2475 and module x = 0.5790 against a 0.5779 threshold.
- **The module was flipping, not slipping.** With the tool on its final waypoint
  the tool-to-module offset changed sign, −0.335 m → **+0.305 m**, while grip error
  read a mild 24 mm. It had swung end-for-end about the pin. The transit was
  commanding nothing on its three rotation channels.
- Holding the grip attitude, bounded to a quarter of the rotation authority
  (`TRANSIT_ATTITUDE_AUTHORITY`), takes grip error through the flight to **11 mm**
  and the module tracks its retreat waypoint exactly. Unbounded, it starves
  translation — a 0.1–0.3 rad error against a 0.020 rad scale saturates
  permanently — and the tool sat at the retreat waypoint 1,450 steps later.
- Legs now finish along **the axis they were laid out along**; the 3-D proximity
  test assumed nothing else moved the tool, and holding attitude does.
- The cross leg and the arrival test target **the bay in the rack's frame**, not a
  displacement from where the episode started. The tool drifts about 93 mm
  laterally during capture and extraction, against a 72.5 mm channel half-width.
- **Refuted:** rate-limiting the final leg to a third of command, the way the
  replayed transit is slowed. Module x went −0.003 → −0.158 and crossings fell
  46 → 19, because the tool then lagged its own waypoint.

**The cause is identified. It is a workspace limit, not a follower parameter.**
The retreat leg now completes for all 64 environments and the **cross does not** —
the distance to the lateral waypoint *grows*, 0.303 → 0.397 m, while the tool holds
its retreated depth. The converged IK calibrator says why
(`artifacts/relocation/cross_control_bay1.json`): solving for the tool at the
retreat depth in the **first** bay, with no lateral component at all, it holds the
head-on capture attitude to 0.0001 rad and **gives up 174 mm of position**. It is
not failing to converge; it is converging to the nearest pose that keeps the
attitude, and that pose is 174 mm short of the retreat depth.

The transit's tool *does* reach that depth, within 4 mm of the same target — with
its attitude unconstrained, because until last night nothing commanded it. **So the
arm was only ever reaching the retreat pose by rotating the wrist into a
configuration the pin cannot hold the module in.** The flip and the stalled cross
are one failure: give up the attitude and the module flips; command it and the legs
cannot complete.

The lateral displacement is **not** the problem. The second bay's staging pose
converges to 0.0060 mm with the same solver and the same attitude, and the two-slot
insert certifies at 98.34% from it. **The depth is the problem, and the depth is
what `TRANSIT_RETREAT_M` buys.**

(An earlier note in `docs/status.md` called this probe mis-targeted, on the belief
that the calibrator servos to a "handle centre" distinct from the grip point. That
was wrong and is corrected: for this blade `GrapplePinBladeCfg.handle_offset` *is*
`GRAPPLE_PIN_GRIP_OFFSET`, and its docstring says so. Do not re-derive it.)

Two ways out, both design changes:

- **Cross before retreating fully.** Move laterally while the module is still
  shallow in the first bay's rails, which constrain its attitude, and retreat only
  as far as the flare plane actually requires at the crossing y. The 78.25 mm
  retreat is derived for a module turning *at* the extraction pose; a path that
  crosses earlier may need less of it. **Cheap, untried, do this first.**
- **Change the workcell.** Bay pitch, base position, or reach. This is the
  "workcell layout, not the interface" hypothesis this project has carried as its
  leading suspect since 2026-08-15, and it now has a direct measurement instead of
  an inference from failure modes. If you go this way, say so as a finding.

Then item 5: `scripts/run_relocation.sh relocate`. It resolves the two-bay insert
checkpoint itself and refuses to run with the single-bay one. Gate ≥ 95%, three
held-out seeds, zero instability, zero non-finite. **Do not start it before item
4's gate passes** — a run today measures the transit failure, not the chain.

## Also outstanding, in priority order

1. **Why the camera arm collapses on seed 5070.** The two-bay arms are run and the
   gate fails, but not the way a cost curve fails: camera scores 86.46% / **25.00%**
   / 83.85% across the three seeds while oracle is flat at 90.62 / 89.58 / 85.94.
   Two seeds are inside the 10-point gate with room; one collapses. It is not the
   occupancy readout (100% exact-match, and the failures are insertion timeouts, not
   wrong-bay attempts) and not the manipulation (oracle is stable). The head's
   held-out p95 is **6.47 mm against a 4 mm insertion lateral tolerance** while its
   mean is 2.81 mm — adequate typical accuracy, inadequate tail.
   **Find what that seed draws before training anything.** The per-run
   randomization covers orbital sun intensity, angle, pitch, yaw and colour
   temperature, rack albedo, metallic and roughness, camera radiation noise, and the
   module's own displacement. `scripts/sweep_camera_calibration.py` and the
   collector's `--camera_offset_mm` / `--camera_tilt_mrad` exist for this. Lighting
   draw → fix collection coverage. Module displacement the head extrapolates badly
   on → fix the label range.
   The sweep costs about **6 minutes per arm-seed** at `ENVS=64`, so all three arms
   across three seeds is under an hour:
   `TASK=Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Install-v0 WORKFLOW=install STAGE=2
   TAG=_twoslot HEAD=checkpoints/module_pose_head_two_slot.pth ENVS=64
   ARMS="oracle camera blind" bash scripts/certify_vision_workflow.sh`.
2. **A labelling defect that touches every grapple-pin report.** `play.py` decides
   whether the lead-in flares are collidable with
   `bool(...collision_props.collision_enabled)`. That field is a tri-state and
   IsaacLab documents `None` as "leave as authored"; the grapple-pin scene leaves
   it `None`, so an enabled collider reports as absent and every report is stamped
   `out_of_distribution: true` with `gate.applies: false`. `train.py` reads the
   same field correctly, treating only an explicit `False` as disabled. No rate is
   wrong; the label on all of them is. Fix it by reading the runtime state off the
   stage the way `train.py` does, then re-run to re-label. Recorded in
   `evidence/RETRACTED.md`.
3. **The occupancy dataset is 67/33 and has no "neither bay" frames.** Two of the
   three collection stages sit in bay 1, and the third — nominally "at the mouth" —
   is 358 mm inside its channel and so counts as occupied. Worth rebalancing if the
   occupancy claim is going in a paper.

## The rules that do not bend

All of `CLAUDE.md`'s non-negotiables, and these two earned their place last night:

- **A diagnostic must be written before anything that formats a report.** The first
  relocation trace cost eleven minutes of simulation and produced no file, because
  a dict literal in the report was missing a workflow key. Fixed, and the lesson
  generalises.
- **A stalled phase and a slow one produce identical phase counts.** If a run is
  not progressing, add the state that distinguishes them before adding a fix. Two
  of last night's six transit runs existed only because the instrumentation did
  not yet say which was happening.

Commit directly to main, tryaksh as author, no Co-Authored-By trailers. Keep
`docs/status.md` current as you go, not at the end.

---
