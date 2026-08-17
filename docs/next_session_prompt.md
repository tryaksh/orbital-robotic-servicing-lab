# Next session: the prompt to paste

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

**The open question, and the next experiment.** The retreat leg completes for all
64 environments. **The cross does not** — the distance to the lateral waypoint
*grows*, 0.303 → 0.397 m, while the tool correctly holds its retreated depth. The
retreat leaves the arm folded back near its own base, and the cross asks for a
220 mm lateral sweep from there. The second bay's *staging pose* is proven
reachable to 0.0060 mm; **the path between the bays at the retreated depth has
never been tested.**

Test that with the converged IK calibrator, not with another follower parameter —
rule 7, `scripts/calibrate_grasp_pose.py` on the Capture task at 3,000 steps.
**And fix the probe first.** The last session's attempt is in
`artifacts/relocation/cross_control_bay1.json` and is inconclusive: its control,
the retreat depth in bay 1, reports not converged with a 0.167 m residual on a
pose the transit reaches every episode to 0.8 mm. The calibrator servos the tool
to the **handle centre** plus the offset, and on this interface the tool sits about
335 mm behind the blade centre at the pin's grip point, so the offset asks for a
different pose than the transit flies. Correct the offset to the grip point, keep
the bay-1 retreat pose as the control, and require the control to converge before
believing anything about the cross.

The answer splits the work cleanly:

- **Controller problem** → change the leg order. Cross while the module is still
  shallow in the first bay's rails, which constrain its attitude, instead of after
  it is fully free. This is the cheapest untried idea and it is consistent with
  everything measured.
- **Workcell problem** → the rack layout cannot be served at this reach, and that
  is a finding worth writing up rather than a bug. Say so, and consider whether
  the bay pitch or the base position is the variable.

Then item 5: `scripts/run_relocation.sh relocate`. It resolves the two-bay insert
checkpoint itself and refuses to run with the single-bay one. Gate ≥ 95%, three
held-out seeds, zero instability, zero non-finite. **Do not start it before item
4's gate passes** — a run today measures the transit failure, not the chain.

## Also outstanding, in priority order

1. **The three vision arms on the two-bay rack.** `GrappleVisionTwoSlot-Install-v0`
   exists precisely so the arms have a manipulation task that completes. Run:
   `TASK=Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Install-v0 WORKFLOW=install STAGE=2
   TAG=_twoslot HEAD=checkpoints/module_pose_head_two_slot.pth ENVS=64
   ARMS="oracle camera blind" bash scripts/certify_vision_workflow.sh`.
   **Budget it properly: about 39 minutes per arm-seed at 64 environments**, so
   three arms across three seeds is roughly six hours. Gate: camera within 10
   points of oracle, blind clearly below both.
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
