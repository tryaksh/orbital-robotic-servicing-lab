# Final session prompt

Copy everything between the rules.

---

You own `D:\6axis-space-robotics`. Act as the senior robotics simulation engineer
who owns it. **FULL AUTHORITY for a 10–20 hour unattended session.** Launch GPU
runs without asking. Make design calls, trade-offs and deletions yourself. Do not
stop to check in — if something is blocked, record what you measured, move to the
next item, and come back.

This is the **final session before this work is published** as a portfolio project.
Everything you produce has to survive a knowledgeable stranger reading it.

## The deliverable, and what "genuine" means

An **industrially credible demonstration of contact-rich zero-g servicing**, with
every claim naming the file in `evidence/` it came from.

The genuine bottleneck this project addresses — do not invent a grander one — is:

> A serviceable module held only by a **passive mechanical interface** becomes
> completely unconstrained the moment its rails release it. A parallel-jaw grip on
> a passive feature cannot resist a moment about its closing axis, so **attitude,
> not position, is the binding constraint** on every operation that moves a
> payload through free space.

That claim is earned here, four independent ways, and the strongest single piece of
evidence is that a module being carried between bays swung **end-for-end** about
its grip while the grip itself read a healthy 24 mm.

Three other findings are genuinely non-obvious and belong in the write-up:

1. **Chained success has repeatedly landed below the product of its parts, and the
   cause was never policy capacity.** Every time it was a hand-off, a phase clock,
   or a controller state. Installation went 89.41% → 96.35% with **no retraining
   at all** — a truncating clock and a settling window that kept squeezing a seated
   module.
2. **A skill's own success rate and what a chain needs from it are different
   questions.** Capture scores 88.78% alone and *fails* its gate, while the same
   checkpoint overruns its capture phase once in 192 chained installations —
   because the skill task ends an episode when its failure predicate fires and the
   chain simply lets the capture finish.
3. **Perception costs almost nothing on a one-bay rack and is brittle on two.**
   Camera equals oracle at 80.38% with one bay; with two, it collapses on one seed
   of three (86.46 / **25.00** / 83.85) while oracle stays flat.

If the relocation does not end up working, **the demonstration is still real**:
removal and installation both certify above 95% with the module held by physical
contact throughout, and the interface finding above is a result in its own right.
Say what works, say what does not, and never dress up the second as the first.

## Exact state as of 2026-08-17

### `main` — certified and working. Every report in `evidence/` describes it.

Tapered grapple pin. Suite green. Verified this session: extraction 98.44% on seed
1070 stage 2.

| | Result | Gate | Evidence |
| --- | ---: | --- | --- |
| Capture, alone | **88.78%** | **fail** | `grapple_grasp_v5_certification.json` — 100 / 87.12 / 79.22 by reset distance; 1,008 of 1,011 failures are refusals, not timeouts. The old 96.10% is retracted |
| Extract | 99.02% | pass | `grapple_extract_v14reset_certification.json` |
| Insert, one bay | 98.27% | pass | `grapple_insert_v6clock30_certification.json` |
| **Insert, both bays** | **98.34%** worse bay | **pass** | `grapple_insert_two_slot_certification.json`, pooled 98.60%, 3,004 episodes |
| **Removal chain** | **98.78%** | **pass** | `workflow_remove_retain_certification.json` |
| **Install chain** | **96.35%** | **pass** | `workflow_install_clock30retain_certification.json` |
| **Relocation chain** | **does not complete** | fail | times out in the lateral transit |
| Vision, one bay: oracle / camera / blind | 80.38 / 80.38 / 43.58 | — | `vision_workflow_*_certification.json` |
| Vision, two bays: oracle / camera / blind | 88.72 / **65.10** / 34.03 | fail | `vision_workflow_*_twoslot_certification.json`; camera per seed 86.46 / **25.00** / 83.85 |
| Pose head, one bay | 1.75 mm | — | `module_pose_head.json` |
| Pose head + bay occupancy | 2.81 mm, **occupancy 100%** | — | `module_pose_head_two_slot.json`, vs a 66.6% majority-class baseline |
| Interface axial hold | 69 N vs 66.4 N | pass | `grapple_pin_axial_pull_gate.json` |

Promoted checkpoints under `logs/rl_games/zero_g_blade_insertion_contact/`:
`grapple_grasp_l0_seed70_v5` ep 1500, `grapple_extract_l0_seed70_v13unsat` ep 5700,
`grapple_insert_l0_seed70_v10twoslot` ep 4400.

### `keyed-interface` branch — the redesign. Solves rotation, breaks extraction.

The tapered wedge replaced by **flat keyed faces between two axial stops**, built
against flight practice (ISS ORUs are gripped on a micro-square then bolted; SIROM
latches at three points; HOTDOCK is form-fit plus a lock — none makes friction
load-bearing).

| | Taper | Keyed |
| --- | ---: | ---: |
| Seated grip offset | 0.0194 m | **0.0007 m** |
| Seated grip attitude | 0.0637 rad | **0.0013 rad** |
| Rotation in extraction failures | 0.30+ rad | **0.10 rad** |
| Lateral load held, no slip | — | **120 N at 0.34 m, 40 N·m** |
| Axial pull gate | 69 N pass | 19–52 N **fail** |
| Extraction | 99.02% | **0.00%** |

Five real defects were found and fixed on the branch — reset noise sized for a
taper, 5 mm pad interference into a rigid flat, `capture_established` never firing
so the hold closure never latched, an 85 mm approach opening around a 30 mm key,
and an axial bound set too tight. **None of them moved the number.** Grip error
stayed identical to a tenth of a millimetre — 0.0349, 0.0347, 0.0347 — across runs
where noise, interference and the latch all changed. Attitude was 0.0224 rad
throughout, so it is not rotation.

**The missing piece is named by the research, not guessed.** Space grippers use
finger V-grooves and flight grapple fixtures use cone-shaped aligning bodies or
three lead-in ramps *to guide a part into place*. The taper was doing two jobs —
clamping **and** funnelling. The keyed pin is a better clamp with no funnel at all.

## The two blockers

**A. Interface axial retention.** The keyed pocket needs lead-in geometry
**dimensioned against the reset distribution**, not tuned. Constraints, all read
from `grapple_geometry.py`: pads 57 mm long, aperture 87.1 mm, slot mouth gap
36 mm (only the shaft passes it), collar face must stay behind x = 0.45 at full
insertion, grip offset −0.3395 m.

**B. Relocation reach.** On `main` the transit fails because the retreat depth is
**not reachable while holding the capture attitude** — the converged IK solver
keeps attitude to 0.0001 rad and gives up **174 mm** of position. The arm only ever
got there by twisting the wrist, which is exactly what let the module flip.

**These may be one problem.** If the *geometry* holds the module's attitude, the
arm no longer has to, and the retreat pose becomes reachable in whatever wrist
configuration works. That is the strategic reason to fix A first.

## Plan — in order, each with a gate and a time box

**1. Interface lead-in (time box 4 h).** On `keyed-interface`. **Compute the
geometry before running anything**: measure the reset's axial and lateral
placement distribution, then size ramps to cover it. Restore the deleted frustum
mesh builder (`git show main:...assets.py` has `_define_wedge`) and use it for
ramps rather than authoring new mesh code.
*Gate:* axial pull ≥ 66.36 N **and** unchanged extract v13unsat ≥ 50% on the new
geometry. If both pass, continue on the branch. **If the time box expires, abandon
the branch, return to `main`, and go to step 2** — that is a legitimate outcome and
the branch's rotation numbers are still a result.

**2. Retrain and re-certify the three skills (4–5 h).** Fine-tune from the promoted
checkpoints — this is a physics change, which this repo resumes across; never
resume across an observation or action dimension change. ~19 PPO epochs/min at 512
envs. Certify each on three held-out seeds.
*Gates:* extract ≥ 95%, insert ≥ 95% on the **worse** bay, capture reported
honestly whatever it is.

**3. Removal and installation chains (1 h).** `scripts/certify_workflow.sh`.
*Gate:* ≥ 95% each, three seeds, zero instability, zero non-finite.

**4. Relocation (3–4 h).** Trace before training: `run_relocation.sh trace` at
`EPISODES=64` reproduces the whole diagnosis in ~4 minutes and the `[CHAIN]` line
already reports the follower's leg, each conjunct of the arrival test, grip error
and tool position. If the reach limit persists, the cheapest untried fix is a
**different leg order** — cross laterally while the module is still shallow in the
first bay's rails, which constrain its attitude, and retreat only as far as the
flare plane requires at the crossing y.
*Gate:* ≥ 95%, three held-out seeds.

**5. Perception (2 h).** Find what evaluation seed 5070 draws that collapses the
camera arm to 25.00% while oracle stays flat. The head's held-out **p95 is 6.47 mm
against a 4 mm insertion lateral tolerance** while its mean is 2.81 mm — adequate
typical accuracy, inadequate tail. `sweep_camera_calibration.py` and the
collector's `--camera_offset_mm` / `--camera_tilt_mrad` exist for this. Fix by
collection coverage or label range, not by retraining blind.
*Gate:* camera within 10 points of oracle on three seeds, blind clearly below.

**6. Publish (2 h).** Rewrite `README.md`, `docs/claim_vs_evidence.md` and
`docs/portfolio.html` for a stranger. Lead with the interface finding, then the
certified chains, then the honest limits. Every number names its evidence file.

## Method rules — these are why the numbers are worth anything

1. **Compute geometry; never parameter-search it.** The previous session burned a
   night nudging constants on a geometry problem. Two failed measurements in a row
   on the same hypothesis means the hypothesis is wrong — change approach.
2. **Before believing a 0%, check `control_steps`.** A p50 of 1 means the reset
   produces unwinnable states and the policy never acted. This masqueraded as
   "extraction is 0.00%" for three runs.
3. **A constant that survives every variable is not caused by any of them.**
4. **Read constants, never restate them.** Eight failures here came from one number
   living in two places. Derive from `grapple_geometry.py`.
5. **Run the unchanged successor policy on any changed distribution before
   training on it.** Five minutes; it has saved several runs and would have saved
   more.
6. **Three seeds, never one.** A single-seed vision sweep reported a gate *pass*
   that three seeds refuted.
7. **Any phase that waits must command or retain.** Paid twice: removal 0.00% →
   98.78%, installation 85.94% → 90.10%.
8. **Never weaken a success threshold to make a gate pass.** Correcting a reset
   that produces *unwinnable* states is different, allowed, and must be stated with
   its measurement. A budget is not a threshold, but changing one obliges you to
   re-certify everything measured under the old one.
9. **Write diagnostics before anything that formats a report.** An 11-minute run
   once produced no file because a dict literal below it raised.
10. **A skill certification is not evidence about a chain.** Certify both.
11. Zero gravity, 30 Hz policy / 120 Hz physics. One Isaac process at a time —
    check `Get-Process kit` before every launch. 512 envs ≈ 0.9 GB of 12 GB.
    `train.py` stdout lags minutes; judge from `summaries/` tfevents and `nn/`
    mtime. `--max_iterations` is an absolute epoch number.
12. Commit after every stage with the numbers in the message. Keep
    `docs/status.md` current as you go. Commit to `main` as `tryaksh`, no
    Co-Authored-By trailers.

## Do not retry — each was built, measured and refuted

- **Any additive anti-rotation feature on the tapered pin.** The anti-yaw yoke cost
  insertion 67 points to buy extraction 0.13; the modelled latch jams the module in
  the rails and collapses travel from 458 mm to 25 mm. You cannot patch keying onto
  a grip that has none — that is what the keyed redesign is for.
- **Reproducing the insert hand-off as a reset distribution.** Four attempts:
  0.00%, 26.32%, 47.17%. `InsertChain-v0` reproduces it at 93.06% by running the
  real capture.
- **Fine-tuning insert on the chain's own distribution.** 300 epochs moved the
  install chain 89.41% → 88.37%. The hand-off costs the skill 2.5 points.
- **Raising the gripper drive to its rated torque.** Capacity *fell*.
- **Force sensing for pose robustness.** Worse beyond the trained range.
- **Rate-limiting the relocation transit's final leg**, and **commanding transit
  attitude at full authority** (starves translation), and **a relative lateral
  cross** (carries 93 mm of drift into a 72.5 mm channel).
- **The scripted realign in the install chain.** `ALIGN_STEPS` defaults to 0.

## Cleanup — do this, it is part of the deliverable

- **Delete the anti-yaw yoke entirely** — refuted, off by default, and superseded
  by keyed flats. It touches ~10 files including `run_yoke_gates.sh`. Keep its
  measurements in `docs/status.md` and `evidence/`; delete the code.
- **Delete or fix `--workflow full`.** It goes non-finite by control step 10 and
  has never been certified. If the relocation works, `full` is redundant.
- Resolve the two long-standing smoke failures or delete the tasks: the contact
  task's inverted finger commands and the capture-in-slot task.
- **Fix the label defect in `play.py`**: it reads the flares' tri-state
  `collision_enabled` with `bool()`, so IsaacLab's `None` ("leave as authored")
  reads as disabled and stamps **every** grapple-pin report
  `out_of_distribution: true` with `gate.applies: false`. `train.py` reads the same
  field correctly. No rate is wrong; the label on all of them is. Fix, then re-run
  to re-label.
- Delete stale branches `agent/zero-g-blade-swap` and `backup/pre-trailer-strip`
  once you are confident. Keep `keyed-interface` until step 1 resolves.
- Keep every retracted result in `docs/status.md` and `evidence/RETRACTED.md`. The
  record of what was refuted is part of what makes this credible.

## What I want at the end

- The chains certified, or a precise account of exactly where each stops and what
  the measurement says.
- `docs/status.md` carrying every result including the negative ones.
- `README.md`, `docs/claim_vs_evidence.md` and `docs/portfolio.html` written for a
  stranger, current, every number naming its evidence file.
- `CLAUDE.md` rewritten as the state of the finished project, not a to-do list.
- A short plain-English table: what works and at what rate, what does not and why,
  what was refuted.
- The repo clean enough that nothing in it misleads.

---
