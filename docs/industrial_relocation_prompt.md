# Session prompt: remove the bottlenecks, build the industrial demonstration

Copy everything between the rules into a fresh session.

---

You own `D:\6axis-space-robotics`, on branch **`industrial-relocation`**. Act as
the senior robotics simulation engineer who owns it. **FULL AUTHORITY, 14–20 hours
unattended, overnight.** Launch GPU runs without asking. Do not stop to check in.
Do not ask permission for any change described below — it is already granted.

**The relocation works by the end of this session, or you report exactly which
physical quantity prevents it and what geometry would fix it.** Those are the only
two acceptable outcomes. "Blocked, needs more time" is not one.

You may and should: **move the robot, add a linear axis, redesign the gripper's
fingers, redesign the rack, redesign the module interface, and retrain every
policy from scratch.** Every certified number in this repository was produced on a
workcell that demonstrably cannot do the job. Preserving those numbers is not a
goal. Beating them on a workcell that *can* is.

`main` is certified and published; leave it alone. Everything happens here.

## The deliverable

```
CAPTURE  ->  EXTRACT  ->  TRANSIT ACROSS  ->  INSERT INTO BAY 2
```

One continuous episode, module held by real contact throughout, no fixed joint
anywhere. **>= 95% over three held-out seeds, zero instability, zero non-finite.**

Plus: every skill and both existing chains re-certified on whatever you change. A
relocation bought by breaking removal is not a demonstration.

## Why it fails today — measured, not suspected

Read `evidence/relocation_reach_boundary.json` and
`evidence/grapple_pin_keyed_interference.json` before planning anything.

### Bottleneck A — the arm cannot hold the required attitude at the required depth

Driving position and the head-on capture attitude at full authority, the tool
parks at local **x = −0.0258**, which is 0.4242 m forward of the base at x = −0.45.

| Pose the task needs | Tool x | Short by |
| --- | ---: | ---: |
| Capture in bay 1 | +0.2434 | reachable |
| **Extraction end** | −0.1145 | **88.7 mm** |
| **Transit retreat** | −0.1928 | **167 mm** |

Two controls make this a workcell fact rather than a solver artefact:

- **attitude free** — every depth converges to 0.00 mm, including 64 mm past the
  retreat. The arm reaches them; not pointing the right way;
- **lift** — the rack's flares are 50 mm plates and the module is 35 mm thick, so
  a 45 mm lift clears them and lets the module cross at the *extraction* depth
  instead of behind the flare plane. The parking point does not move, at 0, 50,
  100 or 200 mm of lift.

It is a **trade**, not a wall: a 2,000-step servo reaches the extraction pose to
3.6 mm by allowing 0.0114 rad of attitude error. Attitude exchanges for reach at
about **7.5 m/rad** here. That is why the transit reached its waypoints when
nothing commanded the attitude, and why the module then swung end-for-end.

**The arithmetic that tells you what to build.** If that near edge is
base-relative at 0.4242 m, reaching the retreat needs the base at or behind
**x = −0.617 m**. At −0.65 the capture pose sits 0.86 m out and 0.57 m up: a
1.03 m radius against the UR10e's 1.30 m reach. Comfortable. **Measure it in
Phase 0 — that is a hypothesis with a number, not a plan.**

**The trap that ate the last attempt.** `--robot_base_x` had never moved anything:
the edit went to a config `parse_env_cfg` discarded and `configure_robustness`
then overwrote. Fixed, and the report now reads the root position back out of the
simulation. But `GRAPPLE_HEAD_ON_ARM_JOINT_POS` is solved for the base at −0.45,
so **moving the base without re-solving the spawn poses makes the arm start
200 mm short of its own capture pose and shove the module 137 mm.** Re-solve
first, every time.

### Bottleneck B — a passive module feature cannot hold attitude, and cannot be made to

A parallel-jaw grip cannot resist a moment about its own closing axis: the pads'
normals lie along that axis, and a normal force cannot oppose a moment about its
own direction. Only friction can, and friction loses.

The obvious fix — a spool shape, narrow middle with a disc each end so the pads
are trapped by form — was built and **cannot be installed on this hand**:

| | |
| --- | ---: |
| Room in front of the seated pads | **7.9 mm** of half-height |
| A stop the pads cannot splay past must exceed | **43.5 mm** |
| Gripped-section heights admitting any forward stop, of 43 swept | 7, all under 7 mm |

The volume ahead of the pads belongs to the hand: solid body to 90 mm from the
flange, moving arms 90–105 mm. And it is self-defeating — a thinner gripped
section lets the hand close further, bringing the arms in.

**So the lock moves into the end-effector.** That is what flight hardware does:
Dextre's OTCM grips a micro-square then bolts it, the SSRMS snares then rigidizes,
SIROM and HOTDOCK are form-fit plus an active latch. None makes friction
load-bearing. **Change the gripper.**

## The plan

Each phase has a gate. Do not start the next before it passes. Commit after each
with the numbers in the message.

### Phase 0 — settle the workcell by computation (2 h, mostly CPU)

Nothing downstream is worth doing until a geometry exists that holds the attitude
everywhere the task needs it.

Write a loop that, per candidate base position, **re-solves the head-on spawn
pose** and then sweeps the reach. `calibrate_grasp_pose.py` has the machinery:
`--robot_base_x`, `--sweep_offset_x`, `--sweep_offset_z`, `--attitude_authority`,
`--free_orientation`, `--steps 3000`, one environment per combination in a single
app launch.

Sweep base x over at least −0.45, −0.55, −0.65, −0.75, −0.85, solving the **four
poses the task needs** at each: capture in bay 1, extraction end, transit retreat,
staging in bay 2.

If no single base position holds all four, the industrial answer is a **linear
rail**: a prismatic base repositioned during *scripted* phases only. Model it as a
base pose written per phase, not as a seventh actuated joint — that keeps every
checkpoint resumable because no action dimension changes. If a rail is not enough
either, move the rack, change the bay pitch, or mount the arm above the rack.
**Something in this space works; find it.**

*Gate:* one base position, or one rail schedule, at which all four poses converge
to **<= 2 mm and <= 0.01 rad** with orientation at full authority. Write
`evidence/workcell_reach_solution.json`.

### Phase 1 — decide how much interface work you actually need (1 h)

**Run the unchanged extract and insert policies on the Phase-0 workcell before
building anything.** Five minutes each. The relocation's blocker is the arm; the
interface is what makes the failure ugly. If the skills survive the new workcell
at anything near their old rates, Phase 2 gets smaller.

*Gate:* a measured number for each unchanged policy on the new workcell, recorded.
That number decides Phase 2's scope. It does not gate progress.

### Phase 2 — put the lock in the end-effector (3 h)

Two candidates. Do both if the first is not decisive.

1. **Latch on release.** `GrappleLatch` is implemented and off. It was refuted
   engaged while the module was *still in the rails*, where it jams the module and
   collapses travel from 458 mm to 25 mm. Engaging it the instant the rails release
   is **untested**, and that is exactly where the lock is needed. Config change.
2. **V-grooved pads.** Give the 2F-85's inner fingers a V notch in collision
   geometry and match the pin's gripped section to it. A V-groove constrains two
   axes by form. This changes the *robot*, so re-measure
   `evidence/gripper_collision_envelope.json` with
   `scripts/measure_gripper_envelope.py` and re-derive every number in
   `grapple_geometry.py` from it. Budget for that; it is the point.

*Gate:* `scripts/grasp_diagnostics.py --load_axis yaw` shows held rotational load
improve **>= 5x** on the plain pin's baseline, **and** axial pull still makes
**>= 66.36 N**, **and** `scripts/check_pin_gripper_clearance.py` passes.

### Phase 3 — re-derive and smoke (1 h)

`check_pin_gripper_clearance.py`, `measure_pin_design_window.py`, `pytest tests/`,
and `train.py --smoke` on capture, extract and insert.

*Gate:* suite green, three tasks smoke clean, the physical-grasp contract prints
the pin sections collidable with the software fixture off.

### Phase 4 — retrain the three skills (5 h)

Fine-tune from the promoted checkpoints. **This is a physics and geometry change,
which this repository resumes across — say so in the commit.** If Phase 2 changed
an observation or action dimension, train from scratch instead; never resume
across those.

~19 PPO epochs/min at 512 environments. Judge progress from `summaries/` tfevents
and the `nn/` checkpoint mtime, never from `train.py` stdout, which is
block-buffered and lags minutes.

### Phase 5 — certify the three skills (1.5 h)

`scripts/certify_demo_policies.sh`, three held-out seeds each.

*Gate:* extract **>= 95%**, insert **>= 95% on the worse bay**. Capture reported
honestly whatever it is — its skill task ends an episode when its failure
predicate fires while the chain simply lets the capture finish, so 88.78% alone
and one overrun in 192 chained installations are both true. It must not be worse
than main's 88.78%.

### Phase 6 — the two existing chains (1 h)

`scripts/certify_workflow.sh`.

*Gate:* removal and installation both **>= 95%**, three seeds, zero instability,
zero non-finite. If either regressed below main's 98.78% / 96.35%, fix it before
touching the relocation.

### Phase 7 — the relocation (2 h)

`scripts/run_relocation.sh trace` reproduces the whole diagnosis at `EPISODES=64`
in about four minutes; the `[CHAIN]` line reports the follower's leg, each
conjunct of the arrival test, grip error, and tool position in the module's frame.
Trace before certifying.

*Gate:* **>= 95%**, three held-out seeds, zero instability, zero non-finite.
`evidence/workflow_relocate_certification.json`.

### Phase 8 — perception on the new geometry (2.5 h)

Any geometry change invalidates the pose head. Re-collect 60,000 two-bay frames
(`collect_grapple_vision.py`), retrain (`train_pose_head.py`), re-certify all three
arms (`certify_vision_workflow.sh`, two-bay settings).

*Gate:* camera within **10 points** of oracle, blind clearly below both.

**Replicate the camera arm.** It is *not* deterministic and the state pipeline is:
two runs with identical seed, task and checkpoints diverge at the first episode
and six runs span 80.73–86.46%, while two oracle runs are bit-identical across all
192 episodes and 21 columns. Run the camera arm **twice per seed** and report the
spread. `evidence/vision_camera_run_variance.json`.

### Phase 9 — publish (1 h)

Update `README.md`, `docs/status.md`, `docs/claim_vs_evidence.md`,
`docs/portfolio.html`, `docs/service_interface_spec.md`, `CLAUDE.md`. Every number
names its evidence file; keep every retracted and negative result.
`scripts/check_evidence_links.py` must pass — a test enforces it.

Open a PR from `industrial-relocation` to `main` with the full before/after table.
**Do not merge without Phases 5, 6 and 7 all green on three seeds.**

## Method rules. These are why the numbers are worth anything

1. **Compute geometry; never parameter-search it.** A session was burned nudging
   constants on a geometry problem. Two failed measurements on one hypothesis
   means the hypothesis is wrong — change approach.
2. **Before believing a probe, prove it moves what it measures.** Paid three times
   in one session: two versions of the pin-clearance check condemned the
   *certified* taper before the third was right, and `--robot_base_x` had never
   moved anything. A control that changes nothing means the probe is broken.
3. **Before believing a solver, converge it — and know which side of a trade it
   converged to.** A pose called unreachable on a 400-step residual converges to
   0.0060 mm at 3,000. The same pose reads 3.6 mm at 0.0114 rad or 88.7 mm at
   0.0002 rad depending on the authority orientation is driven with.
4. **Read constants, never restate them — including the ones that already exist.**
   `PALM_FACE_FROM_FLANGE_M` said "nothing can sit closer to the flange than this",
   and a redesign put a section 13 mm inside it because nobody read it.
5. **Before believing a 0%, check `control_steps`.** A p50 of 1 means the reset
   produces unwinnable states and the policy never acted.
6. **A constant that survives every variable is not caused by any of them.**
7. **Run the unchanged successor policy on any changed distribution before
   training on it.** Five minutes; it has saved several runs.
8. **Any phase that waits must command or retain.** An arm that stops commanding
   while gripping in zero gravity is not holding still, it is being pushed. Paid
   twice: removal 0/570 → 569/576; installation 84.90% → 21.35% for a 2 s pause.
9. **Never weaken a success threshold to make a gate pass.** Correcting a reset
   that produces *unwinnable* states is different, allowed, and must be stated
   with its measurement.
10. **A skill certification is not evidence about a chain.** Certify both.
11. **Three seeds are three samples of the configuration, not of the run.** A
    single-seed sweep once reported a *pass* three seeds overturned; three seeds
    then reported a *failure* a re-run overturned.
12. Zero gravity, 30 Hz policy / 120 Hz physics. One Isaac process at a time —
    check `Get-Process kit` before every launch; one survived five hours and
    slowed everything ~40%. 512 environments ≈ 0.9 GB of 12 GB.
13. Commit after every phase with the numbers in the message. Author `tryaksh`,
    no Co-Authored-By trailers.

## Do not retry. Each was built, measured and refuted

- **Any additive anti-rotation feature on the tapered pin.** The yoke cost
  insertion 67 points to buy extraction 0.13; its code is deleted and a contract
  test keeps it out. The latch jams the module *in the rails* — which is why
  Phase 2 tests it on **release**.
- **A keyed or spool-shaped module feature.** Refuted by geometry, not
  performance: the forward stop needs 43.5 mm and has 7.9 mm at every gripped
  height. Change the hand, not the part.
- **Fixing the relocation by leg order or by lifting over the flares.** Both
  refuted this session: every crossing point is past the attitude limit — 167 mm
  behind the flare plane, 88.7 mm over it — and the limit does not move with
  height.
- **Reproducing the insert hand-off as a reset distribution.** Four attempts:
  0.00%, 26.32%, 47.17%. `InsertChain-v0` reproduces it at 93.06% by running the
  real capture.
- **Fine-tuning insert on the chain's own distribution.** 300 epochs took the
  install chain 89.41% → 88.37%.
- **Raising the gripper drive to its rated torque.** Capacity *fell* — a wedge
  turns closing force into thrust along the pull axis.
- **Force sensing for pose robustness.** Worse beyond the trained range. Force has
  to be actionable, not merely observable.
- **Rate-limiting the transit's final leg**, **full-authority transit attitude**
  (starves translation), **a relative lateral cross** (93 mm of drift into a
  72.5 mm channel), and **reading the transit's arrival in 3-D** (per-axis is the
  correct test).

## The state you start from

Certified on `main` — beat or preserve these:

| | Result | Evidence |
| --- | ---: | --- |
| Removal chain | **98.78%** | `workflow_remove_retain_certification.json` |
| Installation chain | **96.35%** | `workflow_install_clock30retain_certification.json` |
| Extract | 99.02% | `grapple_extract_v14reset_certification.json` |
| Insert, one bay | 98.27% | `grapple_insert_v6clock30_certification.json` |
| Insert, both bays, worse bay | 98.34% | `grapple_insert_two_slot_certification.json` |
| Capture, alone | 88.78%, fails its gate | `grapple_grasp_v5_certification.json` |
| Vision, two bays: oracle / camera / blind | 88.72 / 84.90 / 34.03 | `vision_workflow_*_twoslot_certification.json` |
| Interface axial hold | 69.0 N vs 66.4 N required | `grapple_pin_axial_pull_gate.json` |
| **Relocation** | **does not complete** | `relocation_reach_boundary.json` |

Promoted checkpoints under `logs/rl_games/zero_g_blade_insertion_contact/`:
`grapple_grasp_l0_seed70_v5` ep 1500, `grapple_extract_l0_seed70_v13unsat`
ep 5700, `grapple_insert_l0_seed70_v10twoslot` ep 4400. Pose heads under
`checkpoints/`.

Read `CLAUDE.md` first, then `docs/status.md`, then `evidence/RETRACTED.md`.

## What I want at the end

- **The relocation certified at >= 95% on three held-out seeds** — or the exact
  physical quantity that prevents it and the geometry that would fix it, with
  Phase 0's and Phase 2's outcomes both stated so the next person knows whether
  the workcell, the interface, or both were insufficient.
- Every skill and both chains re-certified on the changed geometry, with a
  before/after table against the numbers above.
- `docs/status.md` carrying every result including the negative ones.
- A pull request to `main` a knowledgeable stranger could review.

---
