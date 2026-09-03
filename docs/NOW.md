# Now

Verified repository state. Evidence status is mechanical in
[`evidence/MANIFEST.json`](../evidence/MANIFEST.json); bounded tasks are in
[`NEXT_WORK.md`](NEXT_WORK.md). Last verified: 2026-09-03 on
`paper/serviceability-qualification`, based on `main` at `bccce6d`.

Everything is simulated. Nothing has run on hardware.

## Trust snapshot

| Item | Verified state |
| --- | --- |
| Evidence | 53 canonical, 11 retracted, 159 historical; quote only canonical |
| Source provenance | 13 reports carry runtime source bindings; two match the working source, one is mechanically recovered, and ten older reports remain lost because they used uncommitted code |
| Current completion result | 22/24, **91.67%**, after visible rack retention engages, both robot-side supports release, and the rack alone holds for at least 0.70 s |
| Boundary decision | **not qualified**; only entry attitude is supported. The rack-clearance axis was re-measured after a sweep defect: `--rack_lateral_clearance_mm` moved each bay's guides and left its lips and entry flares behind, and 6 mm per side goes from 0/64 to 36/64 once the mouth moves with the walls |
| Live RGB-D service | complete but fragile: one continuous episode does the whole changeout with 1,772/1,772 detections, and the pooled cohort scores **4/24** against a paired oracle-pose control at **20/24** on the same code path |
| CI architecture | core modules and CPU tests do not require optional FastAPI imports |
| Checkpoints | reports contain hashes, but weights under `logs/` and `checkpoints/` are absent from a clone |
| Hardware claim | none |

**`check_criterion_currency.py --all` now flags all 121 reports, and that is the
tool working rather than 121 invalidated numbers.** It flags any report generated
before a change to a file that *can* define its criterion, and
`scripts/run_workflow_demo.py` changed four times on 2026-09-03. Every one of
those changes is additive with a behaviour-preserving default:
`--fiducial_guard_bounds` defaults to the shipped estimator bounds,
`--rack_clearance_scope` defaults to `guides` which is what the published sweep
ran, `--remove_entry_flares` defaults off, and the fourth turns a silent deadlock
into an argument error for a combination no published report used successfully.
`tests/test_guard_bounds_arm.py` holds the first three defaults in CI. No
published number moves.

T0 remains open for the ten source-bound reports whose exact uncommitted code
cannot be recovered. New strict-chain and RGB-D evidence starts from clean
commits; the bounded audit finds no lost binding in either new RGB-D report.

## What runs

One continuous zero-gravity episode uses a UR10e to capture, extract, carry,
insert and release a compute module. There is no world constraint, teleport,
direct module pose write or hidden carrier.

| Phase | Controller that executes |
| --- | --- |
| Capture | PPO capture policy |
| Extract | PPO extraction policy |
| Transit | collision-checked solved IK with a robot-side form lock |
| Insert | guarded axial advance while the deployed estimate is inside the derived entry envelope |
| Release | visible rack pawls engage only after measured seating; simultaneous hand and compliant-latch release is followed by a 0.70 s rack-only recheck |

The robot rail indexes the world-fixed robot base and does not model its own load
path. A phase is labelled learned only when policy actions actually step it.

## Measured state

### Chain and skills

The current strict chain with destination retention scores **91.67%**: 22/24
fixed-cohort episodes over three held-out seeds, Wilson 95% **[74.2%, 97.7%]**.
It still fails the unchanged 95% full-chain gate. The three seed results are
6/8, 8/8 and 8/8. There were no non-finite episodes.

The prior **97.92%** result (94/96) is retained as a legacy supported-settle
baseline. It did not independently release both robot-side supports and then
recheck the module under the rack alone, so it is not the current completion
rate.

| Certificate | Result | Decision |
| --- | ---: | --- |
| Capture v7m130, derived rack | **86.90%**, 7,829/9,009 | misses 95%; the earlier current-rack arm was **85.69%** |
| Extract v18pin, derived rack | **87.64%**, 7,891/9,004 | misses 95%; the earlier pin certificate was **87.75%** |
| Learned insert v20chain | **0.00%**, 0/1,536 | preserved negative baseline |
| Learned insert v24, isolated | **36.77%**, 1,103/3,000 | does not transfer to the chain |
| Learned insert v24, real chain handoff | **0.00%**, 0/96 | not selected |
| Guarded insert, real chain handoff | 94/96 under the legacy supported-settle criterion | selected; strict completion with rack retention is 22/24 |

Insertion was not extended blindly. The audit corrected action scaling, matched
the skill and chain handoff geometry, added handoff-conditioned resets, projected
the controller onto module-relative assembly state, and tested staged load-path
release. Learned v24 still fails every recorded predecessor handoff. Its isolated
certificate therefore describes late-stroke states, not the state its caller
delivers. More epochs are not justified until that interface distribution is
made identical and the losing arm is replayed.

The older insertion diagnosis remains preserved: three objective arms ended at
84.26, 84.61 and 84.58 mrad against a **52.4 mrad** success tolerance. Changing
the reward did not move the interface-limited attitude.

### Destination load transfer

The current-source no-rack control exactly reproduces the strict baseline at
**17/24 (70.83%)**. Enabling only the visible rack capture raises the identical
fixed cohorts to **22/24 (91.67%)**, a gain of five episodes / 20.83 percentage
points. All **22/22** episodes that reach the unchanged measured-seating
predicate engage the rack, survive the rack-only recheck and record 0.0 m / 0.0
rad maximum Rack-to-module drift. The remaining two fail upstream and never
engage, so the full-chain 95% gate remains failed rather than being attributed
to load transfer.

The mechanism is two visible 2.5 x 20 x 20 mm rack-owned pawls, with 2.5 mm
rear-face overlap, 0.5 mm no-snap face clearance and an 81.633 mm open half-gap.
Their simulated load path is a disclosed 600 N / 30 N-m break-rated fixed joint
from `Rack` to `SpareBlade`, enabled only after the live seating predicate. It is
not a world constraint and never writes the module pose. Visual pawl contact is
not simulated; the idealized joint carries the load.

### RGB-D perception

The former passing certificate used a tilted tag floating 90 mm above the
current module and is retracted. The datum remains flush with the module top
face. At the current 640 px resolution the single centred datum detected in
937/1,024 held-out frames (**91.50%**) and 683/683 critical-bay frames, with
position p95 1.19 mm, orientation p95 10.93 mrad and exact occupancy under
unchanged gates. The prior camera and decoder arms remain as preserved losers.

**The continuous blocker was geometric, and it is closed.**
[`rack_sightline_occlusion_v1.json`](../evidence/rack_sightline_occlusion_v1.json)
derives, without a simulator, that the destination bay's own vertical lead-in --
an 80 x 60 x 18 mm plate at 12 degrees over the bay centre line, hanging 25 mm
above the module's top face -- covers a centred flush datum for **154 mm of the
529 mm seating stroke** from both fixed cameras. It is a roof: clearing it means
looking under an 82 mm span through 25 mm of headroom, which foreshortens the
marker cell below the resolution the estimator needs, so no camera placement
fixes it. The derivation validates itself against the recorded loss depth of the
dual-camera run before it reports.

The one change is the datum, not the camera: **two flush plates on the same
plane**, ArUco 23 aft and ArUco 15 forward at module-frame x = ∓0.115 m. The
separation is derived -- it must exceed the lead-in's 203 mm shadow, and each
plate must stay in frame, which leaves ∓[0.1025, 0.1275] m; this is that
interval's centre. Marker size, quiet zone, plate plane, camera placement, lens,
resolution and every estimator gate are unchanged.
[`rack_sightline_datum_pair_v1.json`](../evidence/rack_sightline_datum_pair_v1.json)
reports no depth of the stroke where both plates are unreadable, and that holds
on the primary camera alone.
[`servicing_camera_geometry_v4_datum_pair.json`](../evidence/servicing_camera_geometry_v4_datum_pair.json)
keeps 64/64 workflow-envelope poses covered with 8.62 px minimum marker cell
against the unchanged 8.0 px requirement; neither plate covers the envelope alone
(36/64 and 32/64), which is the measurement that says why there are two.

A logic defect was also fixed earlier: missed detections could propagate the
module as if attached to the moving tool before capture. Reset drains the tiled
camera's blank startup buffers, and a complementary fixed RGB-D view was added
for rack entry. The estimator still holds the last observation until physical
capture and fails closed when no current camera can see a datum.

### The continuous demonstration

**One complete continuous episode now exists, on clean source.**
[`rgbd_strict_rack_retention_datum_pair_seed6070.json`](../evidence/rgbd_strict_rack_retention_datum_pair_seed6070.json),
commit `7a82db2`, tracked worktree clean:

| Stage | What the run recorded |
| --- | --- |
| Capture | PPO capture policy |
| Extraction | PPO extraction policy |
| Transit | robot-carried on the form lock; 1.05 mm and 3.27 mrad maximum tool-to-module drift |
| Insertion | guarded advance, 563 advancing control steps, terminal axial target 0.676 m -- the derived seated plane -- reached at module centre 0.6763 |
| Seating | all seven insertion conditions true, including axial depth; still true after the 0.70 s supported settle |
| Release | both robot-side supports released; `all_conditions_including_released_gripper` true |
| Rack only | pawls engaged after the unchanged measured-seating predicate at step 1720; 0.733 s rack-only recheck observed with 0.0 m and 0.0 rad drift |
| Perception | **1,772/1,772 detections, zero failures, zero consecutive failures**; forward plate carried 1,232 and aft plate 540 |

No hidden movement, no simulator-known module position, no teleport, and no
tolerance was changed: the guarded advance ran on the deployed estimator's own
2 mm / 15 mrad bounds throughout, which is eight times tighter than the entry
flare's catch and is now reported as the tolerance that applied.

**This is n = 1.** It is a demonstration, not a rate. The pooled RGB-D chain
certification is T1 and is the next measurement.

### What perception costs, measured as a substitution

**The chain has never been certified on camera-derived state before, because it
could not finish an episode.** With the flush datum pair it can, and the first
pooled cohort is a controlled ablation rather than a single number: the same
task, the same three held-out seeds, the same eight environments, the same
checkpoints, the same guard and the same observation terms, with only the
module-pose source changed.

| Arm | Result | What differs |
| --- | ---: | --- |
| State task, strict chain with rack retention | **22/24, 91.67%** | the published chain |
| Vision task, module pose from the simulator | **20/24, 83.33%**, Wilson [64.1%, 93.3%] | the vision profile's observation terms and camera cadence |
| Vision task, module pose from the cameras | **4/24, 16.67%** | the estimator, and nothing else |

The 8-point step is the vision profile itself and is inside the interval at
n = 24. The **67-point step is the estimator**, and because the two vision arms
differ by one substituted term it is an ablation rather than an inference.

The oracle arm's successful episodes seat at 0.8 mm lateral, 2.2 mm axial and
4.3 mrad, so the pipeline the datum pair unblocked -- two cameras, the guarded
advance, the rack pawls and the release interlock -- works at eight environments
when the pose is good.

**Where it loses: extraction, not insertion.** Thirteen of the twenty-four
camera-driven episodes time out in the extract phase and never engage the form
lock; the module is frequently lost outright. Only three are held by the guard's
attitude bound. The estimator's own error on healthy episodes is about 2 mm.

**The cheap fix is ruled out, by measurement.** The camera runs at half the
control rate, so the pose the policies read is a staircase and a differenced
estimate is zero on one control step and a full jump on the next. With the arm
held still, the velocity channel reads **17.02 mm/s** at the deployed 0.10 s
filter while the *identical* differencing and filtering applied to the
simulator's own pose reads **3.38 mm/s** -- so the estimator contributes
13.65 mm/s, against the 0.69 mm/s a seated module actually moves at. Sweeping
the filter's time constant does not help: at 1 s the mean falls to 9.01 mm/s and
the p95 *rises* from 29 to 59, because a first-order filter integrates the random
walk of held estimates. The shipped 0.10 s is already the best of the sweep.
[`estimator_surrogate_velocity_channel_v1.json`](../evidence/estimator_surrogate_velocity_channel_v1.json)

**And the estimator is no worse on the episodes it loses.** Across the three
vision seeds the winning episodes carry 1.89, 2.00 and 2.11 mm of mean estimator
error and the losing ones carry 2.29, 5.99 and 1.98 mm. The long detection
dropouts a losing run records -- up to 2,865 consecutive misses -- are what
happens after the module has left the cameras' useful envelope, not what put it
there.

So the fix is the training distribution, and
`Isaac-ZeroG-Blade-GrapplePin-{Grasp,Extract,Insert}Noised-v0` are it: the four
module-derived observation terms read a surrogate whose residual, sample-and-hold
and miss rate are inverted from the estimator's own certification rather than
chosen, and whose velocity is the same finite difference the deployed estimator
manufactures. `grapple_extract_l0_seed70_v19noised` resumes the certified v18pin
checkpoint on that task at the same seed, one change from a published arm.

**This is the expected cost of an untrained transfer, and it is not a perception
defect.** Capture, extraction and the guard were trained on simulator state and
are deployed against an estimator with no student training, no distillation and
no estimator error in their training distribution. Published pipelines that do
that transfer properly still lose 20 to 25 points (a state-privileged teacher at
98% distils to a vision student at 73% in *Residual RL for Precise Assembly*,
arXiv 2407.16677). Doing none of it costs 67.

### Serviceability boundary

**Where the closed form stands, in one paragraph.** It is confirmed on four
things and contradicted on one, and the contradiction has an identified cause
rather than a shrug. Confirmed: `2c/L` as a bound on the attitude a *seated*
module holds; the grip criterion's *mechanism*, since the two points it flags are
the only two whose failing episodes sit further off the pin than their successful
ones; the *upper* clearance bound, since 16 mm per side loses episodes before
delivery at a Wilson-separated rate; and the gate that says this bay needs a
correcting lead-in, which fires before any simulation and which the bay satisfies
with a 12-degree flare. Contradicted: the *lower* clearance bound, because a
module does not carry its hand-over attitude to the seated plane. Diagnosed
rather than contradicted: the base-offset axis, whose loss is entirely inside the
learned phases with the channel untouched, so it bounds this policy and not this
geometry. Two axes remain scope limits with no simulated arm at all -- the capture
interface's load capacity, and both idealized load paths.

**The instrument had a defect and it changed one axis completely.**
`--rack_lateral_clearance_mm` selected scene attributes whose name contains
"guide". Each bay's two upper lips and two entry flares are placed from
`GUIDE_CENTER_OFFSET_Y` when `assets.py` is imported, so the flag moved four
bodies per rack and left eight where they were -- a rack whose mouth and walls
disagree by exactly the clearance change rather than a narrower channel.

`--rack_clearance_scope channel` translates the lips and the flares with the
guides. Same seed, same checkpoints, one flag:

| clearance | guides only (published) | channel (corrected) |
| --- | ---: | ---: |
| 6 mm per side | **0/64**, 62 of 64 jam at the mouth | **36/64 = 56.25%**, zero jams |
| 16 mm per side | 26/64 = 40.6% | 27/64 = 42.2% |

So the 6 mm point flips entirely and the 16 mm point does not move. Both arms
are kept: [`chain_robustness_sweep_n64.json`](../evidence/chain_robustness_sweep_n64.json)
is the guides-only sweep and
[`chain_robustness_sweep_n64_channel_v1.json`](../evidence/chain_robustness_sweep_n64_channel_v1.json)
is the corrected one.
[`serviceability_boundary_validation_n64_channel_v2.json`](../evidence/serviceability_boundary_validation_n64_channel_v2.json)
is the decision on the corrected sweep, and the pre-correction decision remains
at [`serviceability_boundary_validation_n64_v1.json`](../evidence/serviceability_boundary_validation_n64_v1.json).

| Dimension | Corrected state (n=64) |
| --- | --- |
| Rack clearance | mismatch, and now on the *lower* bound: 6 mm/side is analytically infeasible and scores 56.25% against nominal's 54.69%. The upper bound holds -- 16 mm/side loses 0.203 of its episodes before delivery against nominal's 0.031, Wilson-separated, which is the grip criterion's own prediction |
| Module section | mismatch; 120x16 loses 0.156 before delivery against nominal's 0.031 -- the direction the grip criterion predicts, not separated at 64 episodes -- and 140x26 shows no jam at all |
| Robot base offset | mismatch, and it is not a geometric one: +10 mm clears the kinematic gate, the channel is untouched, and 60 of 63 failures time out inside the *learned* phases. This is a policy trained at one base position |
| Entry attitude | supported in simulation against the derived `2c/L` boundary |
| Capture geometry | analytical only; no current contact/load certificate |
| Load path | destination transfer supported in 22/22 eligible simulations; both robot- and rack-side joints remain idealized |
| Base compliance | excluded; the fixed robot root prevents the authored spring from deflecting |

**Where the closed form is wrong, and why.** The lateral clearance bound is
two-sided and it is right on one side. The upper bound -- a resting module may
not exceed the seating tolerance, 11.781 mm per side -- governs the grip and is
confirmed. The lower bound -- the channel must admit the attitude the transit
hands over at, 10.35 mm per side -- is contradicted. `2c/theta` at 46 mrad and
6 mm of clearance says a module should wedge at 261 mm of a 529 mm stroke, and
all 64 finished at 0.5 mm of axial error. **The hand-over attitude is corrected
during the stroke rather than carried through it**, so the bound belongs on
whatever does the correcting -- the 12-degree entry flare, or the guarded advance
refusing to push -- and not on the channel. `--remove_entry_flares` is the run
that separates those two, and it is queued.

**This does not contradict the entry-attitude axis, and the distinction is the
useful part.** `2c/L` is confirmed as a bound on the attitude a *seated* module
holds -- the recorded seating sweep measures 0.87 to 1.02 of it across eight
points -- and it is contradicted as a bound on the clearance an *entering* module
needs. Those are different claims about the same law. A module that has arrived
is limited by the channel it is sitting in; a module on its way in is not
carrying its hand-over attitude, because something squares it during the stroke.
The law holds; what was wrong was applying it to the wrong state.

**And the comparison itself was asking the wrong question.** At the nominal
design point, 27 of 29 failures reach the final phase with the form lock engaged
and miss the 2.5 mm terminal gate. Two episodes in five are lost at the design
point to a mode no serviceability criterion claims to predict, so a pooled-rate
comparison makes every boundary point clear that noise floor before a Wilson
interval can separate it.
[`boundary_failure_modes_v1.json`](../evidence/boundary_failure_modes_v1.json)
counts the same episodes by the failure each criterion predicts -- the grip
criterion against episodes that never deliver the module, the entry criterion
against episodes that jam short of the seated plane -- and on that reading the
16 mm clearance point supports the boundary where the pooled protocol called it
a mismatch.

**A rate is not the only thing an episode records, and the mechanism separates
where the rate does not.** The grip criterion bounds how far a pad may slide off
the pin, so where it is violated the failing episodes should sit further off the
pin than the successful ones. Across all seven points that difference is:

| point | tool-to-pin offset, failing minus successful | grip criterion |
| --- | ---: | --- |
| rack_lat_16mm | **+1.43 mm** | violated by 2.89 mm |
| section_120x16 | **+1.18 mm** | violated by 3.92 mm |
| nominal | -0.06 mm | clear |
| base_x_-0.70 | -0.04 mm | clear |
| section_140x26 | -0.04 mm | clear (it fails the entry bound instead) |
| rack_lat_6mm | -0.16 mm | clear |
| base_y_+10mm | -3.04 mm | clear; the module is lost, so this measures a lost module |

The two points the criterion flags are the only two where the signature is
positive, and no point it clears shows one. `section_120x16`'s mode rate does not
separate at 64 episodes -- 0.156 against nominal's 0.031, with intervals
overlapping by two points -- but its mechanism does, on the same episodes, with
no extra sample.

The n=16 sweep is not comparable: on current source it produces stub episodes
at 16 environments, so the August ordering is not reproducible here and cannot
be quoted alongside these figures. The 32-environment certifications are not
directly comparable either: among their survivors the terminal lateral error is
1.69 to 1.85 mm with a standard deviation of 0.054 mm, while the same point at
64 environments runs 0.05 to 2.5 mm. The 2.5 mm gate sits between them, and
whether that is the environment count or the four-times-longer episode is
queued as its own two-run probe.

Every losing arm is retained. No tolerance was widened. The envelope is **not
qualified**.

### The design derivation, as a callable tool

[`servicing_design.py`](../src/zero_g_blade_swap/servicing_design.py) is the
requirement derivation with this workcell's numbers taken out of it. Three
measured quantities go in -- the attitude the manipulator hands over at, the
attitude the interface accepts, and the offset at which a pad still bears on the
capture feature -- and the rack comes out: the two-sided clearance bound, the
channel that maximises the smaller margin, the cross-sections it accepts, and how
accurately the rail has to index. `scripts/derive_rack_requirement.py` is its
CLI and needs no simulator.

For this workcell it derives 10.350 to 11.781 mm of clearance per side, a design
point at 11.065 mm, seven admissible cross-sections of thirty-six, and a **rail
indexing bound of 1.623 mm**. `tests/test_servicing_design.py` asserts all of it
against `check_workcell_geometry.py` over the whole 36-cell grid, so the library
and the certified check cannot drift apart. An arm that hands over at 20 mrad
instead of 46 earns a clearance window 7.3 mm wide instead of 1.4 mm and a rail
it can index to 4.5 mm instead of 1.6 mm; that line is the direction-of-derivation
claim in one sentence.

## Claim limits

- The robot-side latch geometry is visual. Its rigid fixed joint and compliant
  spring-damper are idealized simulation load paths.
- The rack-side pawls are visible geometry without contact colliders. Their
  600 N / 30 N-m `Rack`-to-module fixed joint is an idealized simulation load
  path; its reaction magnitude is not exposed.
- Simulator force probes are diagnostics, not hardware load ratings.
- The robot root is fixed to the world; spacecraft reaction and compliant-base
  tolerance are not modeled.
- The robustness sweep ranks sensitivities but is not a qualified tolerance band.
- Every learned policy comes from one training seed.
- The continuous RGB-D demonstration is one episode at one seed, and the pooled
  camera-driven rate is 16.67%. The demonstration is a favourable sample and is
  labelled as one everywhere it appears.

## Reproduce and continue

```powershell
# CPU trust gate
.\\.venv\\Scripts\\python.exe scripts/check_criterion_currency.py
.\\.venv\\Scripts\\python.exe scripts/check_source_provenance.py --depth 200
.\\.venv\\Scripts\\python.exe scripts/build_evidence_manifest.py --check
.\\.venv\\Scripts\\python.exe scripts/build_script_index.py --check
.\\.venv\\Scripts\\python.exe -m pytest -m "not isaac and not camera and not benchmark"

# Geometry that needs no simulator, including the sight lines
.\\.venv\\Scripts\\python.exe scripts/check_rack_sightlines.py
.\\.venv\\Scripts\\python.exe scripts/check_servicing_camera_geometry.py

# The rack requirement from measured manipulator performance; no simulator
.\\.venv\\Scripts\\python.exe scripts/derive_rack_requirement.py

# Current boundary decision, on the corrected clearance arm; non-zero means not qualified
.\\.venv\\Scripts\\python.exe scripts/validate_serviceability_boundary.py `
  --robustness-sweep evidence/chain_robustness_sweep_n64_channel_v1.json `
  --output <new-versioned-evidence-path>

# The same episodes scored against the failure each criterion predicts
.\\.venv\\Scripts\\python.exe scripts/report_boundary_failure_modes.py `
  --sweep_dir artifacts/robustness64_corrected `
  --compare_dir artifacts/robustness64 `
  --report <new-versioned-evidence-path>

# GPU: strict fixed-cohort chain and one RGB-D chain
bash scripts/run_robot_carried.sh certify
bash scripts/run_robot_carried.sh rgbd

# GPU: does the training-time estimator surrogate reproduce its certificate
C:/isaac-sim/python.bat scripts/check_estimator_surrogate.py `
  --report <new-versioned-evidence-path>

# GPU: the clearance sweep with each bay's mouth moving with its walls
POINTS="rack_lat_6mm rack_lat_16mm" ENVS=64 EPISODES=64 STEPS=6000 `
  SWEEP_EXTRA="--rack_clearance_scope channel" OUT=artifacts/robustness64_channel `
  bash scripts/sweep_chain_robustness.sh
```

Never overwrite evidence; use a new versioned filename.

## Branches

| Branch | Status |
| --- | --- |
| `paper/serviceability-qualification` | active: strict release, insertion handoff audit, current RGB-D gate and boundary validation |
| `main` | baseline at `bccce6d`; unchanged |
| `industrial-relocation` | preserved earlier work; not identical to `main` |
| `keyed-interface` | preserved losing keyed-interface exploration; do not delete |
| `origin/agent/zero-g-blade-swap` | preserved historical line; superseded |

Destination transfer is closed with a narrowed claim: 22/22 eligible episodes
hold, while the full chain remains 22/24 and below 95% because two fail before
seating. The flush-tag camera gate now passes; the next gate is the strict RGB-D
chain and a recording. Learned insertion remains a separate interface-transfer
problem; do not spend more GPU until its reset and real-handoff distributions
are the same by construction.
