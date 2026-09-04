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

**The learned seating phase was never given contact force, and the first run
that has it is not behaving like its predecessors.** There is no force channel in
`InsertPolicyObsCfg` and the grapple-pin scene had no contact sensor, so ten
checkpoints of contact-rich assembly were trained without the signal the task is
about. `BladeContactWrenchObservation` has existed in this repository since the
force-limited insertion work, calls itself "the missing half of force control",
cites FORGE, and was never wired to the skill the chain runs.

`Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0` is one change -- the same task,
the same reward, the same actions, seven added observation values -- and its
training reward against the blind policy's, on the identical reward function:

| checkpoint | epoch | mean reward |
| --- | ---: | ---: |
| `v24rack`, blind, plateaued | 1,900 / 2,000 / 2,100 | 43.3 / 41.7 / 43.9 |
| `v33force`, with contact | 100 | 45.6 |
| `v33force`, with contact | 200 | 93.3 |
| `v33force`, with contact | 400 / 500 / 600 / 700 / 800 | 94.4 / 99.2 / 98.0 / 98.0 / **98.2** |

It passes the blind policy's plateau at epoch 100, doubles it by 200, and has
now held near 98 for four hundred epochs. The blind policy needed 2,100 epochs
to reach 43.9 and never left it. This is a converged plateau at more than twice
the height, reached in a quarter of the epochs, on an identical reward.

**This is a training reward and not a rate, and it must not be quoted as one.**
Reward is not success and no episode has been certified; a policy can collect
reward in ways that never seat a module. `verify_insert_skill.sh` is queued on both halves -- the skill on three
held-out seeds and the same weights inside the chain against the scripted
advance, which is the arm that decides. Until that runs, the honest claim is that
the first seating policy able to feel contact is learning much faster than every
one that could not.

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

**Which channel does it, measured on an unchanged checkpoint -- and the answer is
neither.** The certified `v18pin` weights, one curriculum stage, three held-out
evaluation seeds, four arms differing only in which observation channels read the
training-time surrogate:
[`extract_channel_attribution_v1.json`](../evidence/extract_channel_attribution_v1.json)

| what the policy sees | episodes | success | Wilson 95% | points below the control |
| --- | ---: | ---: | --- | ---: |
| exact state, the control | 1,536 | **90.62%** | [89.1, 92.0] | 0 |
| pose channels noised, velocity exact | 1,536 | **82.29%** | [80.3, 84.1] | 8.33 |
| velocity noised, pose channels exact | 1,537 | **80.42%** | [78.4, 82.3] | 10.21 |
| both | 1,536 | **49.48%** | [47.0, 52.0] | 41.15 |

**The interaction is 22.61 points, larger than the sum of the two parts.** That
refutes the standing hypothesis, which was that the velocity channel is the
culprit because the camera runs at half the control rate and a differenced
staircase manufactures a noise floor twenty times the seated signal. It does
manufacture it, and on its own it costs ten points.

What costs forty is having **no reliable channel left**. A policy can absorb a
noisy pose while its velocity is true, because the velocity still tells it
whether the module is actually moving; it can absorb a noisy velocity while its
pose is true, because the pose still tells it where the module actually is. With
both corrupted it has neither, and that is not a channel defect that a filter or
a longer differencing window could repair. It is why the fix had to be the
training distribution, and why training on *both* channels at once is the fix
rather than an approximation to it.

It also settles a worry about the surrogate. Forty-one points on an unchanged
checkpoint is not a mild stand-in: the surrogate reproduces a large part of what
the cameras actually cost, using only the residual, the sample-and-hold and the
miss rate inverted from a published certificate.

**And a second, independent fix needs no training at all: the guard was using a
noise bound to answer an admissibility question.**
`FIDUCIAL_GUARDED_ORIENTATION_TOLERANCE_RAD` is 15 mrad and its own comment
derives it from "above the certified RGB-D p95 errors" -- a bound on whether the
estimate is *trustworthy*, used to decide whether the module may *enter the bay*.
Those are different questions, and the entry flare catches 73.9 mrad.

`--fiducial_guard_bounds lead_in` runs the same guard on the bay's own catch. On
the published checkpoints, three held-out seeds, eight environments, nothing
retrained:
[`workflow_robot_carried_vision_leadin_guard_v1_certification.json`](../evidence/workflow_robot_carried_vision_leadin_guard_v1_certification.json)

| arm | successes | rate | Wilson 95% |
| --- | ---: | ---: | --- |
| shipped estimator bounds | 4/24 | 16.67% | [6.7, 35.9] |
| entry flare's catch | **12/24** | **50.00%** | [31.4, 68.6] |

Per seed 3, 4 and 5 of 8. **One criterion change, correctly derived, is worth 33
points and reaches the 50% gate on its own** -- and it is a change of criterion
rather than a widened tolerance: the detection interlock is unchanged in both
arms and a missing datum still fails closed.

**First evidence that putting the estimator's error into training works, and it
is a probe rather than a rate.** One seed, eight environments, and a *mid-training*
checkpoint -- `grapple_extract_l0_seed70_v19noised` at epoch 13,400, eight hundred
epochs into a two-thousand-epoch fine-tune -- substituted for the extraction
policy and nothing else:

| arm, seed 4070, 8 environments | successes | what the failures did |
| --- | ---: | --- |
| published extraction (`v18pin`) | **2/8** | three never captured, three held at insertion, module frequently lost outright |
| noised fine-tune at epoch 13,400 | **5/8** | all eight reach the final phase; three time out in extraction; none loses the module |

Every environment's estimator error sits between 2.04 and 2.31 mm -- the tight,
uniform band a working estimator produces -- against the published cohort's
excursions to 154 mm on failing environments. **The module-loss failure mode is
gone**, which is the one that made the camera-driven chain look broken rather
than costly.

Treat it as direction, not as a number: n = 8, one seed, an unfinished
checkpoint, and no Wilson interval worth quoting. The pooled cohort over three
held-out seeds with the finished checkpoint is queued, and it is published beside
the 4/24 whichever way it lands.

**Those pooled arms ran, and a screening bug threw all three away.** Each arm's
first seed completed on 2026-09-03 and wrote both its report and its episode
metrics. The queue then screened each report with `grep -q '"error"'`, which
tests for the *presence* of the key. A successful report carries `"error":
null`, so the screen matched every one of them, declared the arm broken after
one seed, and skipped the other two. The npz files were on disk throughout.
Corrected in `queue_rgbd_cohorts.sh` to test the value; the two missing seeds
per arm are running in `queue_rgbd_seeds.sh`.

What the recovered first seeds say, at 8 environments each:

| arm, seed 4070 | successes | changes from the published cohort |
| --- | ---: | --- |
| noised extract alone | 0/8 | retrained extraction, camera velocity |
| kinematic velocity alone | 3/8 | no retrain at all; velocity from the encoders |
| both, plus the lead-in guard | **7/8** | all three, each measured alone elsewhere |

**One seed each. None of this is a rate and none of it clears anything yet**
-- the gate is the pooled figure over three held-out seeds, which is what the
running queue produces. But the ordering is the one the channel attribution
predicted: restoring a single observation channel does little on its own (0/8
and 3/8 against the published 2/8 at this seed), and restoring both together
with a guard that admits on the flare's catch is where the chain comes back.
The interaction was measured at 22.6 points and it is behaving like a real one.

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
module does not carry its hand-over attitude to the seated plane. Right axis, wrong
criterion: the base offset, where the bound is static -- a pad sliding off a pin
-- and the failure is dynamic. The modules are extracted and still gripped and
fail a *settling* condition instead. Two axes remain scope limits with no simulated arm at all -- the capture
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
| Module section | **half confirmed at n=192**, three held-out seeds. The *grip* criterion is right: 120x16 is inadmissible by 3.92 mm and loses 40 of 192 episodes before delivery, 20.8%, Wilson [15.7, 27.1] against nominal's 2.6% [1.1, 6.0] -- eight times the rate, decisively separated, in exactly the mode the criterion names. The *entry* criterion is not: 140x26 is inadmissible by 0.74 mm and jams zero times |
| Robot base offset | mismatch, and the criterion is the wrong one rather than the number. The pad bound is 1.624 mm and the cliff is between 4 and 6 mm; every failing episode is extracted, still gripped at a normal offset, and fails the settling condition instead, carrying 16 to 30 mm/s against a derived 14.29 mm/s limit. The loss is entirely in extraction and capture is untouched at every rung |
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

**The correcting lead-in was tested by deleting it, and it is the flare.** Same
seed, same checkpoints, one flag -- `--remove_entry_flares` takes each bay's two
lateral entry flares out of the scene:

| arm | success | jammed in the bay | missed the terminal gate |
| --- | ---: | ---: | ---: |
| 6 mm per side, flares fitted | 56.25% | **0.0%** | 43.8% |
| 6 mm per side, flares removed | 31.25% | **53.1%** | 15.6% |
| nominal clearance, flares fitted | 54.69% | 0.0% | 42.2% |
| nominal clearance, flares removed | 59.38% | **0.0%** | 35.9% |

This is now `evidence/boundary_lead_in_deletion_v1.json`, written from a clean
commit through the same failure-mode partition every other boundary point uses.
One caveat travels with it: **the reports written before 2026-09-03 do not
record `--remove_entry_flares`**, so for these two arms the identity rests on the
committed invocation in `artifacts/campaign/queue_flare_removal.sh` rather than
on the report. Every run from now on carries a `geometry_arm` block naming the
flags that define a boundary arm, so a future reader never has to trust a
directory name.

**At 6 mm the flare converts 53% jams into none; at nominal clearance removing it
changes nothing.** So the flare is what squares the module during the stroke, and
it is load-bearing exactly when the channel is tight -- which is the question
`servicing_design.requires_a_correcting_lead_in` exists to answer, tested by
removing the part and watching the predicted failure appear.

The rule is directionally right and conservative at the easy end. It fires at
both clearances: at 11.065 mm the module's admissible engagement falls 48 mm
short of the 529 mm stroke, and at 6 mm it falls 268 mm short. The measurement
says the 48 mm shortfall needs no flare and the 268 mm shortfall needs one
badly. Two points are not a calibration curve, but the jam rate tracks the
predicted shortfall and the mechanism is the predicted one.

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

**The rail's stopping error, as a ladder, and the closed form is about the wrong
failure.** The published sweep measured one point at +10 mm. Five points now
exist, all at 64 environments on seed 4070:

| rail stop error | success | Wilson 95% | lost before delivery | grip error of the lost episodes |
| ---: | ---: | --- | ---: | ---: |
| 0 mm | 54.69% | [42.6, 66.3] | 3.1% | -- (2 episodes) |
| 2 mm | 67.19% | [55.0, 77.4] | 3.1% | -- (2 episodes) |
| 4 mm | 51.56% | [39.6, 63.4] | 10.9% | 12.47 mm |
| 6 mm | 23.44% | [14.7, 35.1] | 62.5% | 12.74 mm |
| 10 mm | 1.56% | [0.3, 8.3] | 95.3% | 12.47 mm |

Flat within noise to 4 mm, then a cliff between 4 and 6 mm. The derived
*geometric* bound is 1.624 mm, so a designer taking it would be safe by a factor
of about three -- but **that is not the interesting part, and reading it as
conservatism would be wrong.**

The bound is a *grip* bound: it says a stop error pushes a module in the channel
corner past the offset at which a pad still bears on the pin. If that were the
mechanism, the episodes lost before delivery would show an elevated tool-to-pin
offset, the way `section_120x16` and `rack_lat_16mm` do. They do not. At 4, 6 and
10 mm the lost episodes sit at 12.47, 12.74 and 12.47 mm against successful
episodes' 13.0 -- the same grip, or a slightly better one. **The module is being
held correctly and the phase is timing out anyway.**

**And the phase it happens in is one phase, which makes the correction
specific.** Counting where each failing episode timed out, over the same five
points:

| rail stop error | capture | extract | every other phase |
| ---: | ---: | ---: | ---: |
| 0 mm | 2 | 0 | 0 |
| 2 mm | 1 | 1 | 0 |
| 4 mm | 0 | 7 | 0 |
| 6 mm | 0 | 40 | 0 |
| 10 mm | 1 | 60 | 0 |

Capture is untouched at every offset. **The rail's indexing requirement is set by
the pull, not by the grasp** -- and not by the pad-bearing bound the closed form
uses for it, which is a bound on the grasp. The pads hold the module the whole
way: the tool-to-pin offset on the lost episodes is normal at every rung. What
fails is getting the module out of the channel once the arm is parked off the
bay's centre line.

**And the failing episodes are extracted.** They finish at a module centre of
0.2214 to 0.2227 m against an extracted plane at 0.225 m, still gripped, with a
normal tool-to-pin offset. The extraction predicate is not "past the line": it
also requires the module to be **settled**, at 14.29 mm/s and 142.86 mrad/s, both
derived from the capture tolerances over the 0.70 s settling window. The failing
episodes carry 16.4 to 29.7 mm/s at the median -- one to two times the linear
limit -- while their angular rates stay inside theirs.

So the mechanism is dynamic and the bound is static. The module comes out
carrying residual motion that zero gravity never removes, and the pads never
slide off -- which is what the 1.624 mm bound is about. The module simply never
stops moving.

**The direction of that motion is not recorded and the reading below is
therefore a hypothesis, not a measurement.** A pull whose line is off the bay's
centre applies a moment to a module held by two flat pads on a pin, which is the
one thing this interface cannot resist and is this project's founding
measurement. That would produce exactly this signature. Confirming it needs the
velocity vector rather than its magnitude, which the episode rows do not carry;
`--handoff_trace` does, and one traced rung would settle it.

The closed form therefore has the right axis and the wrong *criterion*, and the
correction is nameable: an axis whose failure is a settling condition needs a
bound on the residual velocity an off-axis pull imparts, which is a dynamic
quantity the present static tool does not compute. What a designer should take
from this repository today is **"index to better than 4 mm, and the binding
constraint is extraction settling, not the grasp"**. Whether a policy trained
across base positions could null that residual is untested and is the obvious
next arm.

**Raising the section axis to three seeds separated it, as predicted.** At 64
episodes `section_120x16` moved in the direction the grip criterion names and did
not clear nominal's interval. At 192 it does, and by a wide margin:

| point | episodes | success | lost before delivery | Wilson 95% on the loss |
| --- | ---: | ---: | ---: | --- |
| nominal | 192 | 57.29% | 5 (2.6%) | [1.1, 6.0] |
| 120 x 16 mm, grip-inadmissible by 3.92 mm | 192 | 45.83% | **40 (20.8%)** | **[15.7, 27.1]** |
| 140 x 26 mm, entry-inadmissible by 0.74 mm | 192 | 40.62% | 17 (8.9%) | [5.6, 13.7] |

Eight times nominal's rate, in the mode the criterion predicts, with the
intervals nowhere near each other. **The grip criterion is confirmed.** The entry
criterion is not: `140x26` is the point that fails it, and it jams zero times out
of 192, so its elevated delivery loss is not the failure that criterion names.

Both are now evidence rather than prose:
`evidence/chain_robustness_sweep_section_n192_v1.json` carries the pooled points
and `evidence/boundary_failure_modes_n192_v1.json` the mode partition, generated
from a clean commit by `scripts/pool_sweep_points.py` out of archives that
already existed. **The pooled rates still overlap nominal** -- 45.83%
[38.9, 52.9] against 57.29% [50.2, 64.1] -- so tripling the sample did not make
the *rate* separate this axis, and the separation is entirely in the mode. That
is the strongest statement of the method note this project has: a criterion
scored against the pooled rate would still be recorded as a mismatch here, on
192 episodes, while the failure it actually predicts runs eight times over.

**The grip signature does not survive decomposition, and the rate does.**
The signature -- failing episodes' tool-to-pin offset minus successful ones' --
reads +0.71 mm at the violated point against +0.03 at nominal, and it was
offered as independent evidence of mechanism. Splitting the 40 losses by where
they stopped shows what it is made of:

| `section_120x16`, n=192 | episodes | median tool-to-pin offset |
| --- | ---: | ---: |
| timed out **in capture**, never gripped | 33 | 66.86 mm |
| gripped, then lost it | 7 | 13.11 mm |
| succeeded | 88 | 13.13 mm |

The episodes that achieved a grip and then lost it sit at 13.11 mm against the
successes' 13.13 mm -- the same number. **The whole signature comes from the 33
episodes that never captured**, whose "grip error" is the distance the tool
happened to be from the pin when the clock ran out. It restates that an
unsuccessful capture ends far from the pin. It is not independent evidence and
must not be quoted as mechanism.

**What the decomposition gives back is stronger than what it takes.** The
criterion says a module in the corner of the source channel has to stay inside
the offset at which a pad keeps half its face on the pin; violate it and the
robot should struggle to *achieve* a grip at all. It does, and that is a clean
rate:

| point | timed out in capture | of 192 |
| --- | ---: | ---: |
| nominal | 3 | 1.6% |
| `120x16`, grip-inadmissible by 3.92 mm | **33** | **17.2%** |
| `140x26`, grip-admissible | 1 | 0.5% |

Eleven times nominal, at the one point the criterion calls inadmissible, in the
phase the criterion is about. The rate evidence carries the claim on its own and
the signature column should be read as a description of the failure, not as a
second measurement of it.

## The transfer rule is still a story, and the attempt to measure it failed

**Retracted the same night it was written, 2026-09-03.** The idea was to turn
the transfer rule -- a bound transfers when nothing in the process corrects the
quantity it bounds -- into a statistic: take the quantity a criterion bounds *as
the episode hands it over*, and measure how well it ranks the episodes that later
fail in the mode that criterion predicts. The numbers looked decisive. They were
measuring something else.

**The archives do not contain a hand-over value.** `_freeze` in
`run_workflow_demo.py` stores an episode's row *at the moment of judgement*, and
says so in its own docstring: a completed workflow idles for the rest of the
episode, so the state at the timeout is not the state that was achieved. Every
recorded grip error, attitude and velocity therefore describes the state the
outcome was decided in. Ranking those against the outcome is a concurrent
association, not a prediction, and no archive in this repository can support the
claim that was made.

**For velocity it is worse than that: it is circular.** `SEATED_CONDITIONS`
includes `linear_velocity` and `angular_velocity`, so an episode fails partly
*because* its velocity is high. The scan duly returned an AUC of 1.000 with a
bootstrap interval of [1.000, 1.000] for angular velocity on the rail axis. A
perfect separation on 40 events is not a discovery; it is the success predicate
being read back.

`evidence/criterion_retention_v1.json` is **retracted**. What survives is what
the same episodes already supported: the grip signature in
`evidence/boundary_failure_modes_n192_v1.json` (+0.71 mm at the violated point
against +0.03 at nominal), which is the same concurrent association stated
honestly, and `evidence/boundary_lead_in_deletion_v1.json`, which is rate-based
and untouched by any of this.

**The measurement is still worth having and now has a price.** To ask whether a
deviation at hand-over governs the outcome, the episode row has to carry the
hand-over value as its own column -- module attitude and lateral offset at the
moment transit ends -- alongside the judgement-time one. That is a change to
`run_workflow_demo.py` and a re-run of the boundary arms, not an analysis of
archives that exist. Until then the transfer rule stays what it was: a
description of which criteria happened to work, not a prediction of which will.

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
be quoted alongside these figures.

**The sweep's 54.69% nominal is not a second, disagreeing measurement of the
91.67% chain, and reading it as one was an error made and corrected on
2026-09-03.** The sweep does not enable rack retention -- its reports record
`destination_rack_retention.enabled: false` -- so the arm it should be compared
against is the paired no-rack control, not the headline:

| cohort | rack retention | rate | Wilson 95% | survivor terminal lateral |
| --- | --- | ---: | --- | ---: |
| strict chain, 8 envs x 3 seeds | yes | 91.67% | [74.2, 97.7] | 0.706 mm, sd 0.53 |
| paired no-rack control, 8 envs x 3 seeds | no | 70.83% | [50.8, 85.1] | 1.728 mm, sd 0.98 |
| sweep nominal, 64 envs, 1 seed | no | 54.69% | [42.6, 66.3] | 2.093 mm, sd 1.25 |
| legacy supported-settle, 32 envs x 3 seeds | n/a | 97.92% | [92.7, 99.4] | 1.832 mm, sd 0.06 |

The sweep's interval and the no-rack control's overlap over more than fourteen
points, so they are consistent, and the terminal precision runs monotonically in
the same order as the rates. The legacy cohort's standard deviation of 0.06 mm is
not an environment-count artefact either: that criterion measures the module
**while it is still supported**, which is why it is twenty times tighter than any
arm that lets go first.

What remains untested is the residual between 70.83% and 54.69% -- one seed
against three, and the sweep's four-times-longer episode -- and the two-run probe
for that is queued. It is now a small question rather than a threat to every
figure in the repository.

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

[`rack_requirement_sweep_v1.json`](../evidence/rack_requirement_sweep_v1.json)
sweeps that derivation and is the figure the claim is made of. One measured
number about the arm moves; the rack follows:

| hand-over attitude | clearance window per side | window width | rail bound | sections of 36 | correcting lead-in |
| ---: | --- | ---: | ---: | ---: | --- |
| 5 mrad | 1.125 to 11.781 mm | 10.656 mm | 6.236 mm | 17 | no |
| 20 mrad | 4.500 to 11.781 mm | 7.281 mm | 4.548 mm | 15 | no |
| 35 mrad | 7.875 to 11.781 mm | 3.906 mm | 2.861 mm | 8 | no |
| 40 mrad | 9.000 to 11.781 mm | 2.781 mm | 2.298 mm | 7 | **yes** |
| **46 mrad, as built** | 10.350 to 11.781 mm | 1.431 mm | 1.623 mm | 7 | yes |
| 52 mrad | 11.700 to 11.781 mm | 0.081 mm | 0.948 mm | 2 | yes |

The four quantities do not move together, and that is what makes it a
derivation rather than a scaling. Only the window's *lower* bound moves, so the
window widens; the rail bound loosens because a tighter channel leaves the pads
more of their reach; and **between 35 and 40 mrad the correcting lead-in stops
being required at all** -- a part deleted from the rack by an improvement in the
arm. At 52 mrad the window closes to 0.081 mm, which is the manipulator handing
over at the interface's own acceptance limit and the rack having nothing left to
give.

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
