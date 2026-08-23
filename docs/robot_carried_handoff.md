# Robot-carried relocation: what is proven, what is scripted, what is open

`docs/claude_opus_5_handoff.md` is the task this branch was given. This is the
answer to it, and it is the file to read first.

## The question, and the answer

**Can the six-axis robot itself carry the compute module from bay 0 to bay 1?**

Not on the finger pads. Measured on 16 environments with nothing changed but the
interface, the passive parallel-jaw grip loses the module entirely: **0 of 16**
retained the tool-to-module transform the flight was planned from, at a median
drift of **808 mm** and **3.14 rad** — π, a module that has turned end-for-end.
The tool travels 168 mm while the module travels 913 mm. Retention is lost at
the median at control step 303, about ten seconds in, and every one of the 16
workflows times out inside the transit.

Yes on a **robot-side form lock**: the same flight holds the transform to
**2.3 mm** and **6.2 mrad** on a single environment and **2.6 mm** across 32,
and the module travels with the tool rather than away from it. Eleven of those
32 held the transform inside the hand-off tolerance for the whole flight, which
is the honest multi-environment number and is not 32 of 32; the rating sweep
below is why the other 21 are a rating question rather than a mechanism one.

The seating in the destination bay is **not closed**, and section 4 is the
measured reason. It is worth saying plainly what changed in the course of
measuring it: **four faults in the chain were producing results that read as
physics**, three of the conclusions this project had already drawn from them
were wrong, and section 4 retracts them.

## What is learned, scripted, perceived, and constrained

| Phase | What drives it | State |
| --- | --- | --- |
| capture | trained policy, certified checkpoint, unchanged | works |
| seat | scripted pause | works |
| extract | trained policy, certified checkpoint, unchanged | works |
| transit | scripted: five module-space waypoints on the form lock | works, 2.3 mm / 6.2 mrad; 11 of 32 inside tolerance throughout |
| insert | scripted guarded advance on the deployed estimator | **not closed** |
| done | settled re-check, then the hand opens | not reached |

Perception is the calibrated RGB-D fiducial estimator and the occupancy planner,
both unchanged and both fail-closed. Simulator truth is used for scoring and for
one geometric interlock that protects the rack from the mechanism; it is never a
policy observation.

## 1. The lock is on the robot, not the module

Section 8.4 of `docs/service_interface_spec.md` already contained the rule, from
a sweep of the gripped section against the measured hand: a serviceable module
cannot carry a positive axial stop forward of the pads, so the axial lock has to
come from the end-effector. Two module-side attempts had been built and refuted
before it was written. This is the first design on the correct side of it, and
it costs the module nothing — the pin is unchanged, so every certification taken
against it still describes the part that is built.

Section 9 of the specification is the design.
`evidence/service_latch_clearance.json` derives every clearance from the measured
gripper envelope with no simulator; the tightest are +3.0 mm and +5.9 mm. The
derivation rejected two of its own dimensions before it passed.

The carriage **seeks** the collar rather than assuming it, because a tapered
wedge does not seat a module at one depth. Measured travel used: 12.0 mm, which
is the self-seating distance section 4 of the specification predicts, arrived at
from the other direction.

## 2. The transit is planned in module space

A pad-held module moves in the grip, so the old follower servos the tool and
corrects the module afterwards. A form-locked module is a rigid extension of the
wrist, and that structure does not merely waste effort on it — it diverges,
because the alignment sub-phase computes a tool target from the module's
position and rotating the tool moves the module. Measured: 0.66 rad walked to
1.61 rad while the module was dragged 380 mm backwards.

The replacement is five module poses and one servo, with the position command
inverted through the attitude the tool *has* rather than the one it is being
asked for. Without that inversion a standing attitude error becomes a standing
position error of the same size times the 340 mm offset — measured twice, as a
lateral leg parking 61 mm and then 64 mm short.

Five legs rather than three because of the workcell, not the payload: section 6a
measures a region around the arm's own axis where position and the head-on
attitude cannot both be held, and bay 0 sits on that axis. A leg that asks the
arm to cross *and* square there gets a compromise — 0.164 rad, which on a rigid
payload is also 56 mm of position error it cannot correct. Squaring is therefore
a leg of its own, done where the arm can afford it.

## 3. The rack needed a lead-in on an axis nobody had asked about

Section 6 of the specification measures the lateral flares as load-bearing:
removed, two fully trained policies insert nothing. That result is about the
lateral axis. Nothing had ever asked about the vertical one, because **both
insertion skills reset with the module already inside the channel** and so never
entered the mouth from outside.

A relocation enters from outside. Section 6.1 is the requirement that follows,
and section 6.2 is the one after it: a module delivered *rigidly* cannot be
straightened by the channel it is entering, so the channel has to admit the
attitude the manipulator delivers — clearance ≥ L·θ/2. Section 4 below adds the
half of that requirement this session measured: it is bounded **above** as well,
because opening the channel is also what stops it correcting the module.

## 4. What is open, and why it is hard

The seating does not close. The robot carries the module to the destination bay,
drives it 362 mm into the channel, and stops 163 mm short of seated.

**Four faults were found and fixed getting to that sentence, and three of them
were producing measurements that read as physics.** They are listed first
because the conclusions this project had already written down from them were
wrong.

| Fault | What it was | What it made the earlier grid say |
| --- | --- | --- |
| The guarded advance was anchored to the module | the axial target was rebuilt each step as `module_x + clamp(target - module_x, ±10 mm)`, so a module that does not move holds the target 10 mm in front of itself forever | every stiffness, force cap and clearance was applied at one standing 10 mm command error -- which is why "ten times the force moves it 0.1 mm" |
| The vertical ramps did not move with the channel relief | authored from the nominal lip and floor surfaces | opening the channel opened it behind a throat that stayed at 0.5 mm -- which is why "15 mm per side buys 16.5 mm of 163" |
| The lateral flares did not either | section 6 places each so its inner face meets the rail face at the mouth | the same, on the other axis |
| The last transit leg was asking for both its jobs at once | full attitude authority on a leg that pushes 450 mm straight into a lead-in | the solver takes the rotation and drops the advance |

Measured directly: **900 advancing steps, no guard holds, and a commanded depth
that moved 11 mm.**

**Only the first of the four fixes survived measurement, and the three that did
not are results in their own right.** Moving the lead-ins out with the relief is
the tidy rule -- a lead-in continues a channel surface, so it should move when
the surface does -- and it makes the entry *worse*: the module stops dead on the
mouth plane at 0.2249 m, because the lead-ins at the nominal surfaces are what
square a module the arm delivers 67 mrad off. Giving the last transit leg its
attitude authority back is the correct reasoning about a damped least-squares
solver and it also makes things worse: the module starts moving and then
decelerates into the lead-in, 0.1736 m to 0.1890 m over 240 steps.

All three reverts are recorded where they were made, with the number that
reverted them, because a rule that is right in general and wrong here is worth
more written down than deleted.

### What the corrected chain measures

| Compliance centre | Guarded steps advancing | Steps stalled at full stroke | Module advanced, of 163 mm |
| --- | ---: | ---: | ---: |
| At the module's leading face | 900 | 875 | **0.3 mm** |
| At the wrist | 900 | 875 | **0.7 mm** |

The middle column is the one that could not be read before. The guard never
fires -- the deployed estimator says the module is inside the bay's catch on all
900 steps -- and the advance spends 875 of them holding a commanded depth a
**full mating stroke** in front of a module that will not follow. The compliance
is at its hard stop, the interface is rigid again, and the module still does not
move.

So it is not the guard, not the force cap, not the clearance, and not the
compliance centre.

### The clearance sweep, which is what settles it

Every row below is the corrected chain -- deadlock fixed, bay assembled
consistently, guard never firing -- with nothing changed but the destination
bay's per-side relief and the push available to the mating compliance.

| Channel relief, per side | Push available | Module advanced, of 163 mm | Attitude it stopped at |
| ---: | ---: | ---: | ---: |
| 4 mm | 1 kN | 0.7 mm | 20.5 mrad |
| 4 mm | 4 kN | 0.8 mm | 20.4 mrad |
| 6 mm | 1 kN | 5.0 mm | 27.9 mrad |
| 8 mm | 4 kN | 10.1 mm | 35.2 mrad |
| 10 mm | 4 kN | 12.2 mm | 42.6 mrad |
| 12 mm | 4 kN | 14.6 mm | 49.8 mrad |
| 14.2 mm | 4 kN | 16.5 mm | **57.5 mrad** |
| 16 mm | 4 kN | 20.6 mm | **63.5 mrad** |

Report: `evidence/robot_carried_seating_sweep.json`.

Two monotone curves, and they are the whole answer.
The last row closes it. At 16 mm per side the module settles at 63.5 mrad, which
is the attitude the arm delivers with nothing touching it at all: the channel has
stopped correcting the module entirely, and it has bought 20.6 mm of the 163.


**The advance grows about 1.2 mm for every extra millimetre of clearance.**
Extrapolated, closing the remaining 163 mm needs something near 140 mm of relief
per side on a 36 mm channel, which is not a channel.

**The attitude it settles at grows about 3.5 mrad per millimetre**, because the
channel is what was squaring the module and every millimetre of relief is a
millimetre it no longer does that in. At 12 mm the module is at 49.8 mrad
against the 52.4 mrad the seating check allows; by 14.2 mm it is at 57.5 mrad and
**outside it**.

So the two requirements cross at about 12.5 mm per side, and at the crossing the
module has travelled 15 mm of the 163 it needs -- an order of magnitude short.
There is no channel width for this workcell. That is not a tuning result and it
is not a controller result: it is the geometry of a 450 mm rigid part delivered
63 mrad off square into a 36 mm channel, and it is the finding this branch was
built to produce.

Also settled, in the same table: **push is not a lever anywhere on it.** At 4 mm
of relief, four times the force moves the module 0.1 mm further. The retracted
grid said so for the wrong reason; the corrected one says so for the right one.

### The blocker, in one number

The module arrives at the mouth 47 to 67 mrad off square -- the arm's own
accuracy inside the reach boundary of section 6a -- and **that error is not
about one axis.** As an axis-angle at the terminal pose: 0.1 mrad of roll,
**13.8 mrad of pitch, 15.1 mrad of yaw.**

A 450 mm module tilted in two planes at once has to be walked square by two
lead-ins simultaneously, in a channel whose vertical and lateral clearances were
designed one axis at a time. Every configuration that lets one lead-in do its
work takes authority away from the other, which is what the four fixed faults
were hiding and what the two reverted ones demonstrate directly.

There is one more measurement that closes the argument. Open the channel to
20 mm per side -- with the ramps and flares moved to match, so the bay is
consistent -- and the module never reaches the hand-off at all: the last leg
parks 53 mm short with the tool 62.7 mrad off square and the module 62.8 mrad
off with it, the lock holding the two to 0.2 mm throughout. **The channel was
doing the squaring.** Open it far enough that it stops touching the module and
the manipulator's own error appears in full, past the 52.4 mrad the seating
check allows. A wider bay does not buy a crooked seat; it buys no seat.

Which makes the rack requirement two-sided, and section 6.2 now states it that
way:

> Channel clearance per side ≥ *L* · θ_entry / 2 **and** ≤ *L* · θ_seated / 2,
> with θ_entry the attitude the manipulator delivers unaided and θ_seated the
> attitude the seating check allows. Below the first the module cannot enter;
> above the second the channel stops correcting it.

On this workcell, for a *straight* channel, those two bounds are 14.2 mm and
11.8 mm per side and **they do not overlap**: no constant channel width admits a
63 mrad delivery and still ends inside a 52.4 mrad seating check.

What closes that gap is the lead-in, and this is where its role becomes exact.
The bay runs at 4 mm per side -- under both bounds -- and the module does enter,
362 mm of it, because the ramps and flares walk it from the 63 mrad it arrives at
down to the 20.5 mrad it is holding when it stops. The lead-ins are not admitting
the tilt, they are *removing* it, which is section 6's finding again. So the
requirement a designer actually has to meet is not on the channel at all:

> The lead-ins must remove θ_entry − θ_seated of attitude over the length of
> module that has entered before the channel starts constraining it.

Here that is 63 mrad down to under 52.4, on two axes at once, in the 80 mm of
plate each lead-in has. They get it to 20.5 mrad and the module still does not
seat -- so the remaining 163 mm is being refused by something the lead-ins have
already done all they can about.

### The redesign, in order of preference

1. **Move the arm out of the reach boundary.** Section 6a already measures a
   region around the arm's own axis where position and the head-on attitude
   cannot both be held, and the destination bay sits in it. This is the only one
   of these that fixes the cause rather than the symptom, and the sweep says why:
   at 4 mm of relief the module settles at 20.5 mrad and stops after 0.7 mm, so
   the channel is already correcting 42 mrad of the arm's 63 and the residue is
   what jams. Deliver the module at 20 mrad instead of 63 and the channel has
   nothing left to fight. It costs a re-certification of capture and extraction,
   because it moves the workcell those policies were trained in.
2. **Chamfer the module's leading edges.** A chamfer turns a two-plane tilt into
   a self-aligning entry without asking either lead-in for authority the other
   needs. It changes the module, which section 3 has so far kept unchanged, so
   it costs every certification taken against that geometry.
3. **Give the mating compliance a second rotational axis with its own centre**,
   which is what a real remote-centre device does and what
   ``MATING_COMPLIANCE_CENTRE`` currently approximates with one.

Do not read the 0.3 mm and 0.7 mm rows as a tuning gap. They are what an arm
pushing a rigid link into a hole it is not aligned with looks like, and the
alignment is upstream of everything the mating interface can do.

## 5. The RGB-D end-to-end run

One full chain on the vision task, calibrated RGB-D fiducial perception driving
every phase, `visual_randomization: "on"`:

| | |
| --- | ---: |
| Report | `evidence/robot_carried_rgbd_seed6070.json` |
| Phases reached | capture, seat, extract, transit, insert, done |
| Tool-to-module drift across the flight, p50 | **2.95 mm / 7.9 mrad** |
| Module centre reached | 0.5887 m of the 0.750 m seated pose |
| Seating conditions met | lateral, orientation, linear velocity, angular velocity, grasp position, grasp orientation |
| Seating condition **not** met | axial depth |

Read the last two rows together. The module arrives in the destination bay
inside every part of the seating envelope except how deep it is, which is the
same blocker section 4 measures, reached through the deployed estimator rather
than through simulator truth.

A second run of the same seed with the lighting held stable produces the video
(`artifacts/robotcarried/video/`, and
`evidence/robot_carried_rgbd_video_seed6070.json` for its report). The two are
separate runs because the recorder needs a fixed exposure and the evidence needs
the randomization the perception was certified under; the stage takes both as
switches so neither is quietly standing in for the other.

**One honest caveat about seeds.** Seed 4070, which the state-space runs use,
does not get through capture on the vision task: the tool ends 137 mrad off the
pin axis and the extraction never moves the module. That is the vision task's
own capture variance and it is not something this session's changes caused, but
it means the RGB-D chain is demonstrated on one seed rather than certified
across several.

## Where the work is

| Purpose | Path |
| --- | --- |
| The latch's dimensional contract | `src/zero_g_blade_swap/service_latch.py` |
| Its clearance derivation | `scripts/check_service_latch_clearance.py`, `evidence/service_latch_clearance.json` |
| The workflow driver | `scripts/run_workflow_demo.py` |
| Reproducing every stage | `scripts/run_robot_carried.sh` |
| The live compute-service preset | `src/zero_g_blade_swap/service/presets.py` |
| Contracts that keep the shuttle out | `tests/test_robot_carried_contract.py` |
| The latch geometry's contract | `tests/test_service_latch_geometry.py` |

## What was retained, not superseded

`evidence/full_chain_state_16_report.json` and
`evidence/full_chain_rgbd_service_seed4070.json` describe the payload-stage
baseline: a chain in which a hidden world-mounted D6 stage took the module off
the arm after extraction and moved it independently. They are good measurements
of that mechanism and they are **not** robot-carried results. The mechanism is
still reachable behind `--base_rail_on_relocation` so those numbers can be
reproduced, and it is kept out of the live preset by a test rather than by
anyone's memory.
