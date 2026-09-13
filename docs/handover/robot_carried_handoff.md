> **Status superseded, reasoning kept.** The seating *is* closed — the whole
> chain runs end to end and is certified at a pooled rate. Read
> [`final_session_handoff.md`](final_session_handoff.md) for what is true now.
> Everything below remains the correct account of how the faults in section 4
> were found, which is why it is still here.

# Robot-carried relocation: what is proven, what is scripted, what is open

`claude_opus_5_handoff.md` is the task this branch was given. This is the
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

Section 8.4 of `../service_interface_spec.md` already contained the rule, from
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

> **Read section 4a first.** A later session found the cause of the seating
> failure and it is not what the rest of section 4 says. Three of the numbers
> below are retracted there, with the measurements that retract them. The text
> is kept because the reasoning that produced it is worth having, and because
> the sweep it describes turns out to be a correct measurement of the wrong
> thing.

## 4a. The correction: the destination squaring leg has never worked

Every robot-carried report in this branch already contained the answer, in the
one field nobody read: `robot_carried_transit.legs[].residual_orientation_rad`.

| Run | `square_at_destination` | forced by timeout | residual |
| --- | --- | ---: | ---: |
| `evidence/robot_carried_rgbd_seed6070.json` | never met its gate | 1 of 1 | **133.3 mrad** |
| the shipped preset, state task | never met its gate | 1 of 1 | **144.1 mrad** |
| the rack as built, no relief | never met its gate | 1 of 1 | **144.1 mrad** |

The transit's job is four legs: retreat, square at the source bay, cross, square
at the destination bay. The first three meet their gates. The fourth has never
met its gate in any run this branch has recorded. Traced per step, the module's
attitude sits at 144.11 mrad and does not change to five decimal places for
**380 control steps**, with the tool parked inside 150 µm of a fixed point,
until the leg's own timeout ends it. The leg after it — the one that drives the
module 450 mm into the channel — is what does the squaring, from 144 mrad down
to about 19, while it is also pushing.

So the module was never delivered square and then disturbed. It was never
squared at all.

### Why that leg freezes, and why the source leg does not

The two squaring legs are the same code at the same depth, 220 mm apart. The
source one converges in 80 control steps. The destination one does not converge
at all.

`solve_workcell.py` adopted the base at *x* = −0.65 because the deepest pose it
swept — the transit retreat, at the *nominal* clear centre — solved there with
33 mm of margin. The chain does not fly that pose. It derives its retreat from
the module's **measured** front overhang, 232.0 mm against the 225 mm
half-length, plus `TRANSIT_FLARE_CLEARANCE_M`: 17 mm deeper than the certified
pose, and the crossing was observed 27 mm deeper still.

Every pose on the crossing is still reachable at −0.65, to a micrometre and a
microradian. What is not still there is authority. The chain's differential IK
is damped least squares with λ = 0.010, so a commanded twist arrives multiplied
by `J Jᵀ (J Jᵀ + λ²I)⁻¹`, and across the crossing at the executed depth that
falls to **0.72** at the destination bay with the smallest singular value at
0.016 — inside a factor of two of the damping itself. At −0.75 the same profile
never drops below 0.99.

`scripts/check_workcell_geometry.py` computes all of this on the CPU in about a
second, from closed-form UR10e kinematics validated against every configuration
the simulator recorded in `evidence/workcell_reach_solution.json` (0.006 mm and
0.000 mrad of disagreement), and reproduces the published 166.95 mm shortfall at
the old −0.45 base as a control. `evidence/workcell_geometry_check.json`.

### What the clearance sweep was actually measuring

The sweep below reads as a trade: open the channel and the module goes further
in but ends up more crooked, at about 3.5 mrad per millimetre, "because the
channel was what was squaring the module".

A rigid module of length *L* fully inside a channel with *c* of clearance per
side cannot be tilted past 2*c*/*L*. A module pushed in until it wedges
therefore *stops at* 2*c*/*L* — and the slope of that curve is 2/*L*, which is
**4.44 mrad per millimetre on a 450 mm module** and a property of the module's
length and of nothing else. Every one of the eight recorded rows lands within
13% of it:

| Relief per side | Measured stop attitude | 2*c*/*L* | ratio |
| ---: | ---: | ---: | ---: |
| 4 mm | 20.50 mrad | 20.00 | 1.03 |
| 6 mm | 27.93 | 28.89 | 0.97 |
| 8 mm | 35.22 | 37.78 | 0.93 |
| 12 mm | 49.79 | 55.56 | 0.90 |
| 16 mm | 63.55 | 73.33 | 0.87 |

So the sweep is a correct measurement of the module's length. It is not
evidence about the lead-ins, the compliance, the force cap, or the arm — and the
**"the arm delivers 63 to 67 mrad" headline in the rest of section 4 is one of
its rows**, read at 14 to 16 mm of relief where that number is the channel's
own permission rather than the arm's error.

The arm's actual delivered attitude is measured where nothing is touching the
module — the rack as built, where an 18.7 mrad module cannot enter at all and
parks with its front face on the mouth plane at *x* = 0.2250 m. **18.7 mrad**,
3.4 times better than the retracted figure, and still five to eight times more
than the 2.22 mrad vertical and 3.33 mrad lateral that the unmodified channel
admits.

### What the rail moved the failure to

With the rail fitted the crossing and the destination squaring both work — the
module holds 24.7 mrad flat across the crossing against 107 to 157 mrad without
it, and the squaring leg reaches 6.1 mrad against 144.1 — and the *last* leg then
fails somewhere new. It is worth writing down exactly where, because it is a
much better-defined failure than the one it replaced.

The module's leading corner sits at *x* = 0.3727 and stays there to four decimal
places for 900 control steps, against a lead-in flare whose leading plane is at
*x* = 0.371754. **The module is jammed on the destination bay's own lead-in.**
Everything after that — the attitude winding from 6 mrad to 120, the module
drifting 52 mm off the bay centre line, the tool parked 0.44 m from its target —
is the arm pushing on that contact.

Two faults were found and fixed on the way to that sentence, and then the real
one appeared underneath them.

**The crossing leg was judged on an axis the rail does not own.** Without a rail
the arm crosses, so it closes lateral and vertical together and the leg is
judged on both. With one, the carriage owns the lateral axis and only that; the
module's height sits 8 to 12 mm below the staging value because extraction left
it there, so the leg could never meet a gate on both axes and was ended by its
timeout with 16.8 mm of lateral error still open. The flare catches 16.6 mm per
side. Judged on the lateral axis alone the crossing now **meets its gate**, at
3.3 mm.

**The squaring legs wanted more time than they had.** Tightening their gate from
52.36 mrad to the channel's 2.22 gave them work they did not have before:
attitude *and* the 2.5 mm position gate, which do not converge together because
rotating the module about where it stands moves the tool about 50 mm. Four
hundred steps was generous at the old gate and is not at the new one; at 1200
the source leg's position residual falls from 6.31 mm to 2.23 mm.

### What is actually left: 15 milliradians of pitch and a 5 millimetre gap

With both fixed the module is handed to the last leg in good shape — centred to
2 mm on the bay's line, at 0.7192 m against a channel centre of 0.7205, and
**14.8 mrad off square** — and it still does not enter. Traced to the step:

| Step | Module *x* | *y* | *z* | Attitude | Nose corner *x* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2986 | 0.1321 | −0.2219 | 0.7192 | 14.8 mrad | 0.3582 |
| 2990 | 0.1459 | −0.2011 | 0.7000 | 31.0 mrad | 0.3733 |
| 6498 | 0.1423 | −0.1811 | 0.6953 | 69.8 mrad | 0.3723 |

The module drops 24 mm in four control steps and never recovers, and its nose
corner sits on the flare leading plane at 0.371754 for the remaining 3,500.

The arithmetic closes it. At 14.8 mrad of pitch the leading corner dips
225 mm × 0.0148 = **3.3 mm** below the module's centre line, the module is
already sitting 1.3 mm low, and the channel's vertical half-gap at the shipped
preset is 5.1 mm. That leaves about 0.4 mm, and the first contact spends it: the
corner catches the lower lead-in, the contact drives the module down, and it
wedges. On the rack as built the half-gap is 0.5 mm and there is nothing to
spend at all.

So the requirement on the delivery is not the 52.36 mrad the chain used to check
and it is not even the 2.22 mrad the seated channel admits. It is **pitch under
about 5 mrad and vertical centring under a millimetre, at the mouth**, and the
controller's floor is 15.

**And widening the channel cannot reach this one.** Run at 10.00 mm of per-side
relief against the shipped 4.61 mm, the result is byte-identical — the same
module centre at *x* = 0.1399, the same 103.47 mrad, the same residual on all
four legs. `service_destination_channel_relief_m` moves the guides, the floor
and the lips; it deliberately does not move the ramps and the flares, and the
ramps and the flares are what the module is caught on. Every earlier row of the
clearance sweep was measuring a module that had already got past the lead-in.
This one has not.

Three things this rules out, and each was a live suspect:

* **Not the carriage.** `robot_base_y_m` in the transit trace reaches −0.2360.
  The base was commanded and it moved.
* **Not reach.** Walked continuously from the frozen configuration to the leg's
  own target, the closed-form solver converges at every point to 0.1 µm with the
  head-on attitude held exactly, and no joint passes 2.78 rad.
* **Not conditioning.** Realised authority along that same path runs 0.984 to
  0.999 and the smallest singular value 0.079 to 0.342, against λ = 0.010.

**Not the mating compliance either, though it makes things worse.** Softening
the lock at the leg boundary throws the module from 14.8 mrad to 160 mrad in
four steps; run rigid the same transient is 14.8 to 31. The compliance is
costing about 130 mrad of the transient and is not the reason the module cannot
enter, because the rigid run jams in the same place.

### The floor is a limit cycle, and a smaller gain makes it worse

`_drive_tool_to` commands `rotation_error / scale` clamped to the authority, so
a squaring leg commands a full 8 mrad every step against a 2.22 mrad target, and
the differential IK is in relative mode: it re-anchors on the tool's current
pose each control step and drives to current + delta across the decimation, so
while the joints lag the deltas accumulate ahead of the arm. The leg does not
converge, it limit-cycles at about one action scale — 8.9, 16.1, 9.9 mrad on
successive samples.

The obvious remedy is a smaller step, and it was measured: at a quarter
authority the same leg **diverges**, 0.15 to 2.29 rad over four samples, the
module tumbling end for end. A squaring leg is not a pure rotation — rotating
the module about where it stands moves the tool about 50 mm — so rate-limiting
the rotation hands the solver a translation it can satisfy instead, and the
wrist winds. The default is back at full authority and
`RIGID_TRANSIT_SQUARE_AUTHORITY` is kept so the sweep is one variable away.

**Closing 15 mrad to 5 therefore needs a different controller, not a different
gain.** The candidate this codebase has not tried is to stop commanding clamped
relative deltas on the scripted legs and command joint targets from a solved
inverse kinematics instead — which is what a real robot controller does, which
`GraspSettlingDifferentialInverseKinematicsAction.set_joint_target_override`
already supports, and for which `scripts/check_workcell_geometry.py` is a
validated solver.

Three things this rules out, and each was a live suspect:

* **Not the carriage.** `robot_base_y_m` in the transit trace reaches −0.2313.
  The base was commanded 231 mm and moved 231 mm.
* **Not reach.** Walked continuously from the frozen configuration to the leg's
  own target, the closed-form solver converges at every point to 0.1 µm with the
  head-on attitude held exactly, and no joint passes 2.78 rad.
* **Not conditioning.** Realised authority along that same path runs 0.984 to
  0.999 and the smallest singular value 0.079 to 0.342, against λ = 0.010.

### The requirement that was missing

The chain gated its squaring legs on `INSERTION_ORIENTATION_TOLERANCE_RAD`,
52.36 mrad. The channel admits 2.22. That is a factor of **24**, and it is why
every failing run reported `"orientation": true` on a module too crooked to
enter. The gate is now the channel's acceptance, derived by
`scripts/check_workcell_geometry.py`, with the leg's existing timeout as the
escape so a leg that cannot reach it reports the residual instead of hanging.

---


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
