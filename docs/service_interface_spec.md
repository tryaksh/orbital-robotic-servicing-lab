# Service interface specification

**What a modular compute unit must present to be robotically serviceable by a
Robotiq 2F-85-class parallel gripper on a 6-axis arm.**

This is the design output of this project. Every number below is derived from a
measurement in `evidence/`, not chosen. The intent is that a module designer can
read this without reading the simulation.

Everything here is simulation evidence. No number on this page was produced on
real hardware, and the limitations at the end are part of the specification.

---

## 1. Why a specification exists at all

The bottleneck in robotic servicing of modular hardware is not the controller.
It is that modules are not designed to be grabbed.

Measured on this workcell: a parallel-jaw gripper closing on a smooth raised
post holds **about 6 N** against extraction, where the insertion task's own
worst-case contact reaction demands **66.4 N**. That is a factor of eleven, and
it is structural rather than a tuning failure:

- The gripper closes along one axis and the module leaves along another.
- The rails must leave the extraction axis free, or the module could not be
  extracted at all.
- Flat pads on a smooth feature can then oppose that axis only by friction.

Sweeping closure from 0.62 to 0.77 rad made it *worse*, because the pads shove
the post along the free axis and then close on air. No gripper tuning fixes
this. The interface has to move onto the module, and the approach has to align
with the pull axis. This is the reasoning behind grapple fixtures on serviceable
spacecraft hardware, arrived at here from measurement.

Report: `evidence/grasp_axial_pull_gate.json`.

---

## 2. The gripper this is specified against

Measured from collision geometry in the `wrist_3_link` frame by
`scripts/measure_gripper_envelope.py`. **Do not substitute values read from body
origins**: every 2F-85 body in this asset is collapsed to within 18 mm of the
flange, and reading them as pad locations produced a claim this project had to
retract.

| Quantity | Measured |
| --- | ---: |
| Clear opening, drive command 0 rad | 87.08 mm |
| Clear opening, drive command 0.8203 rad (joint limit) | 0.0 mm |
| Closing rate | 106.23 mm/rad |
| Pad span from flange, along the approach axis | 105 to 162 mm |
| Pad length / width | 57 mm / 27 mm |
| Palm face from flange | 90 mm |
| Gripper envelope about the tool axis | 155 mm closing, 75 mm across |

Report: `evidence/gripper_collision_envelope.json`.

**Zero is fully open.** The opening falls monotonically with the command across
the joint's whole range, and 87 mm at the open end matches the hardware's
published 85 mm stroke.

### 2.1 The constraint that decides the interface shape

Slicing the same point cloud by depth shows the inner knuckles sweeping the
**15 mm gap between the palm face (90 mm) and the pad trailing faces (105 mm)**,
reaching within 8 to 24 mm of the tool axis depending on closure.

This rules out the obvious design. A mushroom head with a flat shoulder must sit
in that gap to bear on the pads, must be *wider* than the pad gap to catch them,
and must be *narrower* than the knuckles to fit. **No closure satisfies both.**

A specification that assumed a mushroom head would be unbuildable on this
gripper. The interface below is a tapered wedge clamped inside the pad aperture,
which is the same principle a tapered grapple pin uses.

---

## 3. Required module geometry

Dimensions are along the extraction axis, measured from the module's leading
face outward. Constants live in `src/zero_g_blade_swap/grapple_geometry.py` and
are enforced by `tests/test_grapple_geometry.py`, which runs without a
simulator.

```
        free end                                            module face
   |<-- 60 mm wedge -->|<- 6 ->|<------- 80 mm shaft ------>|
   
   70 mm ============______                                  ^
         tapering     ------====[collar]====================  30 mm
   16 mm ============------                                  v
                       90 mm tall
```

| Feature | Requirement | Why |
| --- | --- | --- |
| **Wedge**, free end | 70 mm across the closing axis | Must pass inside the 87.08 mm aperture. Leaves 8.5 mm of approach clearance a side. |
| **Wedge**, module end | 16 mm across | Sets the taper. |
| **Wedge** length | 60 mm | Must exceed the 57 mm pad length, so the pads seat on the sloped faces rather than overhanging the rim. |
| **Wedge** taper | 24.2 degrees, thickening toward the free end | Axial capacity scales with its sine. Pulling drags thicker material into the pads and forces them apart against the drive. |
| **Collar** height | 90 mm across the closing axis | Must exceed 87.08 mm so a fully open gripper can never pass it. This makes it an absolute depth stop. |
| **Collar** thickness | 6 mm | Also the face the insert direction pushes on, so both directions have a hard stop. |
| **Shaft** length | 80 mm | Set by the rack, see 3.1. |
| **Shaft** cross-section | 30 x 30 mm | Must clear the slot; must not foul the pads. |
| **Width**, all features | 30 mm | Must equal or exceed the 27 mm pad width so the whole pad bears. |
| **Total protrusion** | 146 mm | The sum. |

### 3.1 The shaft length is set by the rack, not the gripper

The gripper must never enter the rack. Its envelope reaches 77 mm from the tool
axis, which fouls the slot floor plate; a module inserted with the gripper
inside the slot cannot be released without collision.

With the module's leading face 75 mm inside the slot mouth when fully inserted,
and the pads 57 mm long, the collar the pads seat against cannot be closer than
80 mm to the module face. **That is the whole derivation.** A shallower rack
allows a shorter pin; a deeper one demands a longer one.

Consequence to accept at design time: the interface protrudes about 62 mm beyond
the rack face when the module is installed.

---

## 4. Required actuation sequence

**Capture and hold are two different commands.** This is the least obvious
requirement here and the one that decides whether the interface works.

A wedge converts closing force into thrust along the pull axis, which is the one
axis the rack must leave free. Closing firmly therefore drives the module away
before it has been taken. Holding, once seated, wants the opposite.

| Phase | Command | Clear opening |
| --- | ---: | ---: |
| Approach | 0.02 rad | 84.9 mm |
| Capture | 0.48 rad | 36.1 mm |
| Hold | 0.68 rad | 14.8 mm |

Measured axial capacity against a single command throughout: **59 N**. With the
capture and hold split: **69 N**.

The capture window is narrow and asymmetric, so bias it low:

| Capture command | Axial capacity |
| ---: | ---: |
| 0.44 rad | 63 N |
| **0.48 rad** | **69 N** |
| 0.52 rad | 68 N |
| 0.56 rad | 26 N |

Reports: `evidence/grapple_pin_axial_pull_gate.json`,
`evidence/grapple_pin_capture_plateau.json`.

**The capture self-seats.** Closing drives the pin along the taper until the
collar catches the pad leading faces, moving the module about 12.5 mm every
time. Budget for that motion; do not treat it as slip.

---

## 5. Verified performance

| Property | Value | Source |
| --- | ---: | --- |
| Axial holding capacity, within 2 mm of slip | 69 N | 3 closures x 121 forces, 1 N resolution |
| Required capacity | 66.4 N | Worst-case contact reaction of the promoted insertion policy |
| Grip formed | 363 / 363 environments | Drive torque at the 10 N-m limit, seated |
| Approach misalignment tolerated | 8.5 mm per side | Geometric, wedge free end inside the aperture |

**Grip force is not the lever.** Raising the modelled drive from 10 N-m to the
24.96 N-m that produces Robotiq's rated 235 N measured *worse* on a matched
grid, 62 N against 66 N, and lost capture entirely above 0.65 rad. Same
mechanism as section 4: a harder squeeze drives an unconstrained payload. Do not
specify a stronger gripper to fix a capture problem.

---

## 6. Required rack-side geometry: the lead-in flare is load-bearing

Everything above specifies the *module*. This section specifies the **rack**, and
it is here because a measurement forced it, not because the scope grew.

The rack's slot mouth carries two 80 mm entry plates, each rotated 12 degrees
outward, which widen the lateral catch from the channel's own 0.75 mm per side to
**16.6 mm per side**. They were built as an aid. They are not an aid.

Removing the flares and changing nothing else, evaluated on two fully trained
insertion policies:

| Slot lateral displacement | Force-aware policy | Force-blind policy |
| ---: | ---: | ---: |
| 0 mm | **0.00%** | **0.00%** |
| 4 mm | 0.00% | 0.00% |
| 8 mm | 0.17% | 0.00% |

**Both policies fail every episode, including at zero displacement, where there
is no uncertainty in the task at all.** The flares were not helping with the
offset; they were performing the insertion. Neither policy ever learned to align
a module into a 0.75 mm-per-side channel, because with a 16.6 mm-per-side catch
in front of it neither ever had to.

| Feature | Requirement | Why |
| --- | --- | --- |
| **Lead-in flare**, lateral catch | 16.6 mm per side | Below this, no policy measured here inserts at all. |
| **Lead-in flare**, plate length | 80 mm | With a 12-degree flare, this is what produces the 16.6 mm catch. |
| **Lead-in flare**, half-angle | 12 degrees | 0.2126 m of opening per metre of length. Shallower needs a longer mouth. |
| **Lead-in flare**, surface | Lowest friction in the slot | A lead-in must guide, not grab. |

Raw rows: `artifacts/noleadin/`. No pooled report is published for this probe:
the runs predate the reporting change that records whether the lead-in is
present, and a 0% result filed as a certification would read as a failed policy
rather than as a removed mechanism.

Two caveats, because this number is strong enough to be misread. Both policies
trained *with* the flares, so this measures dependence on the lead-in, not the
impossibility of learning without one; a policy trained without flares might
learn to align, and none has been. And this is the rack half of the same finding
section 1 records for the module half — there, a parallel-jaw grip could not hold
the module and the fix was geometry on the module; here, a policy cannot align
the module and the fix is geometry on the rack. **Design-for-serviceability is
doing work that control cannot substitute for, measured twice, from opposite
directions.**

---

### 6.1 The other axis, and why nothing had ever asked about it

Everything above is about **y**. The rack also has a **z**, and until a robot
carried a module between bays nothing in this project had ever entered the mouth
from outside it: both insertion skills *reset* with the module already inside its
channel, so the only motion they ever performed was along a channel that was
already holding them.

A relocation enters from outside. And a six-axis arm carrying a 450 mm module
through free space delivers it to the mouth off square -- measured at the
destination hand-off, **0.063 to 0.067 rad**, which swings the module's
leading corner **14 to 15 mm** off the channel's centre plane. The channel
leaves 0.5 mm per side between the floor plate and the upper lips. The module
cannot enter, and no controller gain fixes a geometric interference. Measured,
it fails in one of two places depending on how the bay's lead-ins are assembled:
with the module's nose on the mouth plane at 0.225 m, or 362 mm inside the
channel at 0.588 m against the 0.750 m it needs. Both times the arm pushes
against it until the episode ends.

| Feature | Requirement | Why |
| --- | --- | --- |
| **Entry ramp**, vertical catch | 16.6 mm per side | Accepts 0.074 rad of delivered attitude error on a 450 mm module, against the 0.066 rad an arm was measured to deliver. |
| **Entry ramp**, plate length | 80 mm | The lateral flare's, unchanged. |
| **Entry ramp**, half-angle | 12 degrees | The lateral flare's, unchanged. |
| **Entry ramp**, width | 60 mm | Narrower than the module, deliberately: the latch carriage follows the module to the mouth, and at 160 mm the ramps and the stowed carriage occupy the same volume. Sixty millimetres catches the middle of the leading edge and leaves the carriage 16.5 mm outside the ramps, checked by `scripts/check_service_latch_clearance.py`. |
| **Entry ramp**, surface | Lowest friction in the slot | A lead-in must guide, not grab. The flare's own values. |

### 6.2 A rigidly delivered module needs a channel that admits its attitude

The lead-in above gets the module's nose into the mouth. It does not get the
module *in*, and the reason is a property of how a lead-in works: it pushes a
module square. A module rigidly held by a robot will not be pushed.

That is not hypothetical either: section 9.6 measures it a dozen ways, and every
lock state that carries the module refuses to let the channel align it.

So a rack that accepts a robot-delivered module has to admit the attitude that
robot delivers it at. A straight channel of clearance *c* admits a rigid module
of length *L* at a tilt of at most 2*c*/*L*, which gives the requirement
directly:

> **Requirement.** Channel clearance per side ≥ *L* · θ / 2, where θ is the
> manipulator's delivered attitude accuracy at the mouth.

| Quantity | Value |
| --- | ---: |
| Module length *L* | 450 mm |
| Delivered attitude θ, measured at the destination hand-off | 0.063 to 0.067 rad |
| Required clearance per side | **14.2 mm** |
| The unmodified channel's clearance per side | 0.5 mm vertical, 0.75 mm lateral |

A designer has three ways to satisfy it, and widening the channel is the last of
them: hold the module compliantly for the final approach — which this workcell
cannot, because the pads do not resist lateral load; stand the arm outside the
reach boundary of section 6a, which is what sets the delivered attitude in the
first place; or open the channel. This
implementation opens the channel, on the destination bay only, and reports the
attitude each run actually delivered so a run outside the envelope fails its
seating check rather than being quietly accommodated.

---

This is section 6's finding a second time, on the axis it did not look at, and
it was found the same way: by removing the assumption that something else was
holding the module. There, a policy could not align into a 0.75 mm channel
because the flares had always done it. Here, a controller could not enter one
because the *rack* had always been holding the module before insertion began.

Fitted to the destination bay only, and only on the robot-carried path
(`configure_service_destination()`), because the source bay is never entered from
outside and adding geometry to it would change a scene four certifications
describe. Dimensions in `src/zero_g_blade_swap/tasks/blade_swap/assets.py`.

---

## 6a. Required workcell geometry: where the arm stands is an interface requirement

Sections 3 and 6 specify the module and the rack. This one specifies **where the
arm is**, and it is here for the same reason section 6 is: a measurement forced
it. A servicing interface that a manipulator cannot approach in the required
attitude is not a serviceable interface, and whether it can is decided by the
base position rather than by anything on the module.

### The requirement

| Feature | Requirement | Why |
| --- | ---: | --- |
| Base standoff behind the deepest required tool pose | **≥ 0.4242 m** | Closer than this the arm can reach the pose but cannot hold the approach attitude there |
| Adopted base for this rack | **x = −0.65 m**, 0.4572 m behind the transit retreat | 33 mm of margin on the derived threshold of −0.617 m |
| Alternative, if the base cannot move back | **≥ 200 mm lateral offset** of every serviced bay from the base's own plane (measured half-width 155–167 mm, plus margin) | The boundary is a region around the base's own axis, not a plane at a depth. All 7 failing cells of 64 lie within 110 mm of the centre line; all 40 at 220 mm or beyond succeed, down to the deepest pose the task contains |

The two requirements are alternatives because they are two ways out of one
region. Reaching back toward the base while pointing away from it is unavailable
only when the target is also nearly *on* the base's axis — so a cell escapes
either by putting the poses further forward than the region is deep, or further
to the side than it is wide.

Both are now derived rather than bracketed. The depth moves one millimetre per
millimetre with the base. The width is **167 mm at the transit retreat and about
152 mm at the extraction end** — a cone about the base's axis that widens
slightly with depth — and the retreat's shortfall falls one millimetre per
millimetre of lateral offset until it reaches zero
(`evidence/attitude_wall_lateral_profile.json`). 200 mm is quoted above as the
requirement rather than 167 mm because a workcell tolerance should not be
specified at the measured edge.

### What the number is

Driving position and the head-on capture attitude together at full authority, the
tool parks at a fixed distance in front of the base and goes no further whatever
depth is asked of it. The solver satisfies the attitude to 0.0002 rad and
surrenders the position entirely. That distance is **0.4242 m**, and it moves
with the base one millimetre per millimetre:

| Base x | Retreat shortfall | Extraction shortfall |
| ---: | ---: | ---: |
| −0.45 | 166.95 mm | 88.70 mm |
| −0.55 | 66.95 mm | solved |
| −0.65 | solved | solved |

`evidence/workcell_reach_solution.json`, `evidence/relocation_reach_boundary.json`.

It is a trade rather than a hard limit, and the exchange rate is the useful
number: near this folded configuration attitude buys reach at about **7.5 metres
per radian**, so 0.0114 rad of surrendered attitude buys 85 mm. That is why a
marginal cell does not fail loudly. It fails by giving the attitude away — and on
a parallel-jaw grip that cannot resist a moment about its closing axis, giving
the attitude away is exactly what the payload cannot survive. The module carried
between bays on the old cell swung **end-for-end** about its grip while the grip
error still read a healthy 24 mm.

### Why this belongs in an interface specification

The obvious reading of sections 3 and 8 is that attitude is a *grip* problem, to
be solved with geometry on the module. Three attempts to do that are refuted in
section 8. This section is the other half of the same finding: **some of the
attitude error attributed to the interface was the arm buying depth with it.**
Extraction on the old cell had to finish 88.7 mm past the boundary and ended with
about 0.13 rad of grip attitude error, which this project read as the interface
failing for three sessions.

How much of that 0.13 rad the workcell owns is a separate measurement and is not
claimed here from the kinematics alone. The honest form is the one the sweep
supports: the old cell *forced* the trade at the pose extraction has to finish
in, so a grip-attitude number taken there cannot be attributed to the grip
without a control. Extraction re-certified on the moved cell is that control.

So the specification's requirement on attitude is not only "constrain it
mechanically". It is:

1. put the base far enough back, or the bays far enough off its plane, that the
   arm can hold the approach attitude at every pose the operation needs — a
   workcell requirement, free, and checkable by kinematics before anything is
   built;
2. and only then ask what the interface has to hold, because until (1) is
   satisfied the interface is being blamed for the arm's trade.

The cost of getting (1) wrong is not a warning. It is a clean, monotone,
entirely convincing reach boundary in a report.

---

## 7. Pose accuracy this interface requires

The flare dimension in section 6 is not only a rack requirement. It sets the
pose tolerance the whole system has to hold, and therefore the accuracy any
future pose estimator has to deliver.

Measured by displacing the slot laterally by a known amount the policy is told
nothing about, on two independently trained policies:

| Slot lateral displacement | Force-aware | Force-blind |
| ---: | ---: | ---: |
| 0 mm | 100.00% | 99.94% |
| 2 mm | 100.00% | 99.83% |
| **4 mm** | **99.87%** | **99.77%** |
| 6 mm | 96.94% | 99.56% |
| 8 mm | 87.50% | 94.90% |
| 10 mm | 74.07% | 82.31% |

Reports: `evidence/uncertain_insertion_aware_certification.json`,
`evidence/uncertain_insertion_blind_certification.json`, and the two
`*_envelope.json` files, which are flagged `simulation_capability_envelope`
because they sweep past the trained ceiling.

> **Requirement.** Module and rack pose must be known to **4 mm laterally** at
> the moment of insertion. Inside that, insertion succeeds essentially always
> and the mechanical lead-in absorbs the residual. Outside it, success falls off
> monotonically and no amount of contact sensing recovers it.

Three things this requirement is not:

- **It is not the geometric catch.** The flares catch 16.6 mm per side. 4 mm is
  where policies were *trained and certified*; 6 to 10 mm is an envelope sweep
  outside the training distribution. The mechanism may support more than 4 mm;
  no policy has been trained there, so the specification claims 4 mm.
- **It is not a case for force sensing.** Force feedback was the hypothesis for
  extending this tolerance and it was refuted: beyond 4 mm the force-aware policy
  is *worse* than its force-blind control, by up to 8.2 points, at roughly twice
  the peak contact force. Under a position-controlled action space a policy has
  no action that yields to a force it can read.
- **The original camera could not meet it; the replacement can.** The former
  64x64 view resolved 4 mm as only 0.13 pixels. The current 384x384 RGB-D
  fiducial system measures 1.682 mm position-error p95 and 99.854% detection at
  critical rack poses over 1,024 rendered frames. See
  `evidence/fiducial_rgbd_service_plate.json`.

---

## 8. Attitude must be constrained — and three attempts to do it mechanically failed

> **Read 8.3 before designing anything against this section.** Two features were
> built on the reasoning below, both were trained against and certified, and
> both are measured as net negatives. The rotation this section calls "yaw" was
> decomposed on 2026-08-15 and is not principally about the closing axis at all.
> The section is kept in its original form because the reasoning it records is
> what the refutation is a refutation *of*.

A single-point tapered pin clamped by flat pads does not constrain rotation about
the pin axis. While the module is in its rails this does not matter. Once the
rails release it, it does. This was filed as an interface *result* rather than as
a controller bug, on the argument that no controller change fixes it. That
argument is now known to be wrong; see 8.3.

| Measurement | Value |
| --- | ---: |
| Module orientation error in failing extractions | **0.93 rad (53 degrees)** |
| Grip *position* error in the same episodes | 13.5 mm, the normal seated value |
| Grip error on the return leg of a round trip, start to end | 15 mm to 35 mm |
| Effect of replaying that return leg 4x slower | **worse** |

The grip is not slipping; the module is levering. Failures concentrate at the
curriculum stage that starts with the module furthest out, which is the stage
that spends longest unconstrained. That slowing the return fourfold makes the
degradation worse rather than better is the load-bearing detail: this is rotation
under sustained load, not an acceleration artefact, so it cannot be flown around.

**Consequence.** The remove-and-replace round trip does not close on this
interface. Removal works and installation works; carrying a module between them
does not. An anti-yaw feature — a keyway, or flats the pads bear against
laterally — is the change that closes it.

### 8.1 The feature, and where the gripper allows it

> **The yoke's code was deleted on 2026-08-18.** The dimensions below are kept
> because they are correctly derived from the measured envelope and because 8.3
> is about them; they do not describe anything the repository builds. A contract
> test keeps the feature out.

Re-reading `evidence/gripper_collision_envelope.json` across the whole closure
range gives the constraint that decides the design:

| Quantity | Measured |
| --- | ---: |
| Deepest reach from the flange of any body that is not an inner finger | 0.1245 m |
| Deepest reach of an inner finger | 0.1621 m |
| Widest half-extent of a non-finger body on the third axis | 17.5 mm |
| Inner finger half-width | 13.5 mm |

There is therefore a **37.6 mm band immediately behind the collar containing only
fingers**, and a wall narrower than 17.5 mm is safe inside it and nowhere else.
The module already presents 30 mm of pin width against a 27 mm finger, so the
feature costs no width: it is the wedge's side faces raised into a channel the
fingers run between.

| Feature | Requirement | Why |
| --- | --- | --- |
| **Yoke walls**, inner half-gap | 15.0 mm | Flush with the pin flanks. 1.5 mm per side against a 13.5 mm finger |
| **Yoke**, length from the collar face | 34 mm | Keeps the mouth at 0.128 m from the flange, clear of the knuckle band |
| **Yoke**, parallel section | 24 mm | The engagement length that sets the free yaw, 2c/L = 0.125 rad |
| **Yoke**, lead-in flare | 10 mm at 20 degrees to an 18.6 mm half-gap | 5.14 mm of catch per side, so the capture is not asked to hit a 1.5 mm slot blind |
| **Yoke**, wall height | ±45 mm | The collar's own, so the depth stop's envelope is not exceeded |
| Must preserve | 66.4 N axial capacity | The insertion contact reaction does not change because the pin did |
| Must preserve | 8.5 mm per side approach clearance | Or capture stops working, which is a worse trade |

### 8.2 What has been measured about it, and what has not

**The axial hold survives.** On the same 3-closure by 121-force grid the 69 N
result was measured on, the yoked interface holds **67 N** at the 0.48 rad
capture command against the 66.4 N required, and angular slip under axial pull
falls from 0.1481 to 0.1312 rad at p95. Report:
`evidence/grapple_pin_axial_pull_gate_yoked.json`.

**Whether it fixes yaw is not measured, and a static probe cannot measure it.**
Loading the seated interface laterally moves the module 1.2 mm under 200 N and
rotates it 0.079 rad regardless of load, identically with and without the yoke,
because the module is in its rails and the rails constrain it. That is consistent
with what section 6 of this specification says about the rack doing mechanical
work, and it means **yaw is a property of the interface after the rack releases
the module, not of the seated interface.** It has to be measured on a moving
extraction. Reports:
`evidence/grapple_pin_yaw_probe_railed_plain.json` and `_yoked.json`, both
carrying `gate.applies: false`.

### 8.3 Both mechanical fixes were built, measured, and refuted

**The yoke costs far more than it buys.** All three skills fine-tuned onto it
and certified on three held-out seeds each:

| Skill | Plain pin | Yoked pin |
| --- | ---: | ---: |
| Capture | 95.55% | 88.81% |
| Extract | 0.00% | 0.13% |
| Insert | 95.57% | 28.70% |

**And the axis was never measured.** The capture attitude error decomposed into
the gripper's own axes, on extraction with the plain pin:

| Component | Terminal p50 |
| --- | ---: |
| About the **closing** axis — the only axis these walls oppose | 0.198 rad |
| About the **transverse** axis | 0.199 rad |
| About the approach axis | 0.070 rad |

The rotation is split roughly evenly across two axes and every dimension in 8.1
addresses one of them. That is the entire explanation for recovering 12% of it,
and it is the specification's own version of a mistake this project has made
before: designing against a name instead of a measurement.

**A modelled latch fails differently.** Flight servicing hardware does not hold
a module against extraction by friction on a passive feature — the SSRMS
latching end effector snares a grapple fixture and rigidizes it, and Dextre's
tool changeout mechanism grips a standardised fixture and carries a powered
socket drive. Modelled here as a rated restoring torque engaged by a qualifying
capture, and swept against an *unchanged* policy so no difference can be a
training artefact:

| Latch rating | Transverse rotation p50 | Module travel p50 |
| ---: | ---: | ---: |
| none | — | 458 mm |
| 10 N·m | 0.293 rad | 24 mm |
| 80 N·m | 0.299 rad | 29 mm |

An eightfold rating change moves the target by 0.006 rad and destroys the
extraction, because a restoring torque applied while the rails still hold the
module jams it in the rails.

**What the requirement actually is.** Attitude must be held within 0.20 rad
through the pull — that part of this section stands, and four certifications
measure it. What does not stand is the inference that a passive module-side
feature is how to hold it. The measurements now point at the controller and the
objective: the extraction end pose is reachable with the attitude held to
0.0114 rad, and the reward was charging attitude 0.16 per step against a
progress term weighted 12. **A module-side interface requirement should not be
written against this failure until a force- or impedance-controlled action space
has been tried**, because two passive features have now been paid for and both
made the system worse.

---

### 8.4 A third attempt, and the specification-level result it produced

The natural next move after two failed *additions* is to change the gripped
feature itself: replace the taper with a **flat key between two axial stops**, so
rotation is blocked by plane contact rather than by friction, and axial load is
carried by a shoulder rather than by wedging. That is what flight hardware does —
Dextre's OTCM grips a micro-square and then bolts it; SIROM latches at three
points; HOTDOCK is form-fit plus a lock. None of them makes friction
load-bearing.

**It works as a clamp and it cannot be installed.**

| | Tapered pin | Keyed pin |
| --- | ---: | ---: |
| Seated grip offset | 0.0194 m | **0.0007 m** |
| Seated grip attitude | 0.0637 rad | **0.0013 rad** |
| Rotation in extraction failures | 0.30+ rad | **0.10 rad** |
| Lateral load held without slip | — | **120 N at a 0.34 m arm, 40 N·m** |
| Extraction | 99.02% | **0.00%** |

The reason for the 0.00% is dimensional, not behavioural: with the pads seated,
the keyed pin's nose flange occupies 77–97 mm of depth from the flange at a 45 mm
half-height, and **the palm straddles the tool axis out to 90 mm**. The flange
overlaps the hand by 45.0 mm at every closure, including fully open. The pads were
never in the pocket. `evidence/grapple_pin_keyed_interference.json`.

**And this generalises into a specification rule.** Sweeping the gripped section's
half-height from 1 to 43 mm and reading the room in the window a pocket wall must
occupy — between the palm face at 90 mm and the seated pads at 105 mm:

| | |
| --- | ---: |
| Room forward of the seated pads | **7.9 mm** of half-height |
| Half-opening a stop must exceed to be proof against the pads splaying | **43.5 mm** |
| Gripped-section heights admitting *any* forward stop | 7 of 43, all under 7 mm |

> **Rule.** On a stock parallel-jaw gripper, a serviceable module **cannot carry a
> positive axial stop forward of the pads.** The volume immediately ahead of the
> pads belongs to the hand — the palm below 90 mm and the knuckles above it — and
> the trade is self-defeating in the obvious direction too, because a shorter
> gripped section lets the hand close further and brings the knuckles further in.
> An axial lock therefore has to come from the *end-effector*: V-grooved fingers,
> or a powered latch. It cannot come from the module.

**The corollary is the useful half, and it reverses a judgement this project held
for three sessions.** The 2F-85's throat is itself cone-shaped, and the tapered
pin's profile is close to it — 33.6 mm of half-height where the pads begin,
falling to 8 mm at the collar, against a throat that opens the same way. The taper
stops the pads at 0.186 rad and clears every closure at or below that. The taper
doing double duty as funnel *and* clamp was read as a design smell. It is not: it
is the only shape this hand's throat admits, and the two features that tried to
improve on it both failed because they needed volume the hand occupies.

`scripts/check_pin_gripper_clearance.py` and
`scripts/measure_pin_design_window.py` derive both, from
`evidence/gripper_collision_envelope.json` and no simulator. Run the first with no
arguments to check the shipped pin; it passes.

## 9. The lock is on the robot, it has three states, and this is it

Section 8.4 ends with a rule, derived by sweeping the gripped section's
half-height against the measured hand:

> On a stock parallel-jaw gripper, a serviceable module **cannot carry a
> positive axial stop forward of the pads**. […] An axial lock therefore has to
> come from the *end-effector*: V-grooved fingers, or a powered latch. It cannot
> come from the module.

Two module-side features were built and refuted before that rule existed. This
section is the first design on the correct side of the interface, and it is here
because a measurement demanded it rather than because the scope grew.

---

### 9.1 The measurement that demands it

Everything above about holding capacity is **static**: the arm is held still and
a force is applied. The operation this project is for is not static. It carries
a module from one bay to another through free space, and whether a parallel-jaw
grip survives that had never been asked as its own experiment.

Asked, with nothing changed but the interface — one arm carries the module on
the finger pads alone, the other adds the lock. Same policies, same planned
route. The quantity is the **tool-to-module transform**, recorded every second
control step of the transit and compared against the transform the route was
planned from:

| Carried by | Environments retaining the planned transform | Position drift, p50 | Attitude drift, p50 | Tool travel, p50 | Module travel, p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Finger pads alone** | **0 of 16** | **808.0 mm** | **3.139 rad** | 167.9 mm | 913.0 mm |
| **Robot-side form lock** | **11 of 32** | **2.6 mm** | **0.007 rad** | 296.6 mm | 773.1 mm |

Read the travel columns on the passive row before anything else. The tool travels
168 mm and the module travels 913 mm: the module is not being carried, it is
being *released* — and 3.139 rad is π, a module that has turned end-for-end.
Retention is lost at control step 303 at the median, about ten seconds in, and
every one of the 16 workflows then times out in the transit with the module a
median of 677 mm from the tool.

The latched row is pooled over 32 environments and 21 of them also lost the
transform at this rating, so its travel columns average keepers and losers
together and should not be read as a flight profile. Its **drift** median is the
comparison that matters — 2.6 mm against 808 mm — and *11 of 32* is the honest
rate at which the lock held a whole flight. A single demonstration run holds
2.3 mm and 6.2 mrad with the module travelling 433 mm against the tool's 454 mm,
which is what one successful flight looks like.

This is section 8's finding taken on a moving arm, and it is stronger: **a
parallel-jaw grip on this pin does not carry this module between bays at all.**
No controller change addresses it. The controller is doing what it was told and
the module is leaving anyway.

Report: `evidence/robot_carried_interface.json`, arm `passive_parallel_jaw_only`.

---

### 9.2 Where the latch is allowed to be

The same envelope measurement that forbids a module-side stop grants the
end-effector one, and the window is generous:

| Quantity | Measured | Source |
| --- | ---: | --- |
| Deepest point of any gripper body, from the flange | 162.11 mm | `evidence/gripper_collision_envelope.json` |
| Widest half-extent of the hand on the third axis | 37.5 mm | same |
| Half-extent of an inner finger, the only body past 124.5 mm | 13.5 mm | same |
| Collar's proximal shoulder, from the flange, at the seated grip | 168 mm | derived from section 3 |
| Module's own face, from the flange | 248 mm | derived from section 3 |

So there is an **80 mm length of 30 × 30 mm shaft, immediately behind the
collar, that no part of the hand can ever occupy.** That is where the latch
lives, and it is why this design costs the module nothing: the pin in section 3
is unchanged, so every certification taken against it still describes the part
that is built.

---

### 9.3 What it is

A two-jaw powered latch on a carriage, bolted beside the hand. The jaws extend
along the approach axis, then close on the shaft's flanks with lips that drop
behind the collar's proximal shoulder.

```
                        collar          shaft
   pads ====]         |‾‾‾‾‾‾|      
             \        |      |___________________  module
              wedge   |      |
   pads ====]         |______|
                          ^
                     lips drop in here, above and below the shaft,
                     and bear on the shoulder the collar presents
```

| Feature | Requirement | Why |
| --- | --- | --- |
| **Window**, from the flange | 168 to 248 mm | The only volume the hand cannot reach. Everything below fits inside it. |
| **Jaw** length | 22 mm | Short: the load path is the lip on the shoulder, not a long clamp, and a shorter jaw leaves more carriage seek. |
| **Jaw** half-height | 40 mm | Inside the collar's own 45 mm, so the lip is backed by shoulder rather than overhanging its rim. |
| **Web** inner half-gap, closed | 15.5 mm | Half a millimetre per side on the 30 mm shaft. A clamp, not a capture: the pads have already located the module. |
| **Lip** thickness | 2.5 mm | Sits immediately behind the shoulder. |
| **Lip** inner reach | 4 mm from the centre line | 11 mm of shoulder engaged per side. The two opposing lips stop 8 mm apart because nothing guarantees they arrive together. |
| **Lip** band | 18 to 40 mm from the pin axis | Above and below the shaft, which is the only part of the collar's proximal face that is not the shaft's own root. |
| **Lip** bearing area, total | 968 mm² | 0.62 MPa at 600 N. The lip is not the limit; if it were, the rating below would be a fiction. |
| **Close stroke** | 31 mm | Sized by the *fingers*, not the collar: released, the lip has to park clear of a 13.5 mm inner finger, and that is the tighter of the two. |
| **Extend stroke** | 25 mm | Sized by the rack. Stowed, the deepest latch body stops 8 mm short of the slot mouth with the module fully seated. |
| **Carriage seek** | −5 to +40 mm | See 9.5. |
| **Mating compliance stroke** | 25 mm | See 9.6. |

Every clearance is derived from the measured envelope and this table by
`scripts/check_service_latch_clearance.py`, which runs with no simulator and
refuses to pass if any of them closes. The tightest are **+3.0 mm** (the lip
clearing the shaft it sits above) and **+5.9 mm** (the engaged jaw beginning past
the deepest gripper body). Report: `evidence/service_latch_clearance.json`.

The load path in simulation is a **break-rated PhysX fixed joint between
`wrist_3_link` and the module** while rigid, and a bounded spring-damper on the
same pair while compliant. The hardware above is authored on the wrist as visual
geometry with no collider, so the jaws' contact with the pin is *not* simulated
and the joint is not a second, hidden load path beside it — it is the only one,
and this paragraph is the disclosure the rest of the specification is written
against.

---

### 9.4 The rating is not a preference

A PhysX break threshold is permanent: a lock rated below the load it actually
sees is present for the first second of the flight and absent for the rest of
it, while still reporting that it engaged. That is exactly what the first run
did.

| Rated at | What happened |
| --- | --- |
| 600 N / 30 N·m | Broke during the flight. The lock reported engaged and the tool-to-module transform still moved 435 mm at the median across 16 environments, which a fixed joint cannot do unless it is no longer there. A PhysX break threshold is permanent: a lock rated under its load is present for the first second and absent for the rest, while still reporting that it engaged. |
| 20 kN / 1 kN·m | Held. 2.3 mm and 6.2 mrad on a single environment, 2.6 mm across 32. |

The number that matters is not the payload's inertia. **The latch is preloaded
by the gripper it works beside.** The 2F-85's drive saturates at 10 N·m against a
measured 106.2 mm/rad transmission, about 94 N of pad force, and a 24.2-degree
wedge turns that into hundreds of newtons along the pull axis whether or not the
arm is moving. A form lock that holds the module against the hand's own wedge
thrust is rated for that thrust first and for the flight second — which is not
what a reader would guess from section 5, where the same wedge is the thing
providing 69 N of capacity.

---

### 9.5 The carriage seeks the collar; it does not assume it

A tapered wedge does not seat a module at one depth. Where the pin sits along
the approach axis is set by where its thickness equals the pad opening, so it
moves with the closure command — section 4 measures about 12.5 mm of self-seating
travel on every capture, and seated grip offsets between 12 and 19 mm. A latch
authored at the nominal collar depth would close on the collar's rim, or on air.

So the carriage drives along the approach axis until it finds the shoulder, and
**both ends of its travel are derived**:

| End | Value | Set by |
| --- | ---: | --- |
| Near | −5 mm | An engaged jaw may not come back inside the hand: the pad leading edge is at 162 mm and the jaw's near face stops 1 mm past it. |
| Far | +40 mm | An engaged jaw may not reach the module's own face at 248 mm, or it is clamping the chassis instead of the shaft. |

An engagement whose measured shoulder falls outside that travel is **refused**
and the refusal is counted in the report. A mechanism that cannot reach a part
should say so rather than be modelled as though it had. The travel this workflow
actually used, measured: **12.0 mm** — which is the self-seating distance section
4 predicts, arrived at from the other direction.

---

### 9.6 One state is not enough: rigid to carry, compliant to mate

This is the part that was not obvious, and it cost three measured failures to
find.

**Rigid, it carries and cannot mate.** A lead-in aligns a part by pushing it
square, and a part welded to a stiff arm will not be pushed. Section 6 already
established that this rack's lead-in does not assist the insertion, it performs
it. So the mechanism that makes the flight possible makes the mating impossible.

**Released, it mates and cannot carry.** The pads do not resist lateral load,
which section 8 measures and which this section's own control run demonstrates.

Measured, in one session, three ways:

| Lock state during mating | What the seating did |
| --- | --- |
| rigid throughout | module advanced **0.3 mm** in a 30-second budget |
| released at the phase boundary | module advanced **15.6 mm**, then wedged: already inside the channel, already crooked |
| released before the mouth | module slid **laterally out of the bay** |

Those three are sound: they are about which *state* the lock is in, and the
faults in 9.6.1 do not reach them. The stiffness and clearance sweep that
followed them is the part that has to be retracted.

Four faults were found and fixed during this grid, and 9.6.1 records them
because most of the rows above were measuring those rather than the interface.
Below is what the corrected chain does. The configuration is the same
throughout: 40 kN/m translational, 20 kN·m/rad rotational, a 1 kN force cap set
so that the joint reaches its 25 mm stroke before the cap binds, and 4 mm of
channel relief on the destination bay only.

| Compliance centre | Guarded steps advancing | Steps stalled at full stroke | Module advanced, of 163 mm |
| --- | ---: | ---: | ---: |
| At the module's leading face (remote centre) | 900 | 875 | **0.3 mm** |
| At the wrist (plain spring) | 900 | 875 | **0.7 mm** |

Read the middle column first, because it is the one that could not be read
before. The guarded advance is never blocked by its own guard -- the estimator
says the module is inside the bay's catch on every one of the 900 steps -- and
it spends 875 of them holding a commanded depth that is a **full mating stroke**
in front of a module that will not follow. The interface has run out of
compliance and is rigid again, and the module still does not move.

That is the measured blocker, and it is not any of the things it looked like:

* it is not the guard, which never fires;
* it is not the force cap, which the stroke reaches first;
* it is not the channel clearance, which is four times the requirement for the
  attitude the module actually arrives at;
* it is not the compliance centre, which moves the answer by 0.4 mm across the
  whole range from wrist to tip.

**What it is, in one number.** The module arrives at the mouth 47 to 67 mrad off
square -- the arm's own accuracy inside the reach boundary of section 6a -- and
that error is *not about one axis*. Measured as an axis-angle at the terminal
pose: 0.1 mrad of roll, **13.8 mrad of pitch, 15.1 mrad of yaw**. A 450 mm
module tilted in both planes at once has to be walked square by two lead-ins
simultaneously, in a channel whose vertical and lateral clearances were designed
one axis at a time, and every configuration that lets one lead-in do its work
takes authority away from the other.

So the lock has three states, and switching between them is the design:

| State | When | What it is |
| --- | --- | --- |
| **Rigid** | from rail release to the end of the squaring leg | break-rated fixed joint; carries the module through free space |
| **Compliant** | from there until the module is seated | bounded spring-damper with a finite stroke; still the load path, no longer a refusal to yield |
| **Released** | once the seating predicate fires | nothing; the 0.70 s settling re-check is taken on a module held by its own rails and the pads |

This is not an invention. Assembly compliance devices have done exactly this
since the 1970s, and flight servicing hardware does it too: the SSRMS latching
end effector snares a grapple fixture, *rigidizes* it to carry, and gives that
rigidity up to berth.

**Compliant in translation, stiff in rotation, and that split is the design.**
The rack aligns a module by pushing it, so the interface has to yield in
translation or the lead-in cannot work. It must not yield in rotation: a lead-in
cannot straighten a 450 mm module inside a 1 mm channel, and measured at
160 N·m/rad the module rotated 0.309 rad against the compliance and jammed
crooked. An assembly compliance device is specified the same way, with its
lateral and angular stiffnesses chosen separately.

**And it is a joint, not an applied wrench, for a reason worth stating.** The
first implementation applied a spring-damper wrench to the module at the 30 Hz
control rate. That is stable while it is soft and unusable when it is not:
raising the angular gain to hold the module square put the spring's own period
inside the command interval and the module left the cell at 1.5 m in a quarter
of a second — and with no cap at all, at 15 km. A solver-side joint drive is
integrated implicitly and is stable at any gain, which is the only way to be
soft on one axis and stiff on another at the same time.

**Its centre is a design parameter, and this workcell wants both ends of it.**
An assembly compliance device puts the centre at the part's own contact point so
that a contact force there *translates* the part instead of rotating it into the
wall it has just touched, which is the jamming mode of a clearance fit. That is
right for seating a part that arrives square. This one does not: it arrives 47
to 67 mrad off, and the only thing that can take that out is the lead-in
producing exactly the moment a remote centre is specified to cancel. Swept from
the wrist to the leading face with everything else fixed, the whole range moves
the seating by 0.4 mm, so neither end of it is the answer -- but the parameter
is real and it is exposed as ``MATING_COMPLIANCE_CENTRE`` rather than welded
into the code at one end.

**It has a stroke, and running out of it is recorded as what it is** — the
interface losing the module — rather than smoothed away.

| Quantity | Value | Set by |
| --- | ---: | --- |
| Mating stroke | 25 mm | The lead-in has to push the module about 3 mm off the tool's line, and the module lags the tool a few millimetres under the insertion's own contact reaction |
| Mating rotation limit | 3 degrees | A spring rather than a weld, and far inside the 0.052 rad the seating envelope allows |
| Mating translational stiffness | 40 kN/m | Swept. Softer pushes the module outside the lead-in's own catch; stiffer reproduces the weld |
| Mating rotational stiffness | 20 kN·m/rad | Stiff, because attitude is the one thing a lead-in cannot correct |
| Mating force cap | 1 kN | Measured: more is not better. 400 N walks the module in, 1 kN wedges it at a third of its travel |

---

### 9.6.1 Three faults, and a grid that has to be read as their consequence

The table above was measured over one session and most of it is now known to
have been measuring the same three faults rather than the mating interface.
They are recorded here because a swept grid that turned out to be sweeping a bug
is a result about the method, and deleting it would leave the conclusions it
produced standing with nothing under them.

**The guarded advance was anchored to the module it was pushing.** The axial
target was rebuilt every control step as `module_x + clamp(target - module_x,
+/-10 mm)`, which reads as a bounded lead and behaves as a deadlock: a module
that does not move holds the target a fixed ten millimetres in front of itself
forever. Measured directly -- 900 advancing steps, no guard holds, and a
commanded depth that moved 11 mm. Every stiffness, every force cap and every
channel clearance in the grid was therefore applied at one standing command
error of 10 mm, which is why raising the force cap tenfold moved the module
0.1 mm: the cap was never what limited the push.

**The vertical lead-in did not move with the channel relief.** The ramps are
authored from the nominal lip and floor surfaces. Opening the channel moved the
lips, the floor and the guides and left the ramps where they were, so the relief
opened a channel behind a throat that stayed at 0.5 mm per side. That is why
fifteen millimetres per side -- thirty times the production clearance -- bought
16.5 mm of a 163 mm travel, and why the measurement read as a refutation of the
clearance requirement rather than as a bay assembled inconsistently.

**The lateral flares did not move either**, for the same reason and by the same
rule: section 6 places each flare so its inner face meets the rail face exactly
at the mouth, which makes the flare a function of the rail's position.

The retracted rows are kept because two conclusions had already been written down
from them, and re-measuring both on the corrected chain is what a retraction is
for. **"Opening the channel barely helps" was wrong** -- it was measured behind a
lead-in that had not moved with the relief, and opening the bay consistently
changes the outcome completely, though not in the direction that was expected.
**"More force is not the lever" turns out to be right**, and only now for a
reason that can be defended: at 160 kN/m against a 4 kN cap, four times the
push that the retracted grid could actually deliver, the module advances 0.8 mm
instead of 0.7 mm.

**And the last transit leg was asking for both of its jobs at once.** The rigid
transit gives every leg full attitude authority, for a reason that is right about
the legs that cross: the module is the wrist, so a wrist that winds carries the
module round with it, and at quarter authority the crossing leg wound the module
from 0.12 to 2.85 rad. The last leg does not cross -- it pushes the module 450 mm
along the rack axis into a lead-in. Asked for a 363 mm advance and a 67 mrad
correction in the same 6-D command, a damped least-squares solver takes the
rotation and drops the advance, and the module parks with its nose on the mouth
plane while the tool sits 360 mm behind its own target. The residual tilt on that
leg is the rack's to take out, which is what section 6 says a lead-in is for.

---

**The relief is bounded above as well as below, and the upper bound is the
surprise.** Opened to 20 mm per side -- with the ramps and flares moved to match,
so the whole bay is consistent -- the module never reaches its hand-off at all.
The transit's last leg parks 53 mm short and stays there, and the reason is in
the same log line: the tool is 62.7 mrad off square and the module is 62.8 mrad
off with it, the lock holding the two to 0.2 mm and 0.1 mrad throughout.

That is the arm's *own* attitude at that pose, and it is three times the 20.5 mrad
the same arm delivered into a 4 mm channel. The channel was doing the squaring.
Open it far enough that it no longer touches the module and the manipulator's
reach-boundary error appears in full -- past the 52.4 mrad the seating check
allows, so a wider bay does not buy a crooked seat, it buys no seat.

Which makes the requirement two-sided:

> **Requirement.** Channel clearance per side ≥ *L* · θ_entry / 2, and
> ≤ *L* · θ_seated / 2, with θ_entry the attitude the manipulator delivers
> unaided and θ_seated the attitude the seating check allows. Below the first the
> module cannot enter; above the second the channel stops correcting it and the
> manipulator's own error is what seats.

---

### 9.6.2 The clearance sweep, which is what settles it

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

---

### 9.7 It has to be out of the rack before the module is home

The jaws sit behind the collar, and section 3.1 puts the collar 5 mm *outside*
the slot mouth when the module is fully seated. An engaged jaw therefore enters
the rack before the module does. Derived:

| | |
| --- | ---: |
| Module centre at which an engaged jaw reaches the slot mouth, at zero seek | 0.733 m |
| The same at the far end of the carriage seek | 0.693 m |
| Module centre when seated | 0.750 m |

> **Requirement.** The carriage must be stowed before the module centre passes
> that depth, and the stowed carriage must clear the mouth at the seated pose —
> it does, by 8 mm. This is the same rule section 3.1 applies to the gripper,
> applied to the mechanism bolted beside it.

---

## 10. What this specification does not cover

- Simulation only. PhysX Coulomb friction between primitive geometry, not a
  measured pad compound on a machined fixture.
- No connector, latch, chamfer, cable, or measured force-displacement curve.
  The module is a cuboid and the rack is two rails, a floor, and two lips.
- Contact forces here are a relative damage proxy for comparing designs, not an
  absolute force budget for hardware.
- Static holding capacity with the arm held still. Dynamic slip during a moving
  extraction is characterised only indirectly, through skill success rates.
- Lateral pull and prying moment capacity are not measured at all.
- One gripper, one arm, one rack geometry.
