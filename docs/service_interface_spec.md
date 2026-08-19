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
| Alternative, if the base cannot move back | **≥ 220 mm lateral offset** of every serviced bay from the base's own plane | The boundary is a region around the base's own axis, not a plane at a depth. All 7 failing cells of 64 lie within 110 mm of the centre line; all 40 at 220 mm or beyond succeed, down to the deepest pose the task contains |

The two requirements are alternatives because they are two ways out of one
region. Reaching back toward the base while pointing away from it is unavailable
only when the target is also nearly *on* the base's axis — so a cell escapes
either by putting the poses further forward than the region is deep, or further
to the side than it is wide.

**Design against the depth, not the width.** The depth is derived and moves one
millimetre per millimetre with the base. The width is bracketed between 110 mm
and 220 mm by the candidates that were swept, and a bracket is a worse thing to
build to than a derivation.

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
- **It is not yet achievable with the camera this project owns.** The authored
  64x64 camera resolves 4 mm as **0.13 pixels**, which is an absent signal rather
  than a hard regression problem. `docs/perception_plan.md` derives the fix — a
  narrower field of view rather than more pixels, 180 mm focal length putting
  4 mm at 1.3 px — and requires a rendered frame before anything is trained on
  it.

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

## 9. What this specification does not cover

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
