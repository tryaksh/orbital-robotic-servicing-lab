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

## 8. Second-generation requirement: yaw must be constrained

A single-point tapered pin clamped by flat pads does not constrain rotation about
the pin axis. While the module is in its rails this does not matter. Once the
rails release it, it does. This is filed as an interface *result* rather than as
a controller bug, because no controller change fixes it.

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

| Feature | Requirement | Why |
| --- | --- | --- |
| **Anti-yaw feature** | Bearing surfaces that oppose rotation about the pull axis | 0.93 rad of free rotation once the rails release the module |
| Must preserve | 66.4 N axial capacity | The insertion contact reaction does not change because the pin did |
| Must preserve | 8.5 mm per side approach clearance | Or capture stops working, which is a worse trade |

---

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
