# What sim-to-real would take, and the two experiments that would falsify this

Everything in this repository is simulated. Nothing has run on hardware. A
space-robotics reviewer will not accept that silently, and the useful response is
not a disclaimer -- it is a written account of what is modelled, to what standard,
which claims would move first on hardware, and which single physical experiment
falsifies the specification most cheaply.

This file is analysis, not experiments. Every item is a property of this
simulation that can be checked in the source or in `evidence/`.

## What is modelled, and to what standard

| Element | Modelled as | What that does and does not license |
| --- | --- | --- |
| Gravity | `gravity=(0, 0, 0)` throughout | The load-bearing assumption. A free-floating mass does not settle, and closing pads on a taper ejects it before they grip; that is why capture and extraction are learned and the free-space motion is not. It does not model orbital rate, gravity gradient, or a tumbling client. |
| Robot base | Fixed to the world (`fix_root_link=True`) | No spacecraft reaction, no attitude-control coupling, no compliant mount. A free-flying servicer changes the problem qualitatively: momentum is conserved through the arm, and the arm's own motion moves the base. |
| Robot rail | Indexes a base that is already fixed to the world | The carriage's stiffness, backlash and stopping error are not in the load path. The `base_y` ladder is the closest thing to a stopping-error measurement, and it is an *evaluation* sweep of a policy trained at one base position -- see the caveat below, because it is the one place where a simulation number is easy to misread as a geometric bound. |
| Robot-side form lock | Break-rated PhysX fixed joint (rigid) and bounded spring-damper (compliant) between `wrist_3_link` and the module | Disclosed in every report. The geometry is authored and its clearances are checked; the *load path* is idealised. No pad-on-pin contact is simulated for the lock: the jaws carry no collider. |
| Rack-side retention | Two visible 2.5 x 20 x 20 mm pawls with a 600 N / 30 N-m `Rack`-to-module fixed joint, enabled only after the measured seating predicate | Visible geometry without contact colliders. The reaction magnitude is not exposed, so no pawl load can be quoted. |
| Contact | PhysX rigid contact with authored friction pairs | Forces are a relative damage proxy, not an absolute budget. Friction values are chosen per surface and are not measured from any material pair. |
| Perception | Rendered RGB-D, 15 Hz, with a radiation-noise model on RGB, and a flush ArUco datum pair | No lens distortion, no motion blur, no exposure control, no specular behaviour of real anodised aluminium, no sun-angle sweep, no eclipse transition. The datum is authored as code-native geometry, not printed and photographed. |
| Not modelled at all | connector mating, cabling, thermal expansion, vacuum cold-welding, outgassing, plume, dust, radiation-induced sensor upsets | Any of these can dominate a real changeout. |

## The three claims that would move first on hardware

1. **The `2c/L` admissibility bound would survive; the numbers feeding it would
   not.** The bound is Whitney's classical wedging geometry and does not depend on
   the simulator. What depends on the simulator is the *delivered* attitude that
   goes into it -- 46 mrad at hand-over here -- and a real UR10e with a real
   gripper on a real rail will not deliver that. The specification's shape is
   robust; its constants are not.
2. **The form lock is the largest single idealisation.** Everything downstream of
   capture assumes the module is rigidly attached to the wrist, and that
   assumption is enforced by a joint rather than earned by contact. On hardware
   the lock is a mechanism with backlash, and the transit retention numbers
   (1.05 mm and 3.27 mrad maximum drift in the continuous RGB-D episode) are the
   first thing that would degrade.
3. **Perception would degrade unevenly, not uniformly.** The rendered marker has
   perfect contrast and no blur. The failure the sight-line derivation found --
   the destination bay's own lead-in covering a centred datum for 154 mm of the
   529 mm stroke -- is *geometric* and would reproduce exactly on hardware. The
   detection rate on the frames where the datum is visible would not.

## The one caveat that is easy to misread

A simulated sweep of the rail's stopping error measures **a policy trained at one
base position**, not the geometry's tolerance for one. At +10 mm the chain scores
1.6%, and sixty of its sixty-three failures time out inside the *learned* phases
while the channel is untouched and the analytical kinematic gate passes. Whatever
the ladder between 0 and 10 mm turns out to look like, a number from it bounds
*this policy*, and the rail requirement a spacecraft designer should take from
this repository is the closed-form one, not the swept one.

## The cheapest falsifying experiment

**Do not start with the arm.** Start with a bench mock-up of one bay and one
module, on a linear stage, in 1 g, with the flush datum pair and the shipped
camera calibration. Two afternoons on a fixture that costs less than an arm test
the two claims the paper actually rests on.

### Experiment 1 -- the admissibility law, on a linear stage

Push the module in on the stage at a commanded tilt swept through the derived
`2c/L` bound, and record the depth at which it wedges.

* **What it falsifies.** `evidence/insert_depth_is_attitude.json` claims a module
  held at `theta` engages at most `2c/theta` before it wedges, and the recorded
  seating sweep reports measured attitude across eight points at a ratio to
  `2c/L` that falls monotonically from 1.02 to 0.87 -- a structured deviation
  and not scatter. The fit is `attitude = 3.609 * relief + 6.217 mrad` with
  R^2 = 0.9998, so the *form* is confirmed and the coefficient is 0.81 of the
  law's. This experiment is what separates the two candidate explanations. On a stage that is a direct measurement: plot depth against commanded
  tilt and the law is a line whose slope is `2c`.
* **What it needs.** One bay, one module, a tilt fixture, a linear stage with a
  depth readout, and no robot at all.
* **What a null result means.** If depth does not fall as `1/theta`, the
  admissibility criterion that sizes the rack in this repository is wrong, and
  every derived clearance goes with it.

### Experiment 2 -- the sight-line derivation, statically

With the same fixture, record the datum pair through the full stroke and compare
the measured readable band against
`evidence/rack_sightline_datum_pair_v1.json`.

* **What it falsifies.** The derivation makes a specific, falsifiable prediction
  about where each plate is readable, and it already validated itself against the
  recorded stopping depth of the dual-camera run. On a bench it either reproduces
  the predicted occlusion band or it does not.
* **What it needs.** No robot and no motion control -- a stage, a camera at the
  shipped calibration, and printed datum plates.
* **What a null result means.** If the band is wider than derived, the datum pair
  is over-engineered and one plate would have done. If it is narrower, the
  geometry that unblocked the continuous episode is not the geometry that would
  unblock a real one.

The manipulation result is the third experiment, not the first. Both experiments
above test the *specification*, which is this project's output; a robot test would
only tell us about the controller, which is not.
