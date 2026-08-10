# Perception stage: replacing the injected belief with an estimated one

The pose-belief task displaces the slot by an amount the policy is told nothing
about, and force feedback is the only channel that can recover it. This stage
replaces "told nothing" with "told something approximate, from a camera", which
is where the project's domain randomization finally becomes load-bearing rather
than merely present.

Nothing here is built yet. This page exists because one arithmetic check, done
before any code, changes the design substantially.

---

## 1. The finding that shapes everything

**The authored camera cannot see the quantity it would have to estimate.**

`scene_cfg.make_tiled_camera_cfg` is a 64x64 pinhole with an 18 mm focal length
against a 22 mm aperture, mounted at (-0.55, -0.65, 1.15) in the environment
frame. The pose-belief task's channel mouth now sits at x = 0.84. From those
numbers alone:

| Quantity | Value |
| --- | ---: |
| Horizontal field of view | 62.9 degrees |
| Camera to channel mouth | 1.593 m |
| Scene width in view | 1.948 m across 64 px |
| Ground resolution | **30.4 mm per pixel** |
| A 4 mm slot displacement | **0.13 px** |

A tenth of a pixel is not a hard regression problem, it is an absent signal. A
network trained on those images would learn the prior mean and report it
confidently, and the honest reading of a flat curve afterwards would be
ambiguous between "vision does not help" and "this camera saw nothing".

The camera pose is also inherited unchanged from the deleted eight-phase swap
scene, which framed a workcell that no longer exists and a slot mouth that has
since moved 390 mm downstream.

### What fixes it

Narrowing the field of view, not adding pixels. Holding 64x64 and the mount
where it is:

| Focal length | Field of view | Resolution | A 4 mm offset |
| ---: | ---: | ---: | ---: |
| 18 mm (today) | 62.9 deg | 30.4 mm/px | 0.13 px |
| 60 mm | 20.8 deg | 9.13 mm/px | 0.44 px |
| 120 mm | 10.5 deg | 4.56 mm/px | 0.88 px |
| **180 mm** | **7.0 deg** | **3.04 mm/px** | **1.31 px** |
| 240 mm | 5.2 deg | 2.28 mm/px | 1.75 px |

180 mm is the recommendation. It puts a 4 mm displacement at 1.3 px, which
sub-pixel regression handles routinely, and it keeps the 64x64 tile and its
measured 256-environment throughput. A servicing camera aimed at the interface
rather than at the room is also the more realistic instrument.

**This must be confirmed by rendering a frame, not by trusting the arithmetic.**
A 7-degree cone at 1.6 m sees a 195 mm-wide patch; whether the channel mouth and
the blade's leading edge both fall inside it depends on the mount's aim, which
was authored for a different scene. Re-aiming is a geometry change and needs a
rendered frame to confirm, exactly as the gripper envelope did.

---

## 2. What the estimate should be

The unknown is one number: the channel's lateral displacement from nominal. The
head therefore regresses that displacement, in metres, and the policy consumes
it exactly where the injected belief goes today.

This is deliberately narrower than "estimate the blade pose". The blade's pose is
already known to millimetres from forward kinematics and the fixed-joint grasp;
estimating it again would be regressing a quantity the robot can already compute,
which is the same mistake the pre-pivot tasks made in a different costume. The
slot's position is the only thing in this scene that proprioception cannot reach.

A fixed world-mounted camera makes this the easiest possible form of the problem:
the slot's pixel position *is* the estimate, because nothing else moves it. That
is a feature, not a cheat — it isolates whether randomized lighting and materials
destroy a signal that is geometrically present.

---

## 3. Build order

Each step is small and each has a way of failing loudly.

1. **Re-aim and narrow the camera, then look at a frame.** Render one image per
   randomization setting and confirm the mouth and the blade edge are both in
   view. No training until an image is inspected.
2. **Point the vision task at the pose-belief lineage.** `vision_insertion_env_cfg.py`
   currently sits on the force-feedback task, which has no slot displacement and
   therefore nothing to estimate. It moves onto
   `ZeroGBladeUncertainInsertionEnvCfg`, and its `blade_pose` diagnostic group is
   replaced by the true displacement, which is the regression label.
3. **Collect.** `scripts/collect_teacher.py` already records images beside a label
   and a teacher action. Drive it with the certified force-aware policy from the
   pose-belief run so the images cover states that policy actually visits.
4. **Train the head, and check it against the prior first.** A regression that
   cannot beat "always predict zero" has learned nothing, and that comparison is
   the gate. Report error in millimetres against the 0.75 mm rail clearance and
   the 4 mm training ceiling.
5. **Close the loop.** Feed the estimate in place of the injected displacement and
   re-run the sweep. The comparison is three curves on one axis: injected-perfect
   (no uncertainty), estimated (vision), and injected-at-the-trained-ceiling.

## 4. The result this is expected to produce, and the honest alternative

The likely outcome is **not** that vision replaces force. It is that the two
compose, which is what IndustReal and FORGE both report:

> Vision narrows the uncertainty from tens of millimetres to a few. Force closes
> the last few millimetres to the 0.75 mm the rails allow.

If the pose head lands at, say, 2 mm of residual error, that is a *harder*
starting point than the 4 mm the policy trained against only in the sense that it
is real; the force-aware policy should absorb it and the force-blind one should
not. That is a stronger portfolio claim than either half alone, because it is an
end-to-end result with a measured error budget at each stage.

The alternative worth naming now: if randomized orbital lighting destroys a
1.3 px signal, the head will not beat its prior and the honest report is that a
64x64 camera under this lighting cannot support millimetre servicing. That is a
publishable negative result of exactly the kind this repository already keeps,
and the escape route is a higher-resolution tile or a wrist-mounted camera, both
of which cost throughput that `scripts/benchmark.py` can measure before anyone
commits to them.

## 5. What must not happen

- No training against images nobody has looked at.
- No pose head whose error is reported without the "always predict zero"
  baseline beside it.
- No replacing the injected displacement with an estimate until the estimate's
  error is measured on held-out episodes, in millimetres.
- The camera is presentation-adjacent geometry, so a re-aim gets a rendered frame
  and a recorded number, not an assumption.
