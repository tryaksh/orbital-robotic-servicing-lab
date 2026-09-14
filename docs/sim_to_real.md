# Simulation limits and hardware validation

Nothing in this repository has run on hardware. The useful hardware deliverable
is a set of testable interface requirements; the simulation is evidence about
one model, not flight readiness.

## What is modelled

| Element | Simulation model | Main limitation |
| --- | --- | --- |
| Gravity | Zero throughout the servicing runs | No orbital rate, gravity-gradient effects or tumbling client |
| Robot base | Fixed to the world | No spacecraft reaction or attitude-control coupling |
| Robot rail | Indexed robot base | Carriage compliance, backlash and stopping dynamics are not modelled |
| Robot-side form lock | Break-rated fixed joint during transit; bounded spring-damper during mating | Visible geometry, idealized load path; no realistic jaw/contact mechanism |
| Rack retention | Visible pawls plus a 600 N / 30 N-m Rack-to-module joint | Pawls have no contact colliders; reaction magnitude is not exposed |
| Contact | PhysX rigid-body contact and specified friction | Diagnostic contact loads are not hardware load ratings |
| Perception | Rendered RGB-D, calibrated fixed cameras and flush ArUco markers | No real sensor validation; calibration drift, exposure and blur need measurement |
| Work performed | One module transferred between bays | No electrical reconnection, thermal interface or spare-module logistics |

The module is not teleported, written directly to a target pose, attached to the
world, or moved by a hidden carrier during the selected mission.

## What is likely to change on hardware

Real lock backlash and compliance could increase transit drift. Rail stopping
error could change the learned capture and extraction distributions. Camera
errors could create long correlated failures that a held-pose corpus does not
capture. These effects need direct measurements before the simulation's numerical
bounds can become hardware requirements.

The rail's closed-form indexing bound is a static grip-reach calculation.
A simulation sweep of a policy trained at one base position measures that
policy's sensitivity as well as geometry. Neither number alone is a validated
carriage specification.

## First bench experiments

**Measure entry and seating on a fixture.** Build one bay and one module, use a
linear stage and adjustable tilt fixture, and sweep clearance and entry attitude.
Record contact locations, insertion force, achieved depth and released pose.
Compare the measurements with the approximate `2c/L` attitude and `2c/theta`
depth relations. The simulated relief sweep already shows coefficient differences;
the formulas should be tested as approximations rather than exact predictions.

**Measure the camera through the stroke.** Fit the two flush markers, reproduce
the specified camera geometry, and record each plate's visibility, decoding rate
and pose error throughout insertion. Then vary exposure, motion, calibration and
robot occlusion. The sight-line calculation predicts geometric occlusion only;
it does not predict whether a visible marker will decode.

**Measure capture and load transfer.** Instrument the proposed robot-side lock
and rack pawls. Check capture tolerance, backlash, release interlocks and retention
under representative forces and moments. The simulated joint break thresholds
are inputs to these experiments, not validated mechanism ratings.

Ground fixtures help test geometry and sensing. They do not reproduce zero-gravity
release: gravity may support or move the module. A later experiment must address
that difference and the spacecraft/robot reaction dynamics explicitly.

[Mechanical specification](service_interface_spec.md) ?
[Current results](NOW.md) - [Roadmap](../ROADMAP.md)
