# Fixed-base transfer refinement

A stationary arm can complete this two-bay transfer in the modeled workcell.
The earlier moving carriage is therefore not required for the demonstrated
episode. The refinement changes the controller recipe while keeping the rack,
module, frozen policy weights, actuator effort limits, camera pipeline and
physical completion criteria unchanged.

The selected continuous recording runs from clean source
`88235d8c27314b6092b0aac38506af9526ceea40`, at seed 6070 with one environment and
stable lighting. It passed all nine mission-verifier checks. That seed was used
to diagnose and refine the behavior; this recording is development evidence.
The later application revision `4a433e0` corrects command ownership during
camera-observation gaps. The selected V5 video keeps its original attribution:
its guarded extraction recorded zero blocked steps, so that corrected branch
was not exercised in the recording.

Commit `88235d8` identifies the Python revision; exact historical driver bytes
also require the [source snapshot registry](../evidence/source_snapshots.json).
The mixed-line-ending runtime snapshot is preserved without replacing it with
Git's normalized text. Fresh application validations use the current
byte-stable checkout and their own source hashes.

## What changed

The PPO extraction policy approached the clearance plane slowly, continuing
small corrections near the boundary. A guarded terminal path now takes over
only after the grip is established, within 20 mm of the plane and below
25 mm/s measured speed. A fresh camera detection defines a short axial tool
path ending 4 mm beyond the unchanged plane, with the tool's entry attitude
held fixed. Loss of the observation or grip stops commanded progress.
Capture and initial extraction remain learned; this terminal finish is scripted.
The original grip, velocity, clearance, dwell and lock-engagement predicates
still determine whether extraction actually succeeded.

The old solved transfer inherited an extraction action scale. The new rigid
transfer uses a synchronized quintic position and attitude profile, with zero
nominal endpoint velocity and acceleration. Its duration limits the nominal
Cartesian setpoint to 0.10 m/s and 0.25 m/s², and 0.30 rad/s and 0.80 rad/s².
These are simulation controller settings, not a qualified hardware envelope.
They do not bound physical tracking error, joint acceleration or contact force.

Bounded joint-encoder integral trim compensates static tracking offset while
retaining the original physical joint drives. Bias is limited to 0.06 rad and
changes by at most 0.03 rad/s. A continuing profile retains its accepted
setpoint and compensation rather than restarting from a lagging measured pose.
The robot-side lock handoff likewise holds the accepted joint targets until
physical engagement is observed. The original pre-engagement drift reference
and 2.5 mm transit limit remain in place.

Insertion retains the deployed camera envelope, guarded axial progression,
compliant load path, force caps and seating/release criteria. It now commands
absolute inverse kinematics, with filtered camera-observed spring-deflection
compensation. Separating spring deflection from ordinary arm tracking error
avoids adding the same tracking error twice to the outer correction loop.

Motion commands go through simulated robot joint drives. No module teleport,
world-mounted payload constraint or hidden carrier was introduced. The robot
base remains fixed throughout the selected episode.

## Measurements from the two recordings

The baseline is the previous portfolio video's original seed-6070 recording,
before its 1.5× playback conversion. Both recordings use one environment and
stable lighting. Derivatives below come from unfiltered 30 Hz body states in
simulation time; presentation speed does not enter the calculation.

| Measurement | Previous recording | Fixed-base recording |
| --- | ---: | ---: |
| Maximum robot-base displacement | 245 mm | 0 mm |
| Extraction duration | 17.57 s | 11.50 s |
| Low-speed time near extraction plane | 7.17 s | 0.17 s |
| Longest continuous low-speed interval there | 3.10 s | 0.067 s |
| Peak sampled module speed during transfer | 0.327 m/s | 0.109 m/s |
| Peak sampled module acceleration during transfer | 2.711 m/s² | 0.288 m/s² |
| Maximum tool-to-module transit drift | 1.293 mm | 0.221 mm |
| Mission verification | 9/9 | 9/9 |

The low-speed diagnostic counts whole extraction intervals whose endpoints
are within 5 mm of the 0.225 m plane and whose three-dimensional speed is below
5 mm/s. It is an analysis threshold, not a changed success predicate. Phase
boundary changes remain in the derivatives. These samples cannot reconstruct
substep contact impulses, and the table describes transfer rather than claiming
every phase improved. Capture transients remain in the uncut recording.

The [fixed-base recording](../evidence/workflow_stability_latch_handoff_v5_seed6070.json)
has 1,302/1,302 camera detections and 0.733333 seconds
of rack-only hold. Its 1,294 captured states span 1,293 control intervals, or
43.1 seconds. Its workstation bundles are preserved under
`D:/portfolio/.lab-work/orbital-improvement/website-baseline/` and
`D:/portfolio/.lab-work/orbital-improvement/smooth-fixed-v5/`, each with the
workflow report, body-state capture, motion analysis and verification.
The public [session index](../evidence/workflow_stability_session_index.json)
links the original report, source bindings, compressed state/trace archives,
motion analysis and preserved failed variants. The
[baseline report](../evidence/workflow_stability_baseline_recording_seed6070.json)
retains the earlier recording. The new analysis also records actuator targets;
the old capture did not, so
there is no matched before/after actuator-target-continuity claim.

## A failed screen that the successful video did not reveal

The first eight-condition comparison at seed 4070 rejected V5 for promotion:
the legacy recipe completed **5/8**, while V5 completed **3/8**. The existing
physical retention/release audit agrees with both NPZ counts. Environments
**2 and 6** changed from success to failure, with no gained successes. Both
new failures ended during extraction. Only this first pair ran; there is no
24-condition V5 result, and seeds 5070 and 6070 were not run for that revision.
[Preserved failed screen and original artifacts](../evidence/workflow_stability_camera_cadence_regression_v5_seed4070.json)

| Regressed environment | Guarded motion steps | Reported guard-paused steps | Terminal module speed |
| --- | ---: | ---: | ---: |
| 2 | 275 | 274 | 93.7 mm/s |
| 6 | 263 | 263 | 88.6 mm/s |

The defect was in how a pause handed control to the robot's actuators. On a
blocked camera/grip gate, V5 zeroed the Cartesian action and disabled the
absolute joint override. Relative IK then took over with a different target;
fresh observations restored the biased absolute command. Production-method
tests reproduce this discontinuity under alternating fresh/missing observation
flags. The report counters show repeated pauses, but combine observation and
grip gating; they do not establish an exact camera frequency or a full
per-step sequence.

Revision `4a433e0` retains the last accepted joint target on those blocked
steps, while keeping profile time, trim and the physical grip command unchanged.
It resumes the same path when the gates permit progress. Three focused tests
fail on V5 and pass on the corrected production methods. This establishes the
command-handling fix; its physical performance is evaluated in the separate
fresh comparison below, with all six cohorts rerun.

The original supervisor stopped between completed cohorts when it reached an
empty reserved seed-5070 directory. That scheduling handoff was planned before
the refined first-pair result. After the regression became known, the V5
continuation was abandoned. The original reports, NPZs, logs, manifest,
supervisor exception and pre-result scheduling plan remain intact. The failed
screen is not pooled with the newer comparison or rescored under new criteria.

## Fresh paired regression comparison

The corrected comparison completed **eight conditions per arm at seed 4070**
on clean source `4a433e0`: **5/8 legacy** and **6/8 corrected fixed-base**
successes. The existing physical retention/release audit agrees. No matched
condition changed from success to failure.
[Partial paired result and original artifacts](../evidence/workflow_stability_paired_summary.json)

The completed pair used 1,900 control steps, randomized lighting, no video and
the same checkpoints. All eight conditions remain in each denominator,
including failures. The planned 24-condition-per-arm validation was shortened
at the owner's request: seed 5070 was interrupted and 6070 was not run.
Incomplete artifacts are preserved and unscored. This partial result does not
complete the original protocol or qualify the unchanged 95% target.

This compares complete recipes: base transport, timing, terminal extraction,
joint trim and insertion solver change together. It does not isolate the
contribution of an individual change. The seeds are held out from policy
training, but both 6070 and 4070 were reused during controller development. The result is a
development regression comparison, not an independent generalization estimate.
The older 17/24 camera cohort remains a measurement of the previous v2 recipe.

## Service validation and reproduction

The selected video is separate from the application's exact-command validation
and normal worker execution. Those records check their own source, command,
checkpoint and output bindings:

- [Fixed-base profile validation](../evidence/live_service_stability_validation_seed6070.json)
- [Normal application execution](../evidence/live_service_stability_application_seed6070.json)

The fresh-checkout profile validation passed all nine checks on clean source
`4a433e0`. It recorded **1,173/1,173 detections**, **0.517 mm maximum transit
drift**, **0.045 mm final lateral error**, **1.543 mrad final orientation error**
and **0.733333 seconds rack-only hold**. It uses the exact current command and
byte-stable runtime source. The earlier candidate validation's 1,192 detections
and 0.138 mm drift and the selected V5 video's 1,302 detections and 0.221 mm drift
remain measurements of their separate runs.

The isolated normal worker passed all nine checks and all five artifact hashes
on clean source `e501500`, with 1,218/1,218 detections, 0.164 mm maximum transit
drift and 0.733333 seconds rack-only hold. An earlier concurrent worker's
zero-detection failure remains preserved; its cause is not established.

Run the current application using [the mission guide](compute_service_demo.md).
To regenerate a paired comparison on a clean checkout:

```powershell
python scripts/compare_workflow_stability.py --output artifacts/new-stability-comparison
```

The comparison requires Isaac Sim. The output directory must be new; commands,
logs, eight-row episode archives and both aggregates are retained. Its legacy
arm restores the earlier rail/profile/solver choices without deleting them.

The CPU analyzer can inspect any compatible saved body-state capture:

```powershell
python scripts/analyze_workflow_motion.py --poses <capture>/body_poses.npz --report <capture>/workflow_report.json --trace <capture>/handoff_trace.npz --output <capture>/motion_analysis.json
```

## Boundaries and remaining work

The base is world-fixed. Spacecraft reaction, attitude control and mounting
flexibility are outside the simulation. The visible service frame is offline
rendering context, with no collision or rigid-body physics; it does not supply
hidden support to the module. The renderer refuses it for a capture with a
moving base. Robot-side and rack-side locks still use idealized joint load paths,
so real mechanisms need contact, strength, sensing and release tests.

A small insertion-to-settling ownership transition remains: the absolute arm
override gives way to zero-action relative IK during the supported hold. In
the selected recording it produced a 3.868 mrad maximum joint-target step,
0.084 mm wrist movement on that step and 0.070 mm net wrist movement over the
supported interval. All existing mission checks passed. Its proposed controller
change is deferred so the validated source and paired comparison remain intact.
[Bounded follow-up](NEXT_WORK.md#t23-supported-settling-command-ownership)

The [video](media/orbital-stable-demo.mp4) shows every captured state at real-time
playback, followed by a disclosed 1.5-second final still. It removes no pauses
and interpolates no motion. The still does not extend the measured rack hold.
One successful fixed-base transfer establishes feasibility for this modeled
workcell, not flight readiness or success from arbitrary poses.
