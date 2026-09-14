# Current state

Verified on 2026-09-14. This file owns the measured results and their limits.
[README](../README.md) introduces the project; [NEXT_WORK](NEXT_WORK.md) lists
bounded follow-up tasks. Earlier detailed investigations are preserved in
[the state archive](handover/state_before_mission_application.md).

## Current application

`zero-g-mission` runs the fixed-base v3 camera-driven recipe through the compute
service worker. It checks source and checkpoint hashes before admission, records
video and traces, and independently verifies the report before reporting success.
`verify` rechecks every artifact hash and the physical completion fields offline.

The packaged policies are byte-identical to the original training artifacts:

| Role | Included checkpoint | Executes actions? |
| --- | --- | --- |
| Capture | `policies/servicing_v2/capture_v7m130.pth` | Yes, PPO |
| Initial extraction | `policies/servicing_v2/extract_v19noised.pth` | Yes, PPO, trained with estimator noise; guarded control finishes the pull |
| Insertion | `policies/servicing_v2/insert_v13m130.pth` | No; loaded for policy-set compatibility |

[Checkpoint hashes and original training paths](../policies/servicing_v2/MANIFEST.json)
are in git. Other experimental checkpoints remain under gitignored `logs/` and
`checkpoints/`; a clone does not include those training archives.

The live recipe uses RGB-D module pose, robot-kinematic velocity, the existing
lead-in guard, compliant mating, simultaneous release and rack retention. The
base is stationary. A scripted terminal extraction path follows PPO; synchronized
quintic setpoints and bounded joint trim guide transfer; guarded insertion uses
absolute IK with filtered spring-deflection compensation. No physical success
criterion, tolerance, policy weight or actuator effort limit changed.
[Controller changes and measured motion](TRANSFER_STABILITY.md)

### Fixed-base profile validation

The [fresh-checkout admission validation](../evidence/live_service_stability_validation_seed6070.json)
ran on clean source `4a433e0`, service revision `isaac-rgbd-strict-mission-v3`.
It passed all nine checks with **1,173/1,173 detections**, **0.517 mm maximum
transit drift**, **0.045 mm final lateral error**, **1.543 mrad final orientation
error** and **0.733333 seconds rack-only hold**. Its exact current command and
raw runtime source hashes match the byte-stable checkout.

The [initial candidate-checkout validation](../evidence/live_service_stability_initial_validation_seed6070.json)
on `88235d8` also passed all nine checks:
1,192/1,192 detections, 0.138 mm transit drift, 0.577 / 0.750 mm final lateral /
axial error, 0.119 mrad orientation error and 0.733333 seconds rack-only hold.
A fresh checkout changed raw line-ending bytes in the driver without changing
its Python text. Its exact-source admission is validated separately; the earlier
candidate result must not be relabelled as that fresh-clone validation.

The rack pawls engage after measured seating. Their load path is an idealized
600 N / 30 N-m Rack-to-module fixed joint, not pawl contact. The robot-side lock
uses a break-rated joint during transit and a bounded spring-damper during mating.
A video and trace accompany this report. Its command, checkpoint and runtime
source bindings determine service admission. This is a demonstrated episode,
not a statistical reliability qualification; seed 6070 was reused in development.

### Normal v3 application execution

The [isolated normal worker execution](../evidence/live_service_stability_application_seed6070.json)
passed all nine mission checks and all five artifact hashes on clean source
`e501500`. It recorded **1,218/1,218 detections**, **0.164 mm maximum transit
drift** and **0.733333 seconds rack-only hold**. These are this worker run's
measurements, separate from the profile validation and selected V5 video.

### Selected fixed-base demonstration

The continuous [video](media/orbital-stable-demo.mp4) is the separate V5 recording
from clean commit `88235d8`: 1,302/1,302 detections, 0.221 mm maximum transit drift,
and 0.733333 seconds rack-only hold. All nine mission checks passed. Its 1,293
physical control intervals span 43.1 seconds; playback is real time with a
1.5-second terminal still. This is not an additional physical hold or a new
independent seed. [Recorded demonstration](../evidence/workflow_stability_latch_handoff_v5_seed6070.json)

### Preserved v2 application recordings

The earlier [profile validation](../evidence/live_service_current_validation_seed6070.json)
used clean commit `5387cbf`: 1,304/1,304 detections, 1.436 mm transit drift,
1.672 / 0.590 mm final lateral / axial error, 44.081 mrad orientation error and
0.733333 seconds rack-only hold. It belongs to the earlier moving-carriage recipe.

The earlier [normal worker run](../evidence/live_service_application_seed6070_v1.json)
used clean commit `f7cdf23`, the packaged weights and the readiness checks. It
passed all nine mission checks and verified all five output hashes. It recorded
**1,320/1,320 detections**, **1.370 mm maximum transit drift**, final lateral error
**0.729 mm**, and **0.733333 s rack-only hold**. This is a second recording at
seed 6070, not an independent evaluation seed or a new success rate.

### Unchanged perception validation

[Seed 287 certificate](../evidence/fiducial_rgbd_service_current_seed287.json):
1,024 rendered held-out poses on the deployed 640 x 640 camera and flush datum
pair, ArUco 23 and 15 at module x = -0.115 and +0.115 m.

| Measurement | Result | Unchanged gate |
| --- | ---: | ---: |
| Overall detections | 983/1,024 = 96.00% | At least 90% |
| Critical-bay detections | 682/683 = 99.85% | At least 99% |
| Position error, p95 | 1.931 mm | Below 20 mm |
| Orientation error, p95 | 11.973 mrad | Below 50 mrad |
| Occupancy exact match | 100% | At least 95% |

Error statistics are conditional on detection. The collector holds the robot
still; moving-robot occlusion is tested by the separate continuous mission.
The 327 MB corpus remains local at
`datasets/fiducial_rgbd_service_current_seed287.npz`, with its hash in the report.
[Collection and validation commands](compute_service_demo.md#revalidate-the-profile)

## Research cohorts

### Fixed-base regression comparison

The [completed seed-4070 pair](../evidence/workflow_stability_paired_summary.json)
on clean source `4a433e0` records **5/8 legacy** and **6/8 corrected fixed-base**
successes. The existing physical retention/release audit agrees, with no
matched success-to-failure changes. These are eight conditions per arm, not 24.

The planned three-seed comparison was shortened at the owner's request.
Seed 5070 was interrupted; seed 6070 was not run. Incomplete work is retained
and unscored. The completed pair used randomized lighting and 1,900 control
steps. Several controller choices and base transport change together, and both
4070 and 6070 informed development. This is a partial regression check, not an
independent holdout or a completed qualification of the unchanged 95% gate.

### Earlier full-chain results

The following cohorts retain their original configurations and completion rules.
They are historical comparisons, not recertification of the new fixed-base recipe.

| Configuration | Successes | Rate | Wilson 95% interval |
| --- | ---: | ---: | --- |
| State task, strict rack retention | 22/24 | 91.67% | [74.2%, 97.7%] |
| Paired state-task control without rack retention | 17/24 | 70.83% | [50.8%, 85.1%] |
| Vision task, oracle module pose | 20/24 | 83.33% | [64.1%, 93.3%] |
| Vision task, original camera pose pipeline | 4/24 | 16.67% | [6.7%, 35.9%] |
| Previous v2 application: noise-trained extraction + kinematic velocity + lead-in guard | 17/24 | 70.83% | [50.8%, 85.1%] |

Sources: [strict state chain](../evidence/workflow_robot_carried_release_rack_retention_v1_certification.json),
[no-rack control](../evidence/workflow_robot_carried_release_rack_retention_control_v1_certification.json),
[oracle control](../evidence/workflow_robot_carried_m130pin_vision_oracle_control_v2_certification.json),
[original camera cohort](../evidence/workflow_robot_carried_m130pin_vision_datum_pair_certification.json),
[previous v2 application recipe cohort](../evidence/workflow_robot_carried_vision_noised_extract_kinematic_leadin_certification.json).

Each cohort has three held-out evaluation seeds, eight environments per seed.
The state task and vision task differ in observation and camera timing; compare
the two vision arms to isolate the pose-source substitution. The combined camera
recipe changes three factors from the original. Its result is not attributable
to retraining alone.

All of these earlier full-chain rates remain below the unchanged 95% gate. An older camera
milestone used a 50% target; passing that milestone did not qualify the chain.
The recovered factorial contains further arms that still need a provenance and
scope audit (T21), so the selected canonical recipe is not claimed to be optimal.

The legacy supported-settle result is **94/96 = 97.92%**. It checked the module
while robot support remained and is not the current completion rate.
[Legacy certificate](../evidence/workflow_robot_carried_m130pin_guarded_certification.json)

### Skills and seating controller

| Skill or comparison | Result | Meaning |
| --- | ---: | --- |
| Capture v7m130, current derived rack | 7,829/9,009 = 86.90% | Below 95%; earlier certificate 85.69% |
| Extraction v18pin, current derived rack | 7,891/9,004 = 87.64% | Below 95%; earlier certificate 87.75% |
| Learned insertion v20chain | 0/1,536 = 0.00% | Preserved negative baseline |
| Learned insertion v24, isolated | 1,103/3,000 = 36.77% | Its reset distribution differs from actual chain handoffs |
| Learned insertion v24, chain | 0/96 | Not selected |
| Force-aware insertion v33force, isolated | 2,977/3,001 = 99.20% | Passes its isolated skill gate |
| v33force versus guarded insertion, prescribed 11.065 mm bay | 24/96 versus 23/96 | No useful chain advantage on these paired episodes |

The relevant certificates are indexed by skill in the
[evidence manifest](../evidence/MANIFEST.json). The learned controller has not
replaced guarded insertion. [Controller comparison](seating_controller.md)

Three earlier reward variants ended at **84.26, 84.61 and 84.58 mrad** against a
**52.4 mrad** tolerance. That is evidence that these objective changes did not
resolve this configuration's alignment problem; it does not prove that no
controller could. Training reward is not a rate.
[Diagnosis](../evidence/insert_attitude_diagnosis.json)

Failure attribution on the unchanged skill certificates:

- Capture: **1,170 of its 1,180 failures** ended outside the 10 mm grip-position
  condition; 1,020 violated that recorded condition alone. Median distance was
  95.9 mm on failures, compared with 4.0 mm on successes.
- Extraction: **1,024 of its 1,113 failures** exceeded the derived 14.29 mm/s
  residual linear-velocity limit. Grip loss and residual motion often co-occurred.

These are terminal associations, not a causal time ordering. Some predicate terms
were not recorded as episode columns.
[Attribution report](../evidence/skill_gate_attrition_v1.json)

## Mechanical findings and limits

The channel tool reports the relieved destination as incompatible with guaranteed
passive seating: its geometry allows **69.68 mrad** lateral attitude and 56.06 mrad
vertical attitude, against a **52.36 mrad** gate, and permits offsets up to
15.678 mm against a 2.5 mm lateral gate. It can contain a correctly aligned module;
it simply cannot guarantee that every resting pose meets the gate.
[Channel verdict](../evidence/channel_verdict_shipped_bay_v1.json)

The small-angle clearance relation `2c/L` is a static geometric approximation.
For the measured 46 mrad delivered attitude, the analytical lateral-clearance
window is 10.350 to 11.781 mm per side, with a midpoint near 11.065 mm. The
corrected section check accepts four of the 36 sampled cross-sections. The
rail's 1.623 mm bound is a geometric grip bound, not demonstrated policy tolerance.
[Derivation](../src/zero_g_blade_swap/servicing_design.py)

No passive clearance simultaneously admits that delivered attitude and guarantees
2.5 mm centring. Active alignment is required; the lead-ins can correct attitude
during insertion. Across the recorded relief sweep, the simple law's coefficients
differ from the simulated values, so it must not be presented as a calibrated
millimetre-accurate predictor.
[Recorded geometry check](../evidence/workcell_geometry_check.json)

The serviceability envelope remains **not qualified**. Clearance, section and
base-offset sweeps do not all agree with the static bounds. The corrected
clearance sweep moves the mouth with the walls; at 6 mm per side it scored 36/64,
versus 0/64 when only the guides moved. Those arms are retained separately.
[Boundary decision](../evidence/serviceability_boundary_validation_v2.json)

A separate axial-pull diagnostic demanded **66.4 N**, compared with about 6 N
available from passive pad friction. This is **not a hardware load rating**.
[Pull gate](../evidence/grasp_axial_pull_gate.json)

Two flush markers remove the predicted simultaneous line-of-sight occlusion
through the seating stroke. Geometric visibility does not establish decoding,
exposure, or motion-blur performance; the camera measurements above test those
rendered detections separately.
[Sight-line analysis](../evidence/rack_sightline_datum_pair_v1.json)

## Evidence and reproducibility

The manifest contains **80 canonical, 12 retracted and 228 historical** reports.
Quote canonical reports with their scope. Failed and superseded results are kept;
[evidence/RETRACTED.md](../evidence/RETRACTED.md) records withdrawn claims.

**45 reports carry** source-file bindings. Seventeen are recovered through
reachable git revisions or exact committed source-snapshot overlays;
**28 cannot be fully recovered**. Those comprise 27 inherited gaps from uncommitted code
and the preserved failed axial-only extraction probe, whose exact mixed-line-ending
runtime source was not recovered. Normalized text is not substituted for its
recorded bytes.
[T0](NEXT_WORK.md#t0) remains open for those older experiments. The new mission
and perception reports bind current source, and the mission weights are included.

The default criterion-currency check reads the legacy CLAUDE.md claim list and
currently selects no applicable reports. It is not an audit of the whole corpus.
`--all` is a conservative file-change screen; read its flags with each report's
scope and source bindings.

```powershell
ruff check src scripts tests
pytest -m "not isaac and not camera and not benchmark"
python scripts/check_criterion_currency.py
python scripts/check_source_provenance.py --depth 200
python scripts/build_evidence_manifest.py --check
python scripts/build_script_index.py --check
```

Use bare `pytest`, as CI does. CPU tests require no simulator; optional OpenCV
and camera tests depend on the installed environment. The complete physical
qualification gates remain open despite passing application tests.

## What is not modelled

No hardware validation, spacecraft reaction dynamics, free-flying base,
electrical reconnection, or realistic lock/pawl contact. The current base is
stationary; the old moving carriage's compliant load path was not simulated.
Every headline learned checkpoint
comes from one training seed. [Detailed limits](sim_to_real.md)
