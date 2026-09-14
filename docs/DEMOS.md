# Demonstrations and provenance

A recording shows one episode. Cohort results have separate denominators,
seeds and uncertainty. The selected fixed-base video is included below; larger
raw recordings remain in their original artifact bundles. Reports, captured
states and hashes identify what actually ran.

## Current fixed-base demonstration

[![Fixed-base module servicing](media/orbital-stable-poster.jpg)](media/orbital-stable-demo.mp4)

[Play the continuous recording](media/orbital-stable-demo.mp4).

The V5 demonstration runs from clean source
`88235d8c27314b6092b0aac38506af9526ceea40`, one environment and seed 6070, with
stable lighting. It passed all nine source mission-verifier checks:
**1,302/1,302 detections**, **0.221 mm maximum tool-to-module transit drift**,
and **0.733333 seconds held by the rack alone** after both robot supports release.
[Recorded episode](../evidence/workflow_stability_latch_handoff_v5_seed6070.json)

The recording begins before the first driver action. PPO captures the module
and begins extraction; a guarded terminal path completes the pull. The
stationary arm carries it using smooth solved motion and joint trim, then
camera-guided absolute IK inserts it. The rack engages after measured seating;
the robot releases only after the supported settling check, followed by the
unchanged rack-only recheck.

The video shows all 1,294 recorded states in sequence at 30 fps, with no motion
interpolation or removed pauses. They span 43.1 seconds of simulated motion.
Playback is real time, followed by a 1.5-second final-frame hold for readability.
That still does not extend the measured physical hold. Lighting, materials and
framing are presentation changes; the visible service frame illustrates the
existing fixed mounting assumption and adds no collision geometry or physical
support. [Refinement and measured limits](TRANSFER_STABILITY.md)

Seed 6070 was used during development. This successful video is not an
independent evaluation seed or a new reliability rate. The separate corrected
regression completed only the eight-condition seed-4070 pair: 6/8 versus 5/8
legacy, with no paired losses. The planned 24-condition validation was shortened
at the owner's request; it does not qualify the 95% target. The older 17/24 camera result belongs to the earlier
moving-carriage recipe and must not be attached to this video's controller.

## Current service validations

The fresh-checkout [v3 profile validation](../evidence/live_service_stability_validation_seed6070.json)
passed all nine checks on clean source `4a433e0`. It recorded 1,173/1,173
detections, 0.517 mm maximum transit drift and 0.733333 seconds rack-only hold.
Its exact command and byte-stable runtime source establish profile admission.
These measurements belong to that run, not the selected V5 video above.

The isolated [normal v3 worker execution](../evidence/live_service_stability_application_seed6070.json)
passed all nine checks and all five artifact hashes on clean source `e501500`:
1,218/1,218 detections, 0.164 mm transit drift and 0.733333 seconds rack-only hold.

## Preserved v2 application recordings

The earlier v2 `zero-g-mission run --seed 6070` execution passed all nine completion
checks and all five artifact hashes. It used the included checkpoint set,
RGB-D module pose, robot-kinematic velocity, guarded insertion, simultaneous
release and rack retention.

[Application audit](../evidence/live_service_application_seed6070_v1.json)
records the clean source commit `f7cdf23`, inputs, seed, output hashes and
verification results. The raw files on the measured workstation are:

```text
artifacts/mission_final_seed6070/jobs/65461e26-e57f-4374-af19-913c3312d9c9/
  job.json
  artifacts/workflow_report.json
  artifacts/video/rl-video-step-0.mp4
```

Measured in this recording: **1,320/1,320 detections**, **1.370 mm maximum transit
drift**, and **0.733333 s rack-only hold** after both robot supports released.
One environment, one seed, stable lighting. The joint load paths are idealized;
this is not a hardware result or a new pooled success rate.

An earlier run of that same v2 profile also passed:
[profile validation](../evidence/live_service_current_validation_seed6070.json),
clean commit `5387cbf`, 1,304/1,304 detections and 1.436 mm drift. These are two
recordings at the same seed, not two independent evaluation seeds.

## Reproduce and verify

```powershell
zero-g-mission preflight
zero-g-mission run --seed 6070 --output artifacts/new-recording
zero-g-mission verify <job.json-path-printed-by-the-run>
```

The verifier checks report fields and hashes. Before publishing a cut, also
decode the video, inspect its beginning, transit, insertion and release, and
keep the raw recording's hash. Captions must identify the controller that
executed each phase. An edited video must disclose speed changes and retain the
single-episode scope. [Mission guide](compute_service_demo.md)

## Other historical recordings

| Recording | What it shows |
| --- | --- |
| `artifacts/robotcarried/video_certified_chain_clip_seed5070/rl-video-step-0.mp4` and seed 6070 | Successful single episodes of the state-driven strict chain, clean commit `db5b79f`; no live camera perception |
| `artifacts/robotcarried/video_datum_pair_rack_clean_seed6070/rl-video-step-0.mp4` | A successful original camera-profile episode, clean commit `7a82db2`; 1,772/1,772 detections. Its pooled profile scored 4/24 |
| `artifacts/robotcarried/video/1_grasp_and_extract.mp4` and `2_carry_across_on_the_rail.mp4` | Superseded w65 checkpoints and geometry; retain as historical demonstrations |
| `3_insertion_missed_the_lateral_gate.mp4` and `4_push_in_attempt_no_release.mp4` | Preserved failures, renamed because their original names incorrectly claimed seating and release |

The failed insertion run reported 4.62 mm lateral error against a 2.5 mm gate.
A successful-looking motion does not override that report. Older perception
clips under `artifacts/demo/` predate the current workcell and datum layout.
