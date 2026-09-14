# Demonstrations and provenance

A recording shows one episode. Reliability comes from a separate cohort, with
its seed count and confidence interval. Videos are kept outside git; their
reports and hashes identify what was actually run.

## Current mission application

The normal `zero-g-mission run --seed 6070` execution passed all nine completion
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

An earlier run of the exact profile also passed:
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

## Earlier recordings

| Recording | What it shows |
| --- | --- |
| `artifacts/robotcarried/video_certified_chain_clip_seed5070/rl-video-step-0.mp4` and seed 6070 | Successful single episodes of the state-driven strict chain, clean commit `db5b79f`; no live camera perception |
| `artifacts/robotcarried/video_datum_pair_rack_clean_seed6070/rl-video-step-0.mp4` | A successful original camera-profile episode, clean commit `7a82db2`; 1,772/1,772 detections. Its pooled profile scored 4/24 |
| `artifacts/robotcarried/video/1_grasp_and_extract.mp4` and `2_carry_across_on_the_rail.mp4` | Superseded w65 checkpoints and geometry; retain as historical demonstrations |
| `3_insertion_missed_the_lateral_gate.mp4` and `4_push_in_attempt_no_release.mp4` | Preserved failures, renamed because their original names incorrectly claimed seating and release |

The failed insertion run reported 4.62 mm lateral error against a 2.5 mm gate.
A successful-looking motion does not override that report. Older perception
clips under `artifacts/demo/` predate the current workcell and datum layout.
