# Run and verify a servicing mission

The application runs one camera-driven module transfer, records it, and checks
its physical completion and artifact hashes. It uses the same worker as the
optional local API; a browser is not required.

## Terminal commands

After [installation](INSTALL.md):

```powershell
zero-g-mission preflight
zero-g-mission run --seed 6070 --output artifacts/my-mission
zero-g-mission verify artifacts/my-mission/jobs/<job-id>/job.json
```

`python -m zero_g_blade_swap.service.mission_cli` is equivalent to `zero-g-mission`.
The run prints the exact `job.json` path. `--output` must name a new directory,
so a run cannot overwrite a prior result or recover another worker's jobs.
Ctrl+C cancels the worker and retains partial artifacts.

Preflight checks the NVIDIA GPU, Isaac launcher, packaged policy hashes, the
640-pixel perception certificate, current source bindings, and the exact command
contract used by the successful profile validation. It returns reasons if any
requirement is missing or stale. `ZGBS_ISAAC_PYTHON` selects a non-default launcher.

The mission uses the existing v7m130 capture and v19noised extraction policies.
Its module pose comes from RGB-D; velocity is zero before capture and derived
from wrist motion after capture. The existing lead-in guard controls insertion.
Final seating tolerances and the rack-only hold requirement are unchanged.
The loaded v13m130 insertion policy does not produce actions.

## Completion and outputs

The service independently requires all nine checks:

1. One relocation episode on the camera-driven task.
2. All seven seating conditions and supported settling passed.
3. Both the hand and robot-side latch released after seating.
4. Rack retention engaged after seating and held alone for at least 0.70 s.
5. Robot-carried transit stayed within the existing 2.5 mm / 52.36 mrad bounds.
6. Calibrated RGB-D detections were recorded, with final destination occupancy.
7. The visual source-occupied / destination-clear plan passed.
8. Controller labels match the phases that actually executed.
9. An MP4 recording was saved with a valid container signature.

The result is one simulation outcome. `qualification.passed` stays `null`;
a successful mission does not establish a reliability rate.

```text
<output>/jobs/<job-id>/
  job.json                         # request, command, source, input and output hashes
  events.jsonl                     # progress and lifecycle events
  artifacts/
    execution.log
    workflow_report.json           # simulator measurements
    mission_verification.json       # independent completion checks
    handoff_trace.npz
    video/rl-video-step-0.mp4
```

`verify` checks every listed artifact's size and SHA-256, rejects missing or
unbound required files, checks checkpoint identity and seed, and re-evaluates the
physical report. It works after copying the job directory to another machine.
It does not need Isaac or the original weights. These hashes detect changes
relative to `job.json`; they are not a digital signature against a party able
to replace the whole bundle. Video decoding and visual review remain separate
from the container and integrity checks.

## Verified runs

The [profile validation](../evidence/live_service_current_validation_seed6070.json)
ran from clean commit `5387cbf`. The
[normal application run](../evidence/live_service_application_seed6070_v1.json)
ran from clean commit `f7cdf23`, using the packaged checkpoints and service worker.
Both passed. The latter recorded 1,320/1,320 detections, 1.370 mm maximum transit
drift, and 0.733333 s of rack-only hold. Both use seed 6070, one environment and
stable lighting; they are not independent seeds or a new pooled certificate.

## Revalidate the profile

Use new output paths and commit source changes before collecting evidence.
The service remains unavailable while its source-bound evidence is stale.

```powershell
# A fresh static corpus; leave the process running until the NPZ is fully written.
C:/isaac-sim/python.bat scripts/collect_grapple_vision.py --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Collect-v0 --output datasets/new_service_holdout.npz --samples 1024 --num_envs 16 --seed 287 --rgb_source raw --pose_distribution workflow_envelope
C:/isaac-sim/python.bat scripts/certify_fiducial_perception.py --dataset datasets/new_service_holdout.npz --report artifacts/new_service_perception.json

# The exact proposed service recipe, recorded and independently checked.
python scripts/validate_live_service.py --output artifacts/new_profile_validation --seed 6070
```

The validator deliberately constructs the proposed command without admitting a
service job: its purpose is to produce the evidence required for admission. It
requires a clean tracked checkout and preserves failed outcomes. Promote only
passing reports, retain their source and dataset provenance, then rebuild the
evidence manifest. Changing the controller or guard requires a new comparison;
changing only checkpoint location requires identical file hashes.

## Optional local API

Install `.[service]`, then run `zero-g-service --host 127.0.0.1 --port 8000`.
The API accepts fixed presets and seeds, not arbitrary commands or uploaded code.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Worker and queue state |
| GET | `/api/capabilities` | Preset readiness and missing requirements |
| POST | `/api/jobs` | Submit `{"preset_id":"isaac_full_chain_perception","seed":6070}` |
| GET | `/api/jobs/{id}` | Job state and result |
| DELETE | `/api/jobs/{id}` | Cancel a queued or running job |
| GET | `/api/jobs/{id}/events` | Incremental progress events |
| GET | `/api/jobs/{id}/artifacts` | Files, sizes and hashes |
| GET | `/api/jobs/{id}/artifacts/{path}` | Download an artifact |

`replay_full_chain` exercises orchestration with synthetic events. It is explicitly
labelled replay and supplies no physical evidence.
