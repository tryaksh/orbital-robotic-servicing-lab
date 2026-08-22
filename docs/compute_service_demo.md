# Compute Service Demo Runbook

This runbook presents the repository as a **simulation-backed robotic
serviceability qualification platform**. The reference problem is replacing a
modular compute unit with a six-axis arm in microgravity. The product claim is
smaller and more useful:

> Evaluate whether a module, capture interface, rack, camera layout, and robot
> workcell can be serviced, then preserve the outcome and the reason for it as
> reviewable evidence.

This is not a flight-qualified servicing system, a safety certificate, or proof
that autonomous orbital module replacement is solved.

## What the demo contains

```text
Browser dashboard / REST client
             |
             v
       local FastAPI service
             |
             v
    serialized GPU job worker
          /         \
synthetic replay   live Isaac subprocess
          \         /
             v
  events + result + hashed artifacts + provenance
```

The service deliberately exposes fixed presets rather than arbitrary commands,
paths, environment variables, or uploads. It runs one job at a time, persists
job and event state, supports cancellation, and serves only files contained in
that job's artifact directory. It binds to loopback by default because it has no
built-in authentication.

Two presets are available:

| Preset | Backend | What it proves | What it does not prove |
| --- | --- | --- | --- |
| `replay_full_chain` | Deterministic synthetic replay | The dashboard, API, queue, perception/planning/manipulation phase events, typed result schema, persistence, artifact download, and unknown-value handling work without Isaac or private weights | It performs no physics, perception inference, planning decision, or policy execution and creates no qualification evidence |
| `isaac_full_chain_perception` | Live Isaac Sim subprocess | The installed simulator, rendered RGB-D pose/occupancy estimator, fail-closed bay precondition, learned capture/extraction, physical payload-shuttle transfer, guarded insertion, telemetry parser, video/report export, and service boundary execute together | One seed is only one observed simulation outcome; it is not a hardware validation or a reliability estimate |

The live preset is offered only when the service can find Isaac Python, the
required current-workcell checkpoints, the workflow driver, a responsive NVIDIA
GPU, passed RGB-D perception evidence, passed full-chain evidence, and matching
SHA-256 hashes for the policies and runtime sources. The checked-in evidence now
passes this fail-closed gate; `GET /api/capabilities` reports the live preset as
available on the validated development machine.

## Fast path: run the replay

The replay path needs CPython 3.11 and the project package, but not Isaac Sim or
an NVIDIA GPU. Replace `python` with the explicit interpreter path when it is
not on `PATH`.

```powershell
python -m pip install -e .
python scripts\run_service_api.py --host 127.0.0.1 --port 8000
```

If the project is already installed in Isaac Sim's bundled Python, the same
service can be started with:

```powershell
C:\isaac-sim\python.bat scripts\run_service_api.py --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, choose **Full-chain telemetry replay**, keep the
default seed, and select **Start run**. The run should finish in a few
seconds and show the following labelled sequence:

1. perception;
2. planning precondition;
3. capture;
4. extraction;
5. transit;
6. insertion;
7. verification.

The synthetic replay result intentionally reports:

- `is_live_simulation: false`;
- `qualification.passed: null`;
- `qualification.trials: 0`;
- `planning.gate_passed: null` and no occupancy scores;
- unknown force, torque, confidence, and other values that were not measured.

A green service/job lifecycle is therefore not a robotics pass. It means the
compute path completed. The qualification panel should continue to say that no
qualification decision was produced.

## Live path: run the Isaac preset

The recorded development stack is Windows 11, Isaac Sim 5.1.0, Isaac Lab
v2.3.2, and an NVIDIA GPU. Complete the repository's installation and simulator
validation steps first, then launch the same service with the Isaac launcher:

`w65` in the preset title identifies the current workcell with the robot base at
x = −0.65 m; it is not a claim level or software version.

```powershell
C:\isaac-sim\python.bat scripts\validate_sim.py
C:\isaac-sim\python.bat scripts\run_service_api.py `
  --host 127.0.0.1 `
  --port 8000 `
  --isaac-python C:\isaac-sim\python.bat
```

Open <http://127.0.0.1:8000> and inspect **Capabilities**. Select
**Live RGB-D compute-module service run** when it is marked available. The
service runs one headless, 3,600-step two-bay relocation attempt with randomized
rendered conditions and the 384x384 RGB-D camera. The validated machine has
completed this path through the API and produced a report, trace, events,
provenance hashes, and a 66.7 second video.

The service distinguishes four different ideas:

- `status: succeeded` means the backend returned a valid result whose terminal
  workflow predicate completed;
- `result.completed` is the outcome of this one workflow attempt;
- `result.qualification.passed` remains `null` for a single live attempt because
  no declared sample count and statistical threshold were evaluated;
- `result.qualification.trials` is `1`, while success rate and threshold remain
  unset because the service does not turn one outcome into a reliability claim.

If the workflow exits cleanly but does not satisfy its terminal predicate, the
job is marked failed and retains its logs/report/video for diagnosis. Do not
edit the dashboard or JSON to turn that negative result into a pass.

The live preset requests one fixed plan: bay 0 to bay 1. Its initial RGB
occupancy scores must indicate that the requested source is occupied and the
destination is clear before manipulation proceeds. A missing or contradictory
estimate fails closed. The typed `result.planning` object preserves the request,
two decision scores, threshold, and gate outcome separately from statistical
qualification. These sigmoid scores are decision values, not calibrated
confidence, and the gate validates a fixed plan rather than selecting arbitrary
bays.

## API walkthrough from PowerShell

Keep the service running in one terminal. In a second terminal:

```powershell
$serviceUri = 'http://127.0.0.1:8000'

Invoke-RestMethod "$serviceUri/api/health"
Invoke-RestMethod "$serviceUri/api/capabilities"

$requestBody = @{
  preset_id = 'replay_full_chain'
  seed = 4070
} | ConvertTo-Json

$submitted = Invoke-RestMethod `
  -Method Post `
  -Uri "$serviceUri/api/jobs" `
  -ContentType 'application/json' `
  -Body $requestBody

$jobId = $submitted.id
do {
  Start-Sleep -Milliseconds 500
  $job = Invoke-RestMethod "$serviceUri/api/jobs/$jobId"
  $job | Select-Object id,status,current_stage,progress
} while ($job.status -in @('queued', 'running'))

$events = Invoke-RestMethod "$serviceUri/api/jobs/$jobId/events"
$artifacts = Invoke-RestMethod "$serviceUri/api/jobs/$jobId/artifacts"
$job.result | ConvertTo-Json -Depth 8
$events.events | Select-Object seq,type,stage,message
$artifacts.artifacts | Select-Object path,size_bytes,sha256
```

Download the replay summary:

```powershell
Invoke-WebRequest `
  "$serviceUri/api/jobs/$jobId/artifacts/summary.json" `
  -OutFile "artifacts\service_summary_$jobId.json"
```

To submit the live preset, change only `preset_id`:

```powershell
$requestBody = @{
  preset_id = 'isaac_full_chain_perception'
  seed = 4070
} | ConvertTo-Json
```

Cancel a queued or active job with:

```powershell
Invoke-RestMethod -Method Delete "$serviceUri/api/jobs/$jobId"
```

The public API surface is:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Worker state, queue depth, and active job |
| `GET` | `/api/capabilities` | Presets plus GPU/Isaac availability and missing requirements |
| `POST` | `/api/jobs` | Submit `{ "preset_id": "...", "seed": 4070 }` |
| `GET` | `/api/jobs` | List recent jobs; optional `status` and `limit` query parameters |
| `GET` | `/api/jobs/{id}` | Poll durable job state and the terminal result |
| `DELETE` | `/api/jobs/{id}` | Request cancellation |
| `GET` | `/api/jobs/{id}/events` | Incremental events; use `after` for polling |
| `GET` | `/api/jobs/{id}/artifacts` | List artifact metadata and SHA-256 digests |
| `GET` | `/api/jobs/{id}/artifacts/{path}` | Download one job-scoped artifact |
| `GET` | `/docs` | Interactive OpenAPI documentation |

## Artifact and evidence map

The default runtime root is `artifacts/service_runtime`. Every job receives an
isolated directory:

```text
artifacts/service_runtime/jobs/<job-id>/
|-- job.json
|-- events.jsonl
`-- artifacts/
    |-- summary.json             # replay
    |-- telemetry.jsonl          # replay
    |-- execution.log            # live
    |-- workflow_report.json     # live, when emitted
    `-- video/...                # live, when emitted
```

`job.json` records the fixed preset revision, source revision/dirty state when
Git is available, backend, seed, exact live command, input paths, input SHA-256
values, terminal result, and artifact hashes. The runtime store is evidence
produced by a demo job. The curated files under `evidence/` are separate,
multi-run research
measurements and must not be presented as if the replay regenerated them.

Current workcell evidence is:

| Capability | Measured result | Gate | Evidence |
| --- | ---: | --- | --- |
| Full relocation, state feedback | 16/16, **100%** | Settled success; Wilson 95% interval 80.64%-100% | [`full_chain_state_16_report.json`](../evidence/full_chain_state_16_report.json) |
| Full relocation, rendered RGB-D | **1 settled end-to-end run** | Passed as a single outcome, not a reliability rate | [`full_chain_rgbd_service_seed4070.json`](../evidence/full_chain_rgbd_service_seed4070.json) |
| RGB-D pose holdout | 1.682 mm position p95; 0.0121 rad orientation p95 | Passed declared gates on 1,024 rendered frames | [`fiducial_rgbd_service_plate.json`](../evidence/fiducial_rgbd_service_plate.json) |
| RGB-D availability | 94.824% overall; 99.854% at critical rack poses | Passed 90%/99% gates | [`fiducial_rgbd_service_plate.json`](../evidence/fiducial_rgbd_service_plate.json) |
| Two-bay occupancy | **100% exact match** | Passed 95% gate | [`fiducial_rgbd_service_plate.json`](../evidence/fiducial_rgbd_service_plate.json) |

Earlier 0/64 relocation and weak learned-insertion results remain useful audit
history. They caused the mechanical pivot to a physical payload shuttle and a
receiving interface designed for robotic servicing.

## Perception and full-chain boundaries

The deployment observation path uses a calibrated RGB-D fiducial estimator and
reuses its filtered pose and occupancy estimate across the workflow. It accepts
RGB, registered metric depth, camera intrinsics, and robot proprioception; it
does not consume the simulator module pose. Simulator truth is retained only to
score diagnostic error in evidence.

That implementation boundary is not the same as a qualified capability:

- the deployed 384x384 RGB-D estimator passed its 1,024-frame rendered holdout;
- learned policies execute capture and extraction; the insert checkpoint is
  loaded for provenance but the successful transfer uses guarded physical-stage
  insertion because the learned two-bay insert policy failed its earlier gate;
- workflow transition and terminal seating predicates inspect simulator state
  as supervisory safety and scoring logic, not policy observations;
- source and destination bays are fixed by the preset; estimator occupancy
  scores gate that requested plan but do not choose a plan autonomously, and
  the sigmoid decision scores are explicitly not calibrated confidence;
- the payload shuttle is a simulated force-driven D6 stage, not flight hardware;
- no connector mating, cable/fluid handling, real camera calibration, real
  UR10e, hardware-in-the-loop, or Sim2Real validation is present;
- one live run cannot establish a success rate, Wilson interval, or
  qualification decision.

Historical 64x64 perception and camera/oracle/blind numbers remain useful
research context, but they predate the current workcell and overview camera and
must be labelled as legacy evidence. They are not the accuracy of the current
live preset.

## Five-minute interview demo

Use this script instead of claiming a solved autonomous swap.

**0:00-0:40 — problem.** “Robot projects often start with policy training and
discover too late that the module, rack, camera, or arm placement is physically
unserviceable. I built a qualification layer that makes those failures visible
before hardware integration.”

**0:40-1:20 — product boundary.** Show the dashboard and capabilities panel.
Point out the fixed replay/live presets, serialized GPU worker, local-only
default, explicit simulation boundary, and fail-closed evidence gate.

**1:20-2:20 — observable job.** Start `replay_full_chain`. Follow perception,
planning, capture, extraction, transit, insertion, and verification events.
Open `summary.json` and show `is_live_simulation: false`,
`planning.gate_passed: null`, and `qualification.passed: null`. Explain that
this proves the service delivery path, not robotics.

**2:20-3:30 — engineering evidence.** Show the baseline table. Lead with the
16/16 settled state-feedback chain, the 1,024-frame RGB-D gate, and the complete
single rendered-perception run. Then show the earlier 0/64 result and explain
how failure attribution drove the payload-shuttle and rack-interface redesign. This is the platform's
value—failure attribution changes interface and cell design decisions.

**3:30-4:30 — perception and execution.** Show the local rendered overview frame
from the retained live artifact. Show that the live capability is available,
then run it or open the retained report, trace, event stream, and video. Explain
that capture/extraction are learned, transfer/insertion are guarded physical
stage control, observations use RGB-D estimates, and supervisory scoring still
uses simulator truth.

**4:30-5:00 — close.** “The current deliverable is not orbital autonomy. It is a
reproducible way to answer whether a robotic service design is ready to proceed,
with positive and negative evidence. The next promotion gate is more randomized
RGB-D full-chain trials, followed by
hardware-in-the-loop.”

## Presenter checklist

Before an interview or portfolio recording:

- run `GET /api/capabilities` and do not promise a live run when it is marked
  unavailable;
- pre-warm the NVIDIA asset cache and validate Isaac on the presentation
  machine;
- run the replay once from a clean runtime directory and inspect its artifacts;
- keep the service on `127.0.0.1`; do not use `--allow-remote` on an untrusted
  network;
- say “simulation outcome” for a single live run and reserve “qualification”
  for an aggregate protocol with a declared gate;
- keep failing live artifacts—they are often the most useful evidence in a
  design review;
- never describe the replay's labelled reference metrics as newly measured.

For the independent pursue/pivot decision and product thesis, see
[`independent_assessment.md`](independent_assessment.md). For evidence-level
claim boundaries, see [`claim_vs_evidence.md`](claim_vs_evidence.md).
