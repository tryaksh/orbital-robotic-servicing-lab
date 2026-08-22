# Compute Service Quick Reference

The local service provides a browser dashboard and a fixed, auditable API for
running the simulation workflow. It binds to loopback by default and has no
built-in authentication.

## Start

```powershell
C:\isaac-sim\python.bat scripts\run_service_api.py `
  --host 127.0.0.1 `
  --port 8000 `
  --isaac-python C:\isaac-sim\python.bat
```

Open `http://127.0.0.1:8000`.

## Presets

| Preset | Use |
| --- | --- |
| `replay_full_chain` | Synthetic service/API demonstration without Isaac physics |
| `isaac_full_chain_perception` | Live RGB-D Isaac run |

Important: the live preset currently enables `--base_rail_on_relocation`, which
hands the module to a hidden world-mounted D6 payload stage. It is a retained
baseline, not the final robot-carried demonstration. The work described in
`docs/claude_opus_5_handoff.md` must replace that default.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Worker and queue state |
| `GET` | `/api/capabilities` | Preset and GPU/Isaac readiness |
| `POST` | `/api/jobs` | Submit `{ "preset_id": "...", "seed": 4070 }` |
| `GET` | `/api/jobs` | List recent jobs |
| `GET` | `/api/jobs/{id}` | Poll one job |
| `DELETE` | `/api/jobs/{id}` | Cancel a queued or running job |
| `GET` | `/api/jobs/{id}/events` | Incremental phase events |
| `GET` | `/api/jobs/{id}/artifacts` | Artifact names, sizes, and hashes |
| `GET` | `/api/jobs/{id}/artifacts/{path}` | Download one job artifact |

The service accepts fixed presets only. It does not accept arbitrary commands,
paths, environment variables, or uploads. Remote binding requires the explicit
`--allow-remote` option and should not be used on an untrusted network.

## Live artifacts

Each job writes under:

```text
artifacts/service_runtime/jobs/<job-id>/
|-- job.json
|-- events.jsonl
`-- artifacts/
    |-- execution.log
    |-- handoff_trace.npz
    |-- workflow_report.json
    `-- video/rl-video-step-0.mp4
```

`job.json` records the preset revision, seed, exact fixed command, Git state,
input SHA-256 values, terminal result, and output artifact hashes.

## Result meanings

- `status: succeeded` means the single workflow execution completed its terminal
  predicate.
- `result.completed` is one simulation outcome.
- `result.qualification.passed` remains `null` for a single live run.
- The replay never produces physical or perception qualification evidence.
- Payload-stage baseline results must not be relabelled as robot-carried results.

## Validation commands

```powershell
.venv\Scripts\python.exe -m pytest tests\test_service_api.py tests\test_service_core.py -q
C:\isaac-sim\python.bat -m ruff check src\zero_g_blade_swap\service tests\test_service_api.py tests\test_service_core.py
node --check src\zero_g_blade_swap\service\static\app.js
```
