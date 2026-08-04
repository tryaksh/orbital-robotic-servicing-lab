# Validation Ledger

This page records only checks that produced an artifact in the development
workspace on 2026-08-04. It intentionally separates runtime smoke evidence from
throughput, learning convergence, and real-world transfer. Local JSON and
training outputs are ignored by Git, so the values below are the durable public
summary; rerun the commands on the release commit before publishing a tag.

## Evidence snapshot

| Evidence | Status | What was actually observed | What it does not prove |
| --- | --- | --- | --- |
| `artifacts/sim_validation.json` | Passed | Isaac Sim `SimulationApp` launched headlessly; bundled Python 3.11.13, Torch 2.7.0+cu128, Warp 1.8.2, CUDA available on an RTX 5070 Ti Laptop GPU | Isaac Lab task construction or asset availability |
| `benchmarks/clone_smoke.json` | Passed | Teacher task ran with 8 cloned environments; 85.0485 environment-steps/s, 10.6311 policy/simulation steps/s, 659 MiB observed GPU memory, 8.4 MiB Torch peak allocated and 22.0 MiB reserved | The 1024-environment target, long-duration stability, representative trained-policy throughput, or a complete benchmark matrix |
| `artifacts/smoke_report.json` | Passed | Vision task ran 8 environments for 8 steps with seed 42; RGB shape `(8,64,64,3)`, proprio `(8,47)`, critic `(8,110)`, diagnostic blade pose `(8,14)`, no episode termination | Vision PPO convergence, 128-environment capacity, or camera calibration against flight hardware |
| `benchmarks/teacher_sustained.json` | Passed | 1024 teacher environments survived 200 warm-up and 500 measured steps at 7,378.90 environment-steps/s; 1,037 MiB observed total GPU memory | PPO convergence or the same capacity with other GPU workloads open |
| `benchmarks/vision_sustained.json` | Passed | 256 vision environments survived 200 warm-up and 500 measured steps at 1,597.65 environment-steps/s; 2,266 MiB observed total GPU memory | The extra memory and throughput cost of a long vision PPO run |
| `logs/rl_games/.../smoke_teacher` and `smoke_vision` | Passed as integration | RL-Games produced teacher and vision checkpoints after two PPO epochs | Their names contain `rew_-inf` because no 45-second episode completed in two epochs; no success rate or convergence was demonstrated |
| `artifacts/play_teacher_report.json` and `play_vision_report.json` | Passed | Each saved checkpoint restored and generated deterministic actions for all 16 requested steps in four environments | Useful manipulation behavior or full-swap success |

The vision artifact additionally measured RGB range `[0,1]`, standard deviation
`0.31303`, repeated-frame radiation-noise delta standard deviation `0.02469`,
dark-background fraction `0.36191`, per-environment mean standard deviation
`0.01609`, and 1,749 MiB observed GPU memory.

These artifacts were produced during development, not by CI. They are component
evidence rather than a release certificate. In particular, the four-stage
success-rate curriculum requires long training before promotion can be observed.

## Reproduction gates

Run from the repository root in PowerShell. Close other Isaac/Kit processes so
VRAM measurements are meaningful.

```powershell
# Lightweight unit, integration-contract, and configuration checks
C:\isaac-sim\python.bat -m pytest -m "not isaac and not camera and not benchmark"
C:\isaac-sim\python.bat -m ruff check src scripts tests

# Standalone simulator and CUDA gate
C:\isaac-sim\python.bat scripts\validate_sim.py --output artifacts\sim_validation.json

# Runtime task contracts
C:\isaac-sim\python.bat scripts\smoke_env.py --profile teacher --teacher_steps 100 `
  --output artifacts\smoke_teacher.json
C:\isaac-sim\python.bat scripts\smoke_env.py --profile vision --vision_steps 32 `
  --output artifacts\smoke_vision.json

# Descending first-fit VRAM/throughput search
C:\isaac-sim\python.bat scripts\benchmark.py --profile all --quick `
  --output benchmarks\quick.json
C:\isaac-sim\python.bat scripts\benchmark.py --profile all `
  --output benchmarks\full.json

# RL-Games integration only; this is not a learning benchmark
C:\isaac-sim\python.bat scripts\train.py --task Isaac-ZeroG-BladeSwap-Teacher-v0 `
  --num_envs 8 --smoke --run_name smoke_teacher
```

## Release acceptance criteria

A portfolio release should not claim the full result until all of these are
attached to a single commit:

- Static/unit checks pass in GitHub Actions.
- Fresh teacher and vision smoke JSON report `ok: true` with no non-finite
  observations or rewards.
- A complete descending benchmark records the selected teacher and vision
  counts within the 10,752 MiB project budget.
- At least three teacher seeds report learning curves and full-swap success,
  with evaluation separated from training randomization.
- Teacher checkpoint playback succeeds for a fixed step budget, then a
  demonstration dataset and behavioral-cloning checkpoint are generated.
- At least three vision-policy seeds report full-swap success under held-out
  physics and visual distributions.
- A rendered video shows the entire eight-phase exchange, not only insertion.
- Real or hardware-in-the-loop tests record pose/calibration error, success
  rate, insertion force, cycle time, and failure modes. Until then, label the
  work "Sim2Real-ready methodology," not "Sim2Real validated."

## External asset dependency

The canonical UR10e/Robotiq USD is referenced through Isaac Lab's NVIDIA asset
root. The first cold-cache launch initially failed while network access was
restricted; after the official asset populated the retained NVIDIA cache, the
teacher, vision, PPO, playback, and sustained benchmarks all passed. A fresh
machine must be online for that first asset resolution.
