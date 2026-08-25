# Installation

This project runs inside Isaac Sim's own Python interpreter. It is not a
`pip install` into a virtual environment: Isaac Lab and RL-Games have to be
installed against the simulator's interpreter, and the pinned versions matter.

Developed and measured on Windows 11 with an RTX 5070 Ti Laptop (12 GB). Linux is
untested here.

## What you need first

| | Version | Note |
| --- | --- | --- |
| NVIDIA Isaac Sim | 5.1.0 | Installed at `C:\isaac-sim` by default |
| Python | 3.11 | The one bundled with Isaac Sim |
| Isaac Lab | v2.3.2, commit `37ddf62` | Cloned by the setup script into `.deps/` |
| RL-Games | commit `6b3534f` | Pinned; installed by the setup script |
| NVIDIA driver | 592.01 or newer | RTX GPU with at least 12 GB for training |
| Git | any recent | Windows long paths must be available |

`environment-lock.example.json` records the exact set the published measurements
were taken with. `scripts/write_environment_lock.py` writes
`environment-lock.local.json` for your machine so the two can be compared.

## Setup

From an ordinary PowerShell prompt at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

It will:

1. enable Windows long paths (needs an elevated shell; it warns and continues if
   not) and set `core.longpaths` on the repository;
2. clone Isaac Lab v2.3.2 into `.deps/IsaacLab` and verify the commit;
3. create a junction from `.deps/IsaacLab/_isaac_sim` to your Isaac Sim install;
4. install `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl` and the
   pinned RL-Games into Isaac Sim's Python;
5. install this project in editable mode with its dev extras;
6. import-check every package and write `environment-lock.local.json`.

Useful flags:

```powershell
# Isaac Sim somewhere other than C:\isaac-sim
powershell -File scripts\setup_windows.ps1 -IsaacSimRoot D:\isaac-sim

# Re-run the checks without reinstalling packages
powershell -File scripts\setup_windows.ps1 -SkipInstall
```

## Verify

```powershell
C:\isaac-sim\python.bat scripts\smoke_env.py --headless
```

Then the geometry check, which needs no simulator and should take about a second:

```powershell
python scripts\check_workcell_geometry.py
```

## Running things

Anything that touches Isaac Sim goes through `C:\isaac-sim\python.bat`. The
CPU-only tools — the geometry checks, the report generators, the tests — run on
an ordinary Python 3.11.

```powershell
# The whole servicing job, one environment, end to end
bash scripts/run_robot_carried.sh rail

# The pooled certification: 32 environments on each of three held-out seeds
bash scripts/run_robot_carried.sh certify

# The local service and browser dashboard
C:\isaac-sim\python.bat scripts\run_service_api.py --host 127.0.0.1 --port 8000
```

The shell scripts are POSIX `sh` and expect Git Bash or WSL. They call
`C:/isaac-sim/python.bat` internally.

## Tests

Most tests are contract tests over source and evidence and need neither a GPU nor
a simulator:

```powershell
python -m pytest tests -q
```

Two modules need extra packages: `tests/test_fiducial.py` needs OpenCV and
`tests/test_pose_head.py` needs PyTorch. Tests marked `isaac`, `camera` or
`benchmark` need the simulator and are excluded by default.

## Known friction

- **Long paths.** Isaac Lab's checkout is deep enough to exceed the legacy
  Windows limit. The setup script handles it, but the first run needs an elevated
  shell to set the registry key.
- **Conda.** Isaac Lab's own `isaaclab.bat` prefers an inherited `CONDA_PREFIX`
  and installs a much larger optional set. The setup script clears those variables
  and installs the essential packages directly against the simulator's Python.
- **`flatdict`.** Its isolated build is broken against modern setuptools, so it is
  installed with `--no-build-isolation` after pinning `setuptools<82`.
- **h5py.** Pinned to 3.13.0 to match the HDF5 1.14.6 DLL that ships with
  Isaac Sim 5.1 on Windows.
- **Training memory.** A 1024-environment PPO run fits in 12 GB alongside a small
  evaluation process. Two full training runs at once do not.
