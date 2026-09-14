# Installation

Use ordinary Python for the design tools, mission verification, and CPU tests.
Live simulation runs through Isaac Sim's own Python launcher.

## CPU tools

Python 3.11 is required by the package. From the repository root:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -e ".[dev]"
python scripts/derive_rack_requirement.py
pytest -m "not isaac and not camera and not benchmark"
```

The base install, `pip install -e .`, is enough for the geometry library and
`zero-g-mission`. The `dev` extra adds test and API tools. OpenCV-dependent tests
skip when OpenCV is absent. Tests marked `isaac`, `camera`, and `benchmark` require
additional runtime capabilities; the command above explicitly excludes them.

## Live simulation

The measured platform is Windows 11 with an RTX 5070 Ti Laptop GPU (12 GB).
Linux has not been validated. The recorded environment uses:

| Component | Recorded version |
| --- | --- |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.2, commit `37ddf62` |
| Python | 3.11, bundled with Isaac Sim |
| RL-Games | commit `6b3534f` |
| NVIDIA driver | 592.01 in the reference environment |

[Reference environment](../environment-lock.example.json) contains the full
record. Install Isaac Sim, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
# If Isaac Sim is elsewhere:
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -IsaacSimRoot D:/isaac-sim
C:/isaac-sim/python.bat scripts/smoke_env.py --headless
```

The setup script installs pinned Isaac Lab and RL-Games packages into the
simulator interpreter, installs this project, and records a local environment
lock. `-SkipInstall` reruns its checks without reinstalling packages. Enabling
Windows long paths may require an elevated PowerShell.

The three checkpoints used by the mission application are included in
[policies/servicing_v2](../policies/servicing_v2). Full training runs and other
research checkpoints are not included. Preflight verifies the included files
against the successful report before a GPU run starts.

```powershell
# From the ordinary Python environment installed above:
zero-g-mission preflight
zero-g-mission run --seed 6070
# Or without relying on a console-script PATH entry:
python -m zero_g_blade_swap.service.mission_cli preflight
```

Set `ZGBS_ISAAC_PYTHON` to the full launcher path if it differs from
`C:/isaac-sim/python.bat`. [Mission guide](compute_service_demo.md) covers outputs
and verification. Legacy campaign `.sh` scripts require Git Bash; they use Bash
features and are not portable POSIX `sh` scripts.

## Development checks

```powershell
ruff check src scripts tests
pytest -m "not isaac and not camera and not benchmark"
python scripts/check_criterion_currency.py
python scripts/check_source_provenance.py --depth 200
python scripts/build_evidence_manifest.py --check
python scripts/build_script_index.py --check
```

Use bare `pytest`, as CI does. `h5py==3.13.0` matches the HDF5 DLL bundled with
Isaac Sim 5.1 on Windows. Avoid installing a second Python stack over the
simulator's dependencies. For long runs, system RAM can be a tighter limit than
GPU memory; the measured workstation has 32 GB.
