"""Write local Isaac stack versions without importing Isaac Lab before Kit starts."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_LAB_ROOT = PROJECT_ROOT / ".deps" / "IsaacLab"


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _distribution(name: str) -> dict[str, object | None]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "direct_url": None}
    direct_url_path = Path(dist._path) / "direct_url.json"  # noqa: SLF001
    direct_url = json.loads(direct_url_path.read_text(encoding="utf-8")) if direct_url_path.exists() else None
    return {"version": dist.version, "direct_url": direct_url}


def main() -> int:
    import h5py
    import numpy
    import torch

    sim_root = Path(os.environ.get("ISAACSIM_PATH", "C:\\isaac-sim"))
    sim_version_file = sim_root / "VERSION"

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "isaac_sim": {
            "root": str(sim_root),
            "build": sim_version_file.read_text(encoding="utf-8").strip() if sim_version_file.exists() else None,
        },
        "isaac_lab": {
            "requested_tag": "v2.3.2",
            "expected_commit": "37ddf626871758333d6ed89cf64ad702aef127d0",
            "resolved_commit": _command(
                "git",
                "-c",
                f"safe.directory={ISAAC_LAB_ROOT.as_posix()}",
                "-C",
                str(ISAAC_LAB_ROOT),
                "rev-parse",
                "HEAD",
            ),
        },
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": _command(
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ),
        "rl_games": _distribution("rl-games"),
        "packages": {
            name: _distribution(name)["version"]
            for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")
        },
    }
    output = PROJECT_ROOT / "environment-lock.local.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
