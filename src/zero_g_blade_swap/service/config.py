"""Configuration for the local compute service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_project_root() -> Path:
    # config.py -> service -> zero_g_blade_swap -> src -> repository
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Service settings with conservative, local-only defaults."""

    project_root: Path
    runtime_dir: Path
    static_dir: Path
    isaac_python: Path
    host: str = "127.0.0.1"
    port: int = 8000
    cancel_grace_s: float = 8.0
    replay_step_delay_s: float = 0.35
    allow_remote: bool = False

    @classmethod
    def from_env(cls) -> ServiceSettings:
        project_root = Path(os.environ.get("ZGBS_PROJECT_ROOT", _default_project_root())).resolve()
        runtime_dir = Path(os.environ.get("ZGBS_RUNTIME_DIR", project_root / "artifacts" / "service_runtime")).resolve()
        static_dir = Path(
            os.environ.get(
                "ZGBS_STATIC_DIR",
                project_root / "src" / "zero_g_blade_swap" / "service" / "static",
            )
        ).resolve()
        default_isaac = Path("C:/isaac-sim/python.bat") if os.name == "nt" else Path("/isaac-sim/python.sh")
        isaac_python = Path(os.environ.get("ZGBS_ISAAC_PYTHON", default_isaac)).resolve()
        return cls(
            project_root=project_root,
            runtime_dir=runtime_dir,
            static_dir=static_dir,
            isaac_python=isaac_python,
            host=os.environ.get("ZGBS_HOST", "127.0.0.1"),
            port=int(os.environ.get("ZGBS_PORT", "8000")),
            cancel_grace_s=float(os.environ.get("ZGBS_CANCEL_GRACE_S", "8")),
            replay_step_delay_s=float(os.environ.get("ZGBS_REPLAY_STEP_DELAY_S", "0.35")),
            allow_remote=os.environ.get("ZGBS_ALLOW_REMOTE", "0").lower() in {"1", "true", "yes"},
        )
