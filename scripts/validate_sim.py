"""Launch Isaac Sim headlessly and record a machine-readable readiness marker.

This script must be executed with ``C:\isaac-sim\python.bat``.  It deliberately
does not import Isaac Lab: the cleanup gate proves that the retained simulator
works before the obsolete Lab installation or installer archive is removed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sim_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    # SimulationApp inspects ``sys.argv`` itself.  Remove this script's private
    # flag so Kit never receives an unrelated ``--output`` option.
    sys.argv = [sys.argv[0]]
    payload: dict[str, object] = {
        "success": False,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        app.update()

        import torch
        import warp as wp

        payload.update(
            {
                "success": True,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "warp": wp.__version__,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("SIMULATION_APP_VALIDATED", flush=True)
        return 0
    except Exception as exc:  # pragma: no cover - requires an Isaac runtime
        payload["error"] = f"{type(exc).__name__}: {exc}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
