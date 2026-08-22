"""Command-line entry point for the compute API."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import uvicorn

from .app import create_app
from .config import ServiceSettings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Zero-G Blade Swap compute service.")
    parser.add_argument("--host", help="Bind address (default: ZGBS_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="TCP port (default: ZGBS_PORT or 8000)")
    parser.add_argument("--runtime-dir", type=Path, help="Only directory the service writes")
    parser.add_argument("--isaac-python", type=Path, help="Isaac Sim python.bat/python.sh launcher")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit a non-loopback bind (there is no built-in authentication)",
    )
    parser.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug"), default="info")
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = ServiceSettings.from_env()
    overrides = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise SystemExit("--port must be between 1 and 65535")
        overrides["port"] = args.port
    if args.runtime_dir is not None:
        overrides["runtime_dir"] = args.runtime_dir.resolve()
    if args.isaac_python is not None:
        overrides["isaac_python"] = args.isaac_python.resolve()
    if args.allow_remote:
        overrides["allow_remote"] = True
    settings = replace(settings, **overrides)
    if settings.host not in {"127.0.0.1", "localhost", "::1"} and not settings.allow_remote:
        raise SystemExit("Refusing an unauthenticated non-loopback bind; pass --allow-remote to acknowledge the risk")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
