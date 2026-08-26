"""Local compute-service core with an optional HTTP adapter.

The configuration, runner, registry, and artifact store deliberately have no
FastAPI or Isaac imports. FastAPI is loaded only when :func:`create_app` is
requested; Isaac Sim remains isolated in a child process.
"""

from typing import Any

from .config import ServiceSettings

__all__ = ["ServiceSettings", "create_app"]


def __getattr__(name: str) -> Any:
    """Load the optional FastAPI adapter without coupling core imports to it."""

    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
