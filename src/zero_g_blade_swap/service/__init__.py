"""Local compute service for the zero-g servicing showcase.

The service package deliberately has no Isaac imports.  Isaac Sim is isolated in
one child process and jobs are serialized by :class:`JobManager`.
"""

from .app import create_app
from .config import ServiceSettings

__all__ = ["ServiceSettings", "create_app"]
