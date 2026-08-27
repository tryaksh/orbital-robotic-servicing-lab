"""Small, dependency-free provenance records for generated evidence."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def git_source_revision(root: Path) -> dict[str, object]:
    """Return the commit and tracked-worktree state that generated an artifact."""

    root = root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if commit.returncode != 0:
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "tracked_changes": [],
            "error": commit.stderr.strip() or "git rev-parse failed",
        }
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    changes = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "available": status.returncode == 0,
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() or None,
        "dirty": bool(changes) if status.returncode == 0 else None,
        "tracked_changes": changes,
        "error": None if status.returncode == 0 else status.stderr.strip() or "git status failed",
    }


__all__ = ["git_source_revision"]
