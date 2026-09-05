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


#: Paths whose modification cannot change what a run computed. A report being
#: written while its sibling reports are written in the same batch is the normal
#: case, not a provenance failure, and conflating the two makes ``dirty`` mean
#: nothing: five analysis reports generated in one command will all report a
#: dirty tree because each one's write dirties it for the next.
NON_EXECUTING_PREFIXES = ("evidence/", "docs/", "artifacts/")


def _is_code(path: str) -> bool:
    return not path.startswith(NON_EXECUTING_PREFIXES)


def git_source_revision(root: Path) -> dict[str, object]:
    """Return the commit and tracked-worktree state that generated an artifact.

    ``dirty`` is the whole tracked worktree. ``code_dirty`` is the part of it
    that could have changed the result: a run whose ``code_dirty`` is false was
    produced by exactly the committed source at ``commit``, whatever else in the
    tree had moved. That is the field to read when asking whether a report can
    be reproduced.
    """

    root = root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if commit.returncode != 0:
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "code_dirty": None,
            "tracked_changes": [],
            "error": commit.stderr.strip() or "git rev-parse failed",
        }
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    changes = [line for line in status.stdout.splitlines() if line.strip()]
    code_changes = [line for line in changes if _is_code(line[3:].strip())]
    return {
        "available": status.returncode == 0,
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() or None,
        "dirty": bool(changes) if status.returncode == 0 else None,
        "code_dirty": bool(code_changes) if status.returncode == 0 else None,
        "tracked_changes": changes,
        "error": None if status.returncode == 0 else status.stderr.strip() or "git status failed",
    }


__all__ = ["git_source_revision"]
