from __future__ import annotations

import subprocess
from pathlib import Path

from zero_g_blade_swap.provenance import git_source_revision


def test_git_source_revision_records_commit_branch_and_dirty_state(monkeypatch, tmp_path: Path) -> None:
    responses = {
        ("rev-parse", "HEAD"): subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
        ("branch", "--show-current"): subprocess.CompletedProcess([], 0, "topic\n", ""),
        ("status", "--porcelain=v1", "--untracked-files=no"): subprocess.CompletedProcess(
            [], 0, " M scripts/run.py\n", ""
        ),
    }

    def fake_git(_root: Path, *args: str):
        return responses[args]

    monkeypatch.setattr("zero_g_blade_swap.provenance._git", fake_git)
    assert git_source_revision(tmp_path) == {
        "available": True,
        "commit": "a" * 40,
        "branch": "topic",
        "dirty": True,
        "tracked_changes": [" M scripts/run.py"],
        "error": None,
    }


def test_git_source_revision_fails_closed_outside_git(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "zero_g_blade_swap.provenance._git",
        lambda *_args: subprocess.CompletedProcess([], 128, "", "not a repository"),
    )
    result = git_source_revision(tmp_path)
    assert result["available"] is False
    assert result["commit"] is None
    assert result["dirty"] is None
