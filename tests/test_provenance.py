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
        "code_dirty": True,
        "tracked_changes": [" M scripts/run.py"],
        "error": None,
    }


def test_evidence_writes_do_not_count_as_uncommitted_code(monkeypatch, tmp_path: Path) -> None:
    """Five reports written by one command all dirty the tree for each other.

    That is the normal case and it must not read as an unreproducible run, or
    `dirty` stops distinguishing the thing it exists to flag.
    """

    responses = {
        ("rev-parse", "HEAD"): subprocess.CompletedProcess([], 0, "b" * 40 + "\n", ""),
        ("branch", "--show-current"): subprocess.CompletedProcess([], 0, "topic\n", ""),
        ("status", "--porcelain=v1", "--untracked-files=no"): subprocess.CompletedProcess(
            [], 0, " M evidence/a.json\n M evidence/b.json\n M docs/NOW.md\n", ""
        ),
    }
    monkeypatch.setattr(
        "zero_g_blade_swap.provenance._git", lambda _root, *args: responses[args]
    )
    result = git_source_revision(tmp_path)
    assert result["dirty"] is True
    assert result["code_dirty"] is False


def test_a_modified_script_is_still_uncommitted_code(monkeypatch, tmp_path: Path) -> None:
    responses = {
        ("rev-parse", "HEAD"): subprocess.CompletedProcess([], 0, "c" * 40 + "\n", ""),
        ("branch", "--show-current"): subprocess.CompletedProcess([], 0, "topic\n", ""),
        ("status", "--porcelain=v1", "--untracked-files=no"): subprocess.CompletedProcess(
            [], 0, " M evidence/a.json\n M src/zero_g_blade_swap/evaluation.py\n", ""
        ),
    }
    monkeypatch.setattr(
        "zero_g_blade_swap.provenance._git", lambda _root, *args: responses[args]
    )
    assert git_source_revision(tmp_path)["code_dirty"] is True


def test_git_source_revision_fails_closed_outside_git(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "zero_g_blade_swap.provenance._git",
        lambda *_args: subprocess.CompletedProcess([], 128, "", "not a repository"),
    )
    result = git_source_revision(tmp_path)
    assert result["available"] is False
    assert result["commit"] is None
    assert result["dirty"] is None
