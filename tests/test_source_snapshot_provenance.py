"""Exact committed source recovery without normalizing a recorded hash away."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "snapshot_provenance_check", Path(__file__).resolve().parents[1] / "scripts/check_source_provenance.py",
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
SOURCE = "scripts/driver.py"
ARTIFACT = "evidence/source_snapshots/test/driver.py.raw"
MIXED = b"print('alpha')\r\nprint('beta')\n"
RAW_SHA = hashlib.sha256(MIXED).hexdigest()


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
        cwd=root, capture_output=True, check=True,
    ).stdout.decode().strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    (tmp_path / "scripts").mkdir()
    (tmp_path / SOURCE).write_bytes(MIXED.replace(b"\r\n", b"\n"))
    (tmp_path / ".gitattributes").write_text("*.py text eol=lf\n/evidence/source_snapshots/** -text\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Canonical source")
    source_commit = git(tmp_path, "rev-parse", "HEAD")
    registry = {"version": 1, "snapshots": [{
        "source_path": SOURCE, "sha256": RAW_SHA, "snapshot_path": ARTIFACT, "source_commit": source_commit,
    }]}
    (tmp_path / ARTIFACT).parent.mkdir(parents=True)
    (tmp_path / ARTIFACT).write_bytes(MIXED)
    (tmp_path / check.SNAPSHOT_REGISTRY).write_text(json.dumps(registry))
    monkeypatch.setattr(check, "ROOT", tmp_path)
    return tmp_path, registry, source_commit


def commit_snapshot(root):
    git(root, "add", "evidence")
    git(root, "commit", "-qm", "Preserve exact runtime source")


def test_committed_snapshot_recovers_exact_mixed_bytes_with_explicit_overlay_label(repository):
    root, _, source_commit = repository
    assert not check._matches((root / SOURCE).read_bytes(), RAW_SHA)
    commit_snapshot(root)
    result = check.classify(SOURCE, RAW_SHA, 20, True)
    assert result["state"] == "recovered"
    assert result["recovery"] == "source_snapshot_overlay"
    assert result["snapshot_path"] == ARTIFACT
    assert result["commit"] == source_commit
    assert result["snapshot_commit"] == git(root, "rev-parse", "HEAD")


def test_uncommitted_registry_and_artifact_cannot_establish_recovery(repository):
    root, _, _ = repository
    assert (root / ARTIFACT).is_file()
    assert check.classify(SOURCE, RAW_SHA, 20, True)["state"] == "lost"


def test_raw_artifact_survives_clone_and_overlay_preserves_canonical_git_blob(repository, tmp_path_factory):
    root, _, _ = repository
    commit_snapshot(root)
    clone = tmp_path_factory.mktemp("clone-parent") / "clone"
    git(root, "clone", "-q", "--no-hardlinks", str(root), str(clone))
    assert (clone / ARTIFACT).read_bytes() == MIXED
    assert hashlib.sha256((clone / SOURCE).read_bytes()).hexdigest() != RAW_SHA
    (clone / SOURCE).write_bytes((clone / ARTIFACT).read_bytes())
    assert git(clone, "hash-object", "--path", SOURCE, SOURCE) == git(clone, "rev-parse", f"HEAD:{SOURCE}")
    assert git(clone, "diff", "--", SOURCE) == ""
    # Git may keep a stale stat-based modified flag until the index is refreshed.
    # Adding the proven-identical normalized blob stages no source change.
    git(clone, "add", "--", SOURCE)
    assert git(clone, "diff", "--cached", "--name-only") == ""
    assert git(clone, "status", "--porcelain") == ""
    assert hashlib.sha256((clone / SOURCE).read_bytes()).hexdigest() == RAW_SHA


def test_changed_artifact_bytes_do_not_recover_a_hash(repository):
    root, _, _ = repository
    (root / ARTIFACT).write_bytes(MIXED + b"# changed\n")
    commit_snapshot(root)
    assert check.classify(SOURCE, RAW_SHA, 20, True)["state"] == "lost"


def test_snapshot_cannot_claim_equivalence_to_different_code(repository):
    root, registry, _ = repository
    (root / SOURCE).write_bytes(b"print('different source')\n")
    git(root, "add", SOURCE)
    git(root, "commit", "-qm", "Different program")
    registry["snapshots"][0]["source_commit"] = git(root, "rev-parse", "HEAD")
    (root / check.SNAPSHOT_REGISTRY).write_text(json.dumps(registry))
    commit_snapshot(root)
    assert check.classify(SOURCE, RAW_SHA, 20, True)["state"] == "lost"


def test_unreachable_source_commit_is_not_recovered(repository, monkeypatch):
    root, _, _ = repository
    commit_snapshot(root)
    monkeypatch.setattr(check, "_reachable", lambda *_args: False)
    assert check.classify(SOURCE, RAW_SHA, 20, True)["state"] == "lost"


@pytest.mark.parametrize("path", ["../driver.py", "/driver.py", "C:/driver.py", "scripts\\driver.py", "scripts/./driver.py"])
def test_registry_paths_cannot_escape_or_alias(repository, path):
    root, registry, _ = repository
    registry["snapshots"][0]["snapshot_path"] = path
    (root / check.SNAPSHOT_REGISTRY).write_text(json.dumps(registry))
    commit_snapshot(root)
    with pytest.raises(ValueError, match="paths|escape|location"):
        check.classify(SOURCE, RAW_SHA, 20, True)


def test_artifacts_must_be_in_the_explicit_snapshot_directory(repository):
    root, registry, _ = repository
    registry["snapshots"][0]["snapshot_path"] = "other/driver.py.raw"
    (root / check.SNAPSHOT_REGISTRY).write_text(json.dumps(registry))
    commit_snapshot(root)
    with pytest.raises(ValueError, match="location"):
        check.classify(SOURCE, RAW_SHA, 20, True)


def test_duplicate_registry_bindings_are_rejected(repository):
    root, registry, _ = repository
    registry["snapshots"].append(dict(registry["snapshots"][0]))
    (root / check.SNAPSHOT_REGISTRY).write_text(json.dumps(registry))
    commit_snapshot(root)
    with pytest.raises(ValueError, match="Duplicate"):
        check.classify(SOURCE, RAW_SHA, 20, True)


def test_normal_lf_and_crlf_recovery_is_unchanged(repository):
    root, _, _ = repository
    normal = (root / SOURCE).read_bytes()
    for rendering in (normal, normal.replace(b"\n", b"\r\n")):
        result = check.classify(SOURCE, hashlib.sha256(rendering).hexdigest(), 20, True)
        assert result["state"] == "recovered"
        assert "recovery" not in result
