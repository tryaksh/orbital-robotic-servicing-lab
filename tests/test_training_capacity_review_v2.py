import hashlib
import json
import zipfile

import pytest

from scripts.review_training_capacity_v2 import runtime_source_hashes


def source_archive(path, *, teaching=b"before", worker=b"worker", training=b"frozen"):
    protocol = {"behavior_source_sha256": {"src/assembly_recovery/training_env.py": hashlib.sha256(b"frozen").hexdigest()}}
    entries = {"configs/protocol_v4.json": json.dumps(protocol).encode(),
        "configs/training_capacity_v1.json": b"{}", "configs/study.json": b"{}",
        "src/assembly_recovery/__init__.py": b'"""Package documentation only."""',
        "src/assembly_recovery/training_env.py": training,
        "src/assembly_recovery/recovery_teaching_v1.py": teaching,
        "scripts/profile_training_capacity_v1.py": worker, "scripts/run_training_capacity_v1.py": b"launcher"}
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def test_unimported_new_module_does_not_change_frozen_capacity_dependencies(tmp_path):
    left = source_archive(tmp_path / "left.zip")
    right = source_archive(tmp_path / "right.zip", teaching=b"after")
    assert runtime_source_hashes(left) == runtime_source_hashes(right)
    assert "src/assembly_recovery/__init__.py" in runtime_source_hashes(left)


def test_capacity_worker_change_is_still_detected(tmp_path):
    left = source_archive(tmp_path / "left.zip")
    right = source_archive(tmp_path / "right.zip", worker=b"changed")
    assert runtime_source_hashes(left) != runtime_source_hashes(right)


def test_frozen_physical_dependency_change_is_rejected(tmp_path):
    changed = source_archive(tmp_path / "changed.zip", training=b"changed physics")
    with pytest.raises(ValueError, match="frozen runtime dependency changed"):
        runtime_source_hashes(changed)
