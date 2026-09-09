import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from scripts import run_experiment


def test_source_snapshot_and_hash_use_identical_bytes(tmp_path, monkeypatch):
    for name in ("README.md", "ROADMAP.md", "AGENTS.md", "pyproject.toml", "environment-lock.example.json"):
        (tmp_path / name).write_text(f"original {name}")
    source = tmp_path / "src/assembly_recovery"
    source.mkdir(parents=True)
    (source / "module.py").write_bytes(b"x = 7\n")
    monkeypatch.setattr(run_experiment, "ROOT", tmp_path)
    archive = tmp_path / "source.zip"
    hashes = run_experiment.snapshot_source(archive)
    (source / "module.py").write_text("x = 999\n")
    with zipfile.ZipFile(archive) as content:
        saved = content.read("src/assembly_recovery/module.py")
    assert saved == b"x = 7\n"
    assert hashes["src/assembly_recovery/module.py"] == hashlib.sha256(saved).hexdigest()


def test_child_failure_is_returned_with_its_log(tmp_path):
    command = [sys.executable, "-c", "print('intentional failure'); raise SystemExit(7)"]
    log = tmp_path / "child.log"
    code, timed_out = run_experiment.run_bounded(command, log, time.monotonic() + 20, os.environ.copy())
    assert code == 7
    assert not timed_out
    assert "intentional failure" in log.read_text()


def test_deadline_terminates_owned_process(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    code, timed_out = run_experiment.run_bounded(command, tmp_path / "child.log", time.monotonic() + 0.2, os.environ.copy())
    assert timed_out
    assert code != 0


def test_plan_needs_no_simulator_or_local_artifacts():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/run_experiment.py"), "plan"],
                            cwd=root, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan["task_id"] == "Isaac-Forge-PegInsert-Direct-v0"
    assert plan["requested_transitions"] == 16384
    study = json.loads((root / "configs/study.json").read_text())
    assert plan["unmet_study_gates"] == [k for k, v in study["gates"].items() if not v]


def test_training_launcher_rejects_reserved_seeds_before_simulator_access():
    root = Path(__file__).resolve().parents[1]
    for mode in ("plan", "train"):
        for seed in (10070, 20070):
            result = subprocess.run([sys.executable, str(root / "scripts/run_experiment.py"), mode,
                                     "--seed", str(seed)], cwd=root, capture_output=True, text=True)
            assert result.returncode != 0
            assert "Training seed must belong" in result.stderr
