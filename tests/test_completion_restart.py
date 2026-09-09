import importlib.util
import json
import sys
from pathlib import Path

import pytest

from assembly_recovery.protocol import LAB_COMMIT, sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("completion_restart", ROOT / "scripts/run_completion_restart.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_restart_preserves_frozen_science_and_all_reference_hashes():
    original = json.loads((ROOT / "configs/protocol_v3.json").read_text())
    restart = json.loads((ROOT / "configs/protocol_v3_restart_v1.json").read_text())
    for key in ("architecture", "ppo", "finite_job_contract", "reward_correction", "study_sha256"):
        assert restart[key] == original[key]
    for key in ("seed", "num_envs", "cohorts", "charged_control_equivalent_transitions", "competence_gate"):
        assert restart["training"][key] == original["training"][key]
    for section in ("behavior_source_sha256", "gate_evidence_sha256"):
        for name, digest in restart[section].items():
            assert sha256(ROOT / name) == digest, name
    for section in ("training_registration", "interrupted_run", "previous_protocol"):
        reference = restart[section]
        assert sha256(ROOT / reference["path"]) == reference["sha256"]


def test_battery_preflight_refuses_before_output_or_simulator(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "evidence").mkdir()
    study = tmp_path / "configs/study.json"
    study.write_bytes((ROOT / "configs/study.json").read_bytes())
    protocol = {"training": {"seed": 170, "num_envs": 1024, "cohorts": 22},
                "study_sha256": sha256(study), "behavior_source_sha256": {}, "gate_evidence_sha256": {}}
    (tmp_path / "configs/protocol_v3_restart_v1.json").write_text(json.dumps(protocol))
    (tmp_path / "evidence/completion_credit_controls_v3_r01.json").write_text(json.dumps({"status": "verified", "checks": {"fixture": True}}))
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "git", lambda path, *args: LAB_COMMIT if args == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(launcher, "system_power_status", lambda: {"platform": "windows", "ac_line_status": 0})
    monkeypatch.setattr(sys, "argv", ["restart", "train", "--run-id", "refuse-battery"])
    with pytest.raises(SystemExit):
        launcher.main()
    assert not (tmp_path / "artifacts").exists()
