"""CPU-only checks of budget planning and fail-closed pilot launch gates."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("uniform_launcher", ROOT / "scripts/run_uniform.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_pilot_plan_counts_all_initialization_physics(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_uniform", "plan", "--run-id", "test-plan"])
    assert launcher.main() == 0
    plan = json.loads(capsys.readouterr().out)["specification"]
    assert plan["rollout_control_transitions"] == 2304000
    assert plan["expected_charged_control_equivalent_transitions"] == 2346880


@pytest.mark.parametrize("seed", [10070, 20070])
def test_reserved_seeds_cannot_train(monkeypatch, seed):
    monkeypatch.setattr(sys, "argv", ["run_uniform", "plan", "--run-id", "refuse", "--seed", str(seed)])
    with pytest.raises(SystemExit):
        launcher.main()


def test_open_gate_refuses_launch_before_gpu_or_filesystem_mutation(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    study = json.loads((ROOT / "configs/study.json").read_text())
    study["gates"]["training_terminal_handling_verified"] = False
    (tmp_path / "configs/study.json").write_text(json.dumps(study))
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_uniform", "train", "--run-id", "refuse"])
    with pytest.raises(SystemExit):
        launcher.main()
    assert not (tmp_path / "artifacts").exists()


def test_preserved_v1_configuration_and_evidence_match():
    from assembly_recovery.protocol import sha256, validate_study

    frozen = json.loads((ROOT / "configs/protocol_v1.json").read_text())
    study = json.loads((ROOT / "configs/study_peg_pilot_v1.json").read_text())
    assert not validate_study(study)
    assert sha256(ROOT / "configs/study_peg_pilot_v1.json") == frozen["study_sha256"]
    for name, digest in frozen["gate_evidence_sha256"].items():
        assert sha256(ROOT / name) == digest, name


def test_current_frozen_protocol_matches_executing_sources():
    from assembly_recovery.protocol import sha256, validate_study

    study = json.loads((ROOT / "configs/study.json").read_text())
    assert not validate_study(study)
    frozen = json.loads((ROOT / study["training"]["pilot_protocol"]).read_text())
    assert sha256(ROOT / "configs/study.json") == frozen["study_sha256"]
    for name, digest in frozen["behavior_source_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in frozen["gate_evidence_sha256"].items():
        assert sha256(ROOT / name) == digest, name
