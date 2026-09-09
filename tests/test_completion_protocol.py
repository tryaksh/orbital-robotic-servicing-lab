import json
from pathlib import Path

from assembly_recovery.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_completion_protocol_sources_gates_and_registration_match():
    protocol = json.loads((ROOT / "configs/protocol_v3.json").read_text())
    assert protocol["status"] == "frozen_before_substantial_training"
    assert sha256(ROOT / "configs/study.json") == protocol["study_sha256"]
    for section in ("behavior_source_sha256", "gate_evidence_sha256"):
        for name, digest in protocol[section].items():
            assert sha256(ROOT / name) == digest, name
    for section in ("reward_correction", "training_registration", "previous_protocol"):
        reference = protocol[section]
        assert sha256(ROOT / reference["path"]) == reference["sha256"]
    assert protocol["training"]["charged_control_equivalent_transitions"] == 10326272
    gate = json.loads((ROOT / "evidence/completion_credit_controls_v3_r01.json").read_text())
    assert gate["status"] == "verified" and all(gate["checks"].values())
