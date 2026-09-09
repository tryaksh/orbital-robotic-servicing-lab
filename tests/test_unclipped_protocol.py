import copy
import json
from pathlib import Path

from assembly_recovery.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]


def test_value_unclipped_registration_keeps_task_and_other_optimizer_settings():
    current = json.loads((ROOT / "configs/protocol_v4.json").read_text())
    previous = json.loads((ROOT / "configs/protocol_v3.json").read_text())
    expected = copy.deepcopy(previous["ppo"])
    expected["value_clip"] = None
    assert current["ppo"] == expected
    for key in ("architecture", "finite_job_contract", "study_sha256", "upstream", "reward_correction"):
        assert current[key] == previous[key]
    assert current["training"]["competence_gate"] == previous["training"]["competence_gate"]
    assert current["training"]["charged_control_equivalent_transitions"] == 10326272
    assert current["training"]["num_envs"] == 1024
    assert current["training"]["seed"] == 170
    for section in ("behavior_source_sha256", "gate_evidence_sha256"):
        for name, digest in current[section].items():
            assert sha256(ROOT / name) == digest, name
    for section in ("previous_protocol", "training_registration", "reward_correction"):
        ref = current[section]
        assert sha256(ROOT / ref["path"]) == ref["sha256"]
