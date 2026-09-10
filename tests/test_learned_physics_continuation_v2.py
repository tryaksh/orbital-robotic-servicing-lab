import json
from pathlib import Path

from assembly_recovery.protocol import sha256


def test_continuation_preserves_protocol_and_charges_reused_run_once():
    root = Path(__file__).resolve().parents[1]
    core_path = root / "configs/learned_physics_validation_v1.json"
    core = json.loads(core_path.read_text())
    continuation = json.loads((root / "configs/learned_physics_continuation_v2.json").read_text())
    assert continuation["physics_registration_sha256"] == sha256(core_path)
    assert continuation["remaining_run_labels"] == [r["label"] for r in core["run_order"][1:]]
    assert continuation["additional_charged_reference_transitions"] == 101972.5
    assert continuation["additional_charged_reference_transitions"] + 12834.5 == core["expected_charged_reference_transitions"]
    assert continuation["simulator_worker_changed"] is False
    for path, digest in continuation["source_sha256"].items():
        assert sha256(root / path) == digest, path
