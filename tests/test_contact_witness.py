import pytest
import torch

from assembly_recovery.contact_witness import ContactRecoveryWitness, TensorContactRecoveryWitness, WitnessCriteria


@pytest.mark.parametrize("contact,rise,clearance,success,forbidden,expected", [
    (1., 0.006, 0.002, True, False, True),
    (0., 0.006, 0.002, True, False, False),
    (1., 0.004, 0.002, True, False, False),
    (1., 0.006, 0., True, False, False),
    (1., 0.006, 0.002, False, False, False),
    (1., 0.006, 0.002, True, True, False),
])
def test_contact_withdrawal_and_completion_are_all_required(contact, rise, clearance, success, forbidden, expected):
    criteria = WitnessCriteria(physics_dt=0.1)
    cpu = ContactRecoveryWitness(criteria)
    batch = TensorContactRecoveryWitness(1, criteria)
    for step in range(1, 21):
        force = contact if step <= 12 else 0.
        z = 0.02 if step <= 12 else 0.02 + rise
        stall = 12 if step >= 12 else 0
        cpu.observe(step, force, z, clearance, stall, step <= 19)
        batch.observe(step, torch.tensor([force], dtype=torch.float64), torch.tensor([z], dtype=torch.float64),
                      torch.tensor([clearance], dtype=torch.float64), torch.tensor([stall]), torch.tensor([step <= 19]))
    job = {"success": success, "elapsed_s": 1.9, "forbidden_events": [{"event": "reset"}] if forbidden else []}
    assert batch.results([job]) == [cpu.result(job)]
    assert cpu.result(job)["witnessed_complete_recovery"] == expected


def test_contact_gap_must_be_sustained_and_withdrawal_must_precede_success():
    c = WitnessCriteria(physics_dt=0.1)
    cpu = ContactRecoveryWitness(c)
    for step in range(1, 21):
        force = 1. if step <= 12 or step % 2 == 0 else 0.
        cpu.observe(step, force, 0.02 if step <= 12 else 0.03, 0.002, 12 if step >= 12 else 0, True)
    assert not cpu.result({"success": True, "elapsed_s": 2., "forbidden_events": []})["witnessed_complete_recovery"]
    cpu.observe(21, 0., 0.03, 0.002, 12, True)
    cpu.observe(22, 0., 0.03, 0.002, 12, True)
    assert not cpu.result({"success": True, "elapsed_s": 2.2, "forbidden_events": []})["witnessed_complete_recovery"]
    assert cpu.result({"success": True, "elapsed_s": 2.3, "forbidden_events": []})["witnessed_complete_recovery"]
