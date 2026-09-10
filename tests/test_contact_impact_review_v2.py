import copy
from dataclasses import asdict

import pytest

from assembly_recovery.retry_controller import RetrySettings
from scripts.review_contact_impact_v2 import known_settings_verifier_failure


def _failure():
    return (
        {"status": "verification_failed", "returncode": 0, "timed_out": False,
         "resource_guard_stop": None, "checks": {"process_completed": True, "frozen_script_settings": False}},
        {"status": "completed"},
        {"status": "check_failed", "failed_checks": ["frozen_script_settings"],
         "checks": {"process_completed": True, "frozen_script_settings": False}},
    )


def test_sparse_overrides_resolve_to_full_frozen_controller_settings():
    overrides = {"insertion_load_n": 4., "initial_pressure_offset_m": .006}
    resolved = asdict(RetrySettings(**overrides))
    assert resolved != overrides
    assert resolved["hold_s"] == 1.
    assert resolved["insertion_load_n"] == 4.
    assert resolved["initial_pressure_offset_m"] == .006
    assert known_settings_verifier_failure(*_failure())


@pytest.mark.parametrize("key,value", [
    ("returncode", 1), ("timed_out", True), ("resource_guard_stop", "low_memory"),
    ("status", "artifact_check_failed"),
])
def test_cpu_correction_never_excuses_process_failure(key, value):
    manifest, report, verification = _failure()
    manifest[key] = value
    assert not known_settings_verifier_failure(manifest, report, verification)


def test_cpu_correction_never_excuses_additional_failed_physical_check():
    manifest, report, verification = _failure()
    changed = copy.deepcopy(verification)
    changed["checks"]["native_impulse_to_force"] = False
    assert not known_settings_verifier_failure(manifest, report, changed)
    manifest["checks"]["source_archive_hash"] = False
    assert not known_settings_verifier_failure(manifest, report, verification)

