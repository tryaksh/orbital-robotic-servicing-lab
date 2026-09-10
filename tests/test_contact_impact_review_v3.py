"""Float32 verification correction cannot hide physical or process failures."""
import copy

import pytest
import torch

from scripts.review_contact_impact_v3 import (
    PRECISION_CHECKS,
    contact_precision_checks,
    known_precision_verifier_failure,
    rounding_comparison,
)


def test_one_ulp_at_large_contact_scale_passes_but_scale_error_does_not():
    left = torch.tensor([[574.2781372070312]], dtype=torch.float32)
    right = torch.nextafter(left, torch.full_like(left, float("inf")))
    checks, summary = rounding_comparison(left, right, torch.ones_like(left, dtype=torch.bool),
                                         absolute_floor=3e-5, operations=4)
    assert all(checks.values())
    assert summary["active"]["max_float32_ulp_error"] == 1
    assert summary["active"]["entries_exceeding_old_absolute_bound"] == 1
    wrong = right * 1.001
    checks, _ = rounding_comparison(left, wrong, torch.ones_like(left, dtype=torch.bool),
                                   absolute_floor=3e-5, operations=4)
    assert not checks["scaled_roundoff_bound"]


def test_active_critical_magnitudes_retain_original_absolute_precision():
    left = torch.tensor([[19.0]], dtype=torch.float32)
    right = left.clone()
    for _ in range(6):
        right = torch.nextafter(right, torch.full_like(right, float("inf")))
    checks, _ = rounding_comparison(left, right, torch.ones_like(left, dtype=torch.bool),
                                   absolute_floor=1e-5, operations=12)
    assert checks["scaled_roundoff_bound"]
    assert not checks["active_at_most20n_original_absolute_bound"]


def test_even_one_ulp_contact_threshold_change_is_rejected():
    force = torch.zeros((1, 1, 1, 3, 3))
    force[0, 0, 0, 0, 0] = 5.0
    physics = torch.zeros((1, 1, 14), dtype=torch.float64)
    physics[..., 11] = torch.nextafter(torch.tensor(5.0), torch.tensor(float("inf"))).double()
    report = {"criteria": {"physics_dt": 1 / 960}, "jobs": [{"elapsed_s": 1 / 960}]}
    checks, diagnostics = contact_precision_checks(
        report, {"native": {"contact_force": force, "contact_impulse": force / 960}, "physics": physics})
    assert checks["contact_evaluator_input"]
    assert not checks["active_contact_threshold_predicates_unchanged"]
    assert diagnostics["contact_threshold_margins"]["5.0"]["active_predicate_differences"] == 1


def known_failure():
    required = {
        "process_completed", "independent_cpu_job_and_witness_replay", "source_archive_hash",
        "all_archived_source_hashes", "first_raw_20n_crossing_ends_active_job", "full_native_cost",
    }
    checks = {key: True for key in required} | {key: False for key in PRECISION_CHECKS}
    return (
        {"status": "verification_failed", "returncode": 0, "timed_out": False,
         "resource_guard_stop": None, "refinement": 8, "checks": copy.deepcopy(checks)},
        {"status": "completed", "refinement": 8, "cost": {"charged_reference_transitions": 13674.5}},
        {"status": "check_failed", "verifier_version": "contact_impact_review_v2",
         "failed_checks": sorted(PRECISION_CHECKS), "checks": checks},
    )


def test_only_the_exact_two_preserved_numerical_checks_can_be_corrected():
    assert known_precision_verifier_failure(*known_failure())
    manifest, report, saved = known_failure()
    saved["checks"]["first_raw_20n_crossing_ends_active_job"] = False
    assert not known_precision_verifier_failure(manifest, report, saved)
    manifest, report, saved = known_failure()
    manifest["checks"]["source_archive_hash"] = False
    assert not known_precision_verifier_failure(manifest, report, saved)


@pytest.mark.parametrize("key,value", [
    ("returncode", 1), ("timed_out", True), ("resource_guard_stop", "low_memory"),
    ("status", "failed"), ("refinement", 4),
])
def test_numerical_correction_never_excuses_process_failure(key, value):
    manifest, report, saved = known_failure()
    manifest[key] = value
    assert not known_precision_verifier_failure(manifest, report, saved)
