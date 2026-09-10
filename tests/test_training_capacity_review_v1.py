import copy

import pytest

from scripts.review_training_capacity_v1 import adoption_decision


def summaries():
    baseline = {"status": "verified", "throughput": {"startup_inclusive": {
        "active_samples_per_s": 100, "charged_transitions_per_s": 1000}}}
    candidate = {"status": "verified", "throughput": {"startup_inclusive": {
        "active_samples_per_s": 110, "charged_transitions_per_s": 1100}},
        "resources": {"peak_used_mib": 8000, "minimum_free_ram_mib": 9000},
        "invalid_initializations": 0, "invalid_jobs": 0, "resource_guard_stop": None}
    return baseline, candidate


def test_adoption_requires_both_useful_and_charged_throughput_gain():
    baseline, candidate = summaries()
    assert adoption_decision(baseline, candidate)["selected_num_envs"] == 2048
    candidate["throughput"]["startup_inclusive"]["active_samples_per_s"] = 109
    assert adoption_decision(baseline, candidate)["selected_num_envs"] == 1024
    baseline, candidate = summaries()
    candidate["throughput"]["startup_inclusive"]["charged_transitions_per_s"] = 1099
    assert adoption_decision(baseline, candidate)["selected_num_envs"] == 1024


@pytest.mark.parametrize("field,value", [("peak_used_mib", 10240), ("minimum_free_ram_mib", 4096),
                                        ("minimum_free_ram_mib", None)])
def test_missing_or_insufficient_headroom_rejects_fast_candidate(field, value):
    baseline, candidate = summaries()
    candidate["resources"][field] = value
    assert adoption_decision(baseline, candidate)["selected_num_envs"] == 1024


def test_invalid_jobs_or_artifact_failure_cannot_earn_adoption():
    baseline, candidate = summaries()
    invalid = copy.deepcopy(candidate)
    invalid["invalid_initializations"] = 1
    assert adoption_decision(baseline, invalid)["selected_num_envs"] == 1024
    candidate["status"] = "failed_verification"
    assert adoption_decision(baseline, candidate)["selected_num_envs"] == 1024
