import copy
import json
from pathlib import Path

import pytest

from assembly_recovery.faults import assert_training_case, development_cases, validate_fault_support


@pytest.fixture
def study():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/study.json").read_text())


def test_development_matrix_is_deterministic_and_covers_each_declared_bin(study):
    cases = development_cases(study, 10070)
    assert cases == development_cases(study, 10070)
    assert len(cases) == 4 * len(study["fault_support"]["bins"])
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert {c["bin_id"] for c in cases} == {b["id"] for b in study["fault_support"]["bins"]}
    other = development_cases(study, 10071)
    assert not {c["case_id"] for c in cases} & {c["case_id"] for c in other}


@pytest.mark.parametrize("seed", [170, 20070, 20071, 20072, -1])
def test_validation_rejects_non_development_seeds(study, seed):
    with pytest.raises(ValueError, match="development seeds"):
        development_cases(study, seed)


def training_case(study, bin_id):
    case = next(c for c in development_cases(study, 10070) if c["bin_id"] == bin_id)
    return {**case, "case_id": "train-170-example", "seed": 170, "split": "training"}


def test_training_rejects_development_identity_and_reserved_combinations(study):
    for case in development_cases(study, 10070):
        with pytest.raises(ValueError):
            assert_training_case(study, 170, case)
    case = training_case(study, "bias_medium")
    assert_training_case(study, 170, case)
    case["approach_xy_m"] = [0.004, 0.0]
    with pytest.raises(ValueError, match="reserved"):
        assert_training_case(study, 170, case)


@pytest.mark.parametrize("change", ["range", "seed", "id", "dynamic_friction", "nonfinite"])
def test_training_rejects_outside_support_or_mislabeled_cases(study, change):
    case = training_case(study, "bias_medium")
    if change == "range":
        case["fixture_bias_xy_m"] = [0.003, 0.0]
    elif change == "seed":
        case["seed"] = 20070
    elif change == "id":
        case["case_id"] = "dev-10070-example"
    elif change == "dynamic_friction":
        case["fixed_dynamic_friction"] = 0.6
    else:
        case["fixture_bias_xy_m"] = [float("nan"), 0.0]
    with pytest.raises(ValueError):
        assert_training_case(study, 170, case)


def test_fault_bin_derivation_rejects_unbounded_severity(study):
    support = copy.deepcopy(study["fault_support"])
    support["bins"][1]["range"][1] = 0.1
    with pytest.raises(ValueError, match="envelope"):
        validate_fault_support(support)


@pytest.mark.parametrize("outcome,contact,stall,recovery,expected", [
    ("success", True, True, True, "witnessed_recovery"),
    ("success", True, False, False, "completion_without_witnessed_contact_failure"),
    ("success", True, True, False, "completion_without_full_recovery_witness"),
    ("deadline", True, True, False, "contact_stall_deadline"),
    ("deadline", True, False, False, "contact_without_stall_deadline"),
    ("deadline", False, False, False, "free_space_miss_deadline"),
    ("force_abort", True, True, False, "force_abort"),
    ("initialization_invalid", False, False, False, "initialization_invalid"),
])
def test_matrix_distinguishes_recovery_prevention_and_free_space_misses(outcome, contact, stall, recovery, expected):
    from scripts.summarize_fault_matrix import classify_job

    job = {"outcome": outcome, "success": outcome == "success"}
    assert classify_job(job, stall, recovery, contact) == expected


@pytest.mark.parametrize("field,value", [("gravity", "disabled"), ("velocity", "upstream"), ("seed", 20070),
                                          ("seconds", 35), ("settings_sha256", "changed"), ("study_sha256", "changed")])
def test_matrix_rejects_changed_physics_or_plan_before_launch(study, field, value):
    from assembly_recovery.faults import validate_development_request

    plan = {"study": study, "cases": development_cases(study, 10070), "deadline_s": 30,
            "study_sha256": "study", "retry_settings_sha256": "settings"}
    request = {"study_sha256": "study", "settings_sha256": "settings", "seed": 10070,
               "controller": "retry", "seconds": 30, "gravity": "enabled", "velocity": "corrected"}
    validate_development_request(plan, **request)
    request[field] = value
    with pytest.raises(ValueError):
        validate_development_request(plan, **request)
