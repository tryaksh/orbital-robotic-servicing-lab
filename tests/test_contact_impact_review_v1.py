import copy

import torch

from scripts.review_contact_impact_v1 import _window_metrics, compare_pair, relative_difference, verify_single


def _summary(hz):
    return {"physics_hz": hz, "cases": [
        {"case_id": f"case-{i}", "outcome": "probe_end", "active_end_s": 4.,
         "active": {"wrist_peak_n": 10., "fixture_impulse_magnitude_ns": 1.}} for i in range(28)]}


def test_registered_gate_retains_single_outlier_without_changing_median():
    coarse, fine = _summary(240), _summary(480)
    fine["cases"][0]["active"]["wrist_peak_n"] = 100.
    fine["cases"][0]["active_end_s"] = 2.
    result = compare_pair(coarse, fine)
    assert result["material_margins_pass"]
    assert result["median_peak_wrist_relative_difference"] == 0
    assert result["outliers"][0]["case_id"] == "case-0"
    assert result["exposure_changed_cases"] == ["case-0"]
    assert len(result["all_cases"]) == 28


def test_outcome_count_and_individual_disagreements_are_separate_margins():
    coarse, fine = _summary(240), _summary(480)
    fine["cases"][0]["outcome"] = "force_abort"
    fine["cases"][1]["outcome"] = "success"
    assert compare_pair(coarse, fine)["material_margins_pass"]
    fine["cases"][2]["outcome"] = "success"
    result = compare_pair(coarse, fine)
    assert result["abort_count_difference"] == 1
    assert not result["margins"]["individual_outcome_changes_at_most_two"]
    assert not result["material_margins_pass"]


def test_load_margin_inclusive_boundary_and_zero_contact_cases():
    coarse, fine = _summary(240), _summary(480)
    for case in fine["cases"]:
        case["active"]["wrist_peak_n"] = 12.5
    assert compare_pair(coarse, fine)["material_margins_pass"]
    unstable = copy.deepcopy(fine)
    for case in unstable["cases"]:
        case["active"]["fixture_impulse_magnitude_ns"] = 2.
    assert not compare_pair(coarse, unstable)["material_margins_pass"]
    assert relative_difference(0., 0.) == 0.
    assert relative_difference(0., 1.) == 1.


def test_impulse_duration_and_peaks_separate_terminal_from_absorbing():
    raw = torch.zeros((3, 1, 6))
    raw[:, 0, 0] = torch.tensor([10., 21., 100.])
    force = torch.zeros((3, 1, 1, 3, 3))
    force[:, 0, 0, 0, 0] = torch.tensor([2., 3., 90.])
    native = {"raw_wrist": raw, "contact_force": force, "contact_impulse": force / 240}
    active = _window_metrics(native, 0, 0, 2, 1 / 240)
    absorbing = _window_metrics(native, 0, 2, 3, 1 / 240)
    assert active["wrist_peak_n"] == 21
    assert absorbing["wrist_peak_n"] == 100
    assert active["fixture_peak_n"] == 3
    assert abs(active["fixture_impulse_magnitude_ns"] - 5 / 240) < 1e-8
    assert active["fixture_contact_duration_s"] == 2 / 240
    assert active["wrist_over_20_duration_s"] == 1 / 240
    assert _window_metrics(native, 0, 0, 0, 1 / 240)["wrist_peak_n"] is None


def test_missing_artifacts_return_failure_flags(tmp_path):
    result = verify_single(tmp_path)
    assert result["status"] == "verification_error"
    assert result["checks"] == {"verifier_completed": False}

