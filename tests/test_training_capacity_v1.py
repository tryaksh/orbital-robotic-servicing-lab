import copy
import json
from pathlib import Path

import pytest
import torch

from assembly_recovery.protocol import sha256
from scripts.profile_training_capacity_v1 import finite_tensor_tree, reloaded_actions_match
from scripts.run_training_capacity_v1 import charged_cost_bounds, report_checks, resource_summary, specification_for

ROOT = Path(__file__).resolve().parents[1]


def test_capacity_budget_reports_twice_the_work_without_equal_sample_claim():
    registration = json.loads((ROOT / "configs/training_capacity_v1.json").read_text())
    lower = specification_for(registration, 1024)
    upper = specification_for(registration, 2048)
    assert lower["cohorts"] == upper["cohorts"] == 2
    assert lower["expected_charged_transitions"] == 938752
    assert upper["expected_charged_transitions"] == 1877504
    assert lower["rollout_control_transitions"] + lower["expected_initialization_physics_env_steps"] / 8 == lower["expected_charged_transitions"]
    assert upper["requested_jobs"] == 2 * lower["requested_jobs"]
    assert "no equal-sample" in registration["work_comparison"]
    with pytest.raises(ValueError):
        specification_for(registration, 4096)


def passing_artifact(tmp_path):
    # Small synthetic completed run: two active transitions, two absorbing transitions.
    spec = {"num_envs": 2, "cohorts": 1, "controls_per_cohort": 2,
            "rollout_control_transitions": 4, "expected_initialization_physics_env_steps": 16,
            "expected_charged_transitions": 6}
    exports = []
    for name in ("diagnostics_0.pt", "benchmark_cohort_1.pt"):
        path = tmp_path / name
        path.write_bytes(b"artifact contents checked by hash")
        exports.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {"status": "completed", "version": "training_capacity_v1", "research_result": False,
        "cost": {"rollout_control_transitions": 4, "initialization_physics_env_steps": 16,
                 "charged_control_equivalent_transitions": 6, "active_control_transitions": 2, "absorbing_control_transitions": 2},
        "optimizer_updates": [{"active_samples": 2, "optimizer_steps": 4}],
        "throughput_intervals": [{"active_samples": 2, "absorbing_samples": 2}],
        "cohorts": [{"jobs": [{"finished": True, "forbidden_events": []}] * 2,
            "initialization": [{"valid": True}] * 2, "done_counts": [1, 1], "active_count_by_control": [2, 0],
            "checkpoint_validation": {"model_and_optimizer_finite": True, "identical_loaded_actions": True,
                "optimizer_state_nonempty": True, "diagnostics_finite": True, "diagnostic_shape": [2, 2, 46]},
            "exports": exports}]}
    return spec, report


def test_capacity_rejects_absorbing_optimizer_samples_and_incomplete_jobs(tmp_path):
    spec, report = passing_artifact(tmp_path)
    assert all(report_checks(report, spec, tmp_path).values())
    contaminated = copy.deepcopy(report)
    contaminated["optimizer_updates"][0]["active_samples"] = 4
    assert not report_checks(contaminated, spec, tmp_path)["optimizer_uses_all_and_only_active_samples"]
    incomplete = copy.deepcopy(report)
    incomplete["cohorts"][0]["jobs"][0]["finished"] = False
    assert not report_checks(incomplete, spec, tmp_path)["jobs_finished_without_mutation"]
    missing = copy.deepcopy(report)
    missing["cohorts"][0]["done_counts"] = [1, 0]
    assert not report_checks(missing, spec, tmp_path)["one_terminal_per_job"]


def test_capacity_rejects_post_export_corruption_even_with_completed_report(tmp_path):
    spec, report = passing_artifact(tmp_path)
    path = tmp_path / "benchmark_cohort_1.pt"
    path.write_bytes(b"x" * path.stat().st_size)
    assert not report_checks(report, spec, tmp_path)["all_exports_present_and_hashed"]


def test_finite_tree_catches_nonfinite_nested_optimizer_moments():
    clean = {"model": {"weight": torch.ones(2)}, "optimizer": {"state": {1: {"exp_avg_sq": torch.zeros(2)}}}}
    assert finite_tensor_tree(clean)
    clean["optimizer"]["state"][1]["exp_avg_sq"][0] = float("nan")
    assert not finite_tensor_tree(clean)


def test_missing_resource_telemetry_cannot_be_reported_as_measured_headroom():
    empty = resource_summary([])
    assert empty["peak_used_mib"] is None and empty["sample_count"] == 0
    measured = resource_summary([{"used_mib": 5000, "utilization_percent": 40, "free_ram_mib": 10000},
                                 {"used_mib": 6000, "utilization_percent": 80, "free_ram_mib": 9000}])
    assert measured["peak_used_mib"] == 6000
    assert measured["minimum_free_ram_mib"] == 9000
    assert measured["mean_gpu_utilization_percent"] == 60


def test_incomplete_native_work_has_cost_bounds_even_when_partial_weights_exist():
    spec = {"num_envs": 1024, "expected_charged_transitions": 938752}
    partial = {"status": "failed", "cost": {"charged_control_equivalent_transitions": 10000}}
    bounds = charged_cost_bounds(partial, spec)
    assert bounds["lower"] == 9872 and bounds["upper"] is None
    assert bounds["planned_charged_transitions"] == 938752
    assert charged_cost_bounds({}, spec)["upper"] is None
    partial["status"] = "completed"
    bounds = charged_cost_bounds(partial, spec)
    assert bounds["lower"] == bounds["upper"] == 10000


def test_checkpoint_rebuild_preserves_next_cohort_rng_state():
    from assembly_recovery.study_ppo import StudyPolicy

    model = StudyPolicy(3, 4, action_size=2, widths=(8,))
    observation = {"policy": torch.ones(2, 3), "critic": torch.ones(2, 4)}
    loaded = {"spec": model.spec, "model": model.state_dict()}
    before = torch.get_rng_state().clone()
    assert reloaded_actions_match(model, loaded, observation, "cpu")
    assert torch.equal(before, torch.get_rng_state())
