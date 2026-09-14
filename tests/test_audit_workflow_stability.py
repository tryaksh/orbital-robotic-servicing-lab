"""Adversarial fixtures for report/NPZ joining and unchanged physical limits."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "workflow_stability_audit", Path(__file__).resolve().parents[1] / "scripts/audit_workflow_stability.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
COMMIT = "a" * 40
POLICY = "b" * 64


def report(seed=4070, count=8):
    return {
        "seed": seed, "task": audit.TASK, "workflow": "relocate", "num_envs": count,
        "source_revision": {"commit": COMMIT, "dirty": False}, "policy_set_sha256": POLICY,
        "robot_carried_transit": {
            "carrier": "six_axis_robot",
            "observed_per_environment": [{
                "env": env, "entered_transit": True, "retained_throughout": True,
                "samples": 100, "tool_travel_m": 0.3, "module_travel_m": 0.3,
                "max_position_drift_m": 0.001, "max_orientation_drift_rad": 0.01,
            } for env in range(count)],
        },
        "destination_rack_retention": {
            "enabled": True, "world_constraint": False, "module_pose_write": False,
            "joint_body0": "Rack", "joint_body1": "SpareBlade",
            "observed_per_environment": [{
                "env": env, "engaged_after_measured_seating": True,
                "full_rack_only_recheck_observed": True, "rack_only_interval_s": 0.733333,
                "max_rack_to_module_position_drift_m": 0.0,
                "max_rack_to_module_orientation_drift_rad": 0.0,
            } for env in range(count)],
        },
        "capture_interface": {"observed_per_environment": [{
            "env": env, "released_after_seating": True, "hand_opened_after_settling_verification": True,
        } for env in range(count)]},
    }


@pytest.mark.parametrize(
    ("section", "field", "value", "check"),
    [
        ("robot_carried_transit", "entered_transit", False, "entered_transit"),
        ("robot_carried_transit", "retained_throughout", False, "retained_throughout"),
        ("robot_carried_transit", "samples", 0, "transit_samples"),
        ("robot_carried_transit", "tool_travel_m", 0.0, "tool_travel"),
        ("robot_carried_transit", "module_travel_m", 0.0, "module_travel"),
        ("robot_carried_transit", "max_position_drift_m", 0.002688, "transit_position"),
        ("robot_carried_transit", "max_orientation_drift_rad", 0.052361, "transit_orientation"),
        ("destination_rack_retention", "engaged_after_measured_seating", False, "rack_engaged_after_seating"),
        ("destination_rack_retention", "full_rack_only_recheck_observed", False, "rack_recheck_observed"),
        ("destination_rack_retention", "rack_only_interval_s", 0.699999, "rack_hold_duration"),
        ("destination_rack_retention", "max_rack_to_module_position_drift_m", 0.002501, "rack_position"),
        ("destination_rack_retention", "max_rack_to_module_orientation_drift_rad", 0.052361, "rack_orientation"),
        ("capture_interface", "released_after_seating", False, "latch_released"),
        ("capture_interface", "hand_opened_after_settling_verification", False, "hand_released"),
    ],
)
def test_each_physical_failure_rejects_an_otherwise_successful_npz(section, field, value, check):
    data = report()
    data[section]["observed_per_environment"][3][field] = value
    rows = audit.audit_cohort(data, np.ones(8), seed=4070)
    assert rows[3]["npz_success"]
    assert not rows[3]["strict_physical_success"]
    assert rows[3]["failed_checks"] == [check]
    assert sum(row["strict_physical_success"] for row in rows) == 7


@pytest.mark.parametrize(
    ("section", "field", "value", "check"),
    [
        ("robot_carried_transit", "carrier", "payload_stage", "robot_carrier"),
        ("destination_rack_retention", "enabled", False, "rack_enabled"),
        ("destination_rack_retention", "world_constraint", True, "rack_load_path"),
        ("destination_rack_retention", "module_pose_write", True, "rack_load_path"),
        ("destination_rack_retention", "joint_body0", "World", "rack_load_path"),
        ("destination_rack_retention", "joint_body1", "Robot", "rack_load_path"),
    ],
)
def test_global_load_path_failure_rejects_every_row(section, field, value, check):
    data = report()
    data[section][field] = value
    rows = audit.audit_cohort(data, np.ones(8), seed=4070)
    assert all(row["failed_checks"] == [check] for row in rows)


def test_exact_published_boundaries_pass_and_npz_failure_remains_failure():
    data = report()
    for row in data["robot_carried_transit"]["observed_per_environment"]:
        row["max_position_drift_m"] = 0.0025
        row["max_orientation_drift_rad"] = 0.05236
    for row in data["destination_rack_retention"]["observed_per_environment"]:
        row["rack_only_interval_s"] = 0.70
    rows = audit.audit_cohort(data, np.r_[0.0, np.ones(7)], seed=4070)
    assert rows[0]["failed_checks"] == ["npz_terminal_success"]
    assert all(row["strict_physical_success"] for row in rows[1:])


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.001, True, "0.001"])
def test_invalid_measurements_fail_closed(value):
    data = report()
    data["robot_carried_transit"]["observed_per_environment"][0]["max_position_drift_m"] = value
    assert "transit_position" in audit.audit_cohort(data, np.ones(8), seed=4070)[0]["failed_checks"]


def test_unordered_records_join_by_environment_not_list_position():
    data = report()
    data["capture_interface"]["observed_per_environment"][3]["released_after_seating"] = False
    for section in ("robot_carried_transit", "destination_rack_retention", "capture_interface"):
        data[section]["observed_per_environment"].reverse()
    rows = audit.audit_cohort(data, np.ones(8), seed=4070)
    assert [row["env"] for row in rows] == list(range(8))
    assert rows[3]["failed_checks"] == ["latch_released"]


@pytest.mark.parametrize("section", ["robot_carried_transit", "destination_rack_retention", "capture_interface"])
@pytest.mark.parametrize("fault", ["missing", "duplicate", "invalid", "boolean"])
def test_missing_duplicate_or_invalid_environment_records_are_rejected(section, fault):
    data = report()
    rows = data[section]["observed_per_environment"]
    if fault == "missing":
        rows.pop()
    else:
        rows[-1]["env"] = {"duplicate": 0, "invalid": 8, "boolean": True}[fault]
    with pytest.raises(ValueError, match="environment"):
        audit.audit_cohort(data, np.ones(8), seed=4070)


def complete_comparison(root):
    commands = []
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            directory = root / arm / f"seed{seed}"
            directory.mkdir(parents=True)
            data = report(seed)
            # A V4-like accepted NPZ whose transit violates the existing bound.
            if arm == "refined_fixed_base" and seed == 4070:
                data["robot_carried_transit"]["observed_per_environment"][3]["max_position_drift_m"] = 0.002688
            (directory / "workflow_report.json").write_text(json.dumps(data))
            (directory / "execution.log").write_text(
                "[INFO] relocate -> episode 92.7 s\n"
                "[PLAN] visual occupancy preflight passed=8/8 source=bay0 destination=bay1\n"
            )
            np.savez_compressed(
                directory / "episodes.npz", rows=np.ones((8, 1)), fields=np.array(["success"]),
                metadata=json.dumps({
                    "seed": seed, "task": audit.TASK, "checkpoint_sha256": POLICY,
                    "source_revision": {"commit": COMMIT, "dirty": False},
                }),
            )
            commands.append({"arm": arm, "seed": seed, "argv": [
                "python", "driver.py", "--num_envs", "8", "--steps", "1900", "--seed", str(seed),
            ]})
    for arm in audit.ARMS:
        (root / arm / "aggregate.json").write_text(json.dumps({
            "overall": {"episodes": 24}, "gate": {"minimum_stage_success_rate": 0.95, "passed": True},
        }))
    (root / "comparison.json").write_text(json.dumps({
        "seeds": list(audit.SEEDS), "environments_per_seed": 8, "steps": 1900,
        "video": False, "lighting": "randomized", "source_commit": COMMIT, "runs": commands,
    }))


def test_complete_audit_is_reproducible_preserves_inputs_and_identifies_exact_paired_regression(tmp_path):
    complete_comparison(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = audit.build_audit(tmp_path)
    assert result == audit.build_audit(tmp_path)
    assert before == {path: path.read_bytes() for path in before}
    assert result["arms"]["legacy_rail"]["npz_successes"] == 24
    assert result["arms"]["refined_fixed_base"]["npz_successes"] == 24
    assert result["arms"]["refined_fixed_base"]["strict_physical_successes"] == 23
    change = result["paired_changes"]["strict_physical_pass_to_fail"]
    assert len(change) == 1 and change[0]["seed"] == 4070 and change[0]["env"] == 3
    assert result["paired_changes"]["npz_pass_to_fail"] == []
    assert result["original_aggregates"]["refined_fixed_base"]["original_gate"]["passed"]
    assert len(result["inputs"]) == 21


def test_timeout_collection_is_rejected_even_if_eight_rows_were_saved(tmp_path):
    complete_comparison(tmp_path)
    path = tmp_path / "comparison.json"
    data = json.loads(path.read_text())
    data["runs"][0]["argv"].extend(["--episodes", "8"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bounded collection"):
        audit.build_audit(tmp_path)


def test_mismatched_pair_order_is_rejected():
    baseline = audit.audit_cohort(report(), np.ones(8), seed=4070)
    refined = copy.deepcopy(baseline)
    refined.reverse()
    with pytest.raises(ValueError, match="same ordered initial conditions"):
        audit.paired_changes(baseline, refined)


def test_repeated_preflight_rejects_potentially_mixed_episodes(tmp_path):
    complete_comparison(tmp_path)
    path = tmp_path / "legacy_rail/seed4070/execution.log"
    with path.open("a") as stream:
        stream.write("[PLAN] visual occupancy preflight passed=1/1 source=bay0 destination=bay1\n")
    with pytest.raises(ValueError, match="potentially reset episodes"):
        audit.build_audit(tmp_path)


def test_short_episode_horizon_rejects_possible_timeout_resets(tmp_path):
    complete_comparison(tmp_path)
    path = tmp_path / "legacy_rail/seed4070/execution.log"
    path.write_text(path.read_text().replace("92.7 s", "30.0 s"))
    with pytest.raises(ValueError, match="episode timeout"):
        audit.build_audit(tmp_path)


def test_repeated_joint_creation_warnings_do_not_fabricate_a_reset(tmp_path):
    complete_comparison(tmp_path)
    path = tmp_path / "legacy_rail/seed4070/execution.log"
    with path.open("a") as stream:
        stream.write("[Warning] PhysicsUSD: CreateJoint - disjointed body transforms env_2\n" * 2)
    assert audit.build_audit(tmp_path)["arms"]["legacy_rail"]["strict_physical_successes"] == 24
