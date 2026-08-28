"""Fail-closed reporting for the rack-mouth insertion intervention."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_insert_distribution_intervention",
    ROOT / "scripts" / "report_insert_distribution_intervention.py",
)
assert SPEC and SPEC.loader
REPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTER)

OLD_HASH = "A" * 64
NEW_HASH = "B" * 64


def _workflow(checkpoint: str, axial: float) -> dict:
    return {
        "seed": 1070,
        "source_revision": {"commit": "abc123", "dirty": False},
        "checkpoint_sha256": {"insert": checkpoint},
        "evaluation_condition": {
            "protocol": "insertion_condition_v2",
            "station": 0,
            "initial_state_sha256": "state",
            "load_path": {"joint_mode": "fixed"},
        },
        "chain": {
            "episodes": 64,
            "successes": 0,
            "success_rate": 0.0,
            "terminal_metrics": {
                "axial_error_m": {"p50": axial},
                "lateral_error_m": {"p50": 0.01},
                "orientation_error_rad": {"p50": 0.1},
            },
        },
    }


def _training() -> dict:
    return {
        "task": "InsertHandoff",
        "source_revision": {"commit": "abc123", "dirty": False},
        "checkpoint_sha256": NEW_HASH,
        "episodes_completed": 64,
        "termination_counts": {"insertion_success": 0},
        "success_rate": 0.0,
        "terminal_metrics": {"terminal_metrics": {"axial_error_m": {"p50": 0.20}}},
    }


def test_report_keeps_both_arms_and_refuses_promotion_without_success() -> None:
    report = REPORTER.build_report(_workflow(OLD_HASH, 0.25), _workflow(NEW_HASH, 0.23), _training())
    assert set(report["identical_state_comparison"]["arms"]) == {"v24", "v25_handoff_only"}
    assert report["identical_state_comparison"]["median_change"]["axial_error_m"] == pytest.approx(-0.02)
    assert report["candidate_on_its_training_task"]["seed_reported_in_json"] is False
    assert report["decision"]["promoted"] is False


def test_report_rejects_unpaired_states_and_mixed_checkpoints() -> None:
    candidate = _workflow(NEW_HASH, 0.23)
    candidate["evaluation_condition"]["initial_state_sha256"] = "different"
    with pytest.raises(ValueError, match="different initial states"):
        REPORTER.build_report(_workflow(OLD_HASH, 0.25), candidate, _training())
    training = _training()
    training["checkpoint_sha256"] = "C" * 64
    with pytest.raises(ValueError, match="different checkpoints"):
        REPORTER.build_report(_workflow(OLD_HASH, 0.25), _workflow(NEW_HASH, 0.23), training)
