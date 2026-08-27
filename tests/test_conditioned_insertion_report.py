"""Paired insertion comparisons fail closed and retain losing arms."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from zero_g_blade_swap.evaluation import TERMINAL_METRIC_FIELDS, TERMINATION_REASONS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "conditioned_insertion_report_for_test", ROOT / "scripts" / "report_conditioned_insertion.py"
)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def _rows(successes: int, episodes: int = 4) -> np.ndarray:
    rows = np.zeros((episodes, len(TERMINAL_METRIC_FIELDS)), dtype=np.float64)
    success = TERMINAL_METRIC_FIELDS.index("success")
    reason = TERMINAL_METRIC_FIELDS.index("termination_reason")
    rows[:successes, success] = 1.0
    rows[:successes, reason] = TERMINATION_REASONS.index("insertion_success")
    rows[successes:, reason] = TERMINATION_REASONS.index("time_out")
    return rows


def _run(controller: str, station: int, seed: int, successes: int, digest: str | None = None):
    return {
        "path": Path(f"{controller}_{station}_{seed}.npz"),
        "file_sha256": controller.upper(),
        "fields": TERMINAL_METRIC_FIELDS,
        "rows": _rows(successes),
        "metadata": {
            "seed": seed,
            "controller": controller,
            "evaluation_condition": {
                "protocol": REPORT.PROTOCOL,
                "kind": "reset_station",
                "station": station,
                "initial_state_sha256": digest or f"state-{station}",
            },
            "source_revision": {"available": True, "commit": "a" * 40, "dirty": False},
            "checkpoints": {"insert": "B" * 64} if controller == "policy" else {},
        },
    }


def test_report_keeps_each_losing_arm_and_applies_every_condition_rule() -> None:
    runs = [
        _run("guarded", 0, 1070, 4),
        _run("policy", 0, 1070, 1),
        _run("guarded", 8, 1070, 2),
        _run("policy", 8, 1070, 4),
    ]
    report = REPORT.build_report(runs, expected_policy_sha256="b" * 64)

    assert [row["losing_arm"] for row in report["paired_conditions"]] == ["policy", "guarded"]
    assert all(set(row["arms"]) == {"guarded", "policy"} for row in report["paired_conditions"])
    assert report["overall"]["by_controller"]["guarded"]["episodes"] == 8
    assert report["decision"]["policy_not_worse_on_every_paired_condition_and_pooled"] is False
    assert report["decision"]["recommended_controller"] == "guarded"


def test_report_rejects_mismatched_states_and_missing_arms() -> None:
    with pytest.raises(ValueError, match="same state"):
        REPORT.build_report(
            [
                _run("guarded", 0, 1070, 4, digest="guarded-state"),
                _run("policy", 0, 1070, 4, digest="policy-state"),
            ]
        )
    with pytest.raises(ValueError, match="both controller arms"):
        REPORT.build_report([_run("policy", 0, 1070, 4)])


def test_loader_rejects_dirty_provenance(tmp_path: Path) -> None:
    path = tmp_path / "dirty.npz"
    metadata = _run("policy", 0, 1070, 4)["metadata"]
    metadata["source_revision"]["dirty"] = True
    np.savez_compressed(
        path,
        rows=_rows(4),
        fields=np.asarray(TERMINAL_METRIC_FIELDS),
        metadata=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="dirty tracked worktree"):
        REPORT.load_runs([path])


def test_workflow_exposes_same_driver_paths_for_conditioned_stations() -> None:
    source = (ROOT / "scripts" / "run_workflow_demo.py").read_text(encoding="utf-8")
    assert '"--start_insert_station"' in source
    assert "insertion_reference = ZeroGBladeGrapplePinInsertTwoSlotEnvCfg()" in source
    assert 'env_cfg.events.reset_stroke.params["noise_rad"] = 0.0' in source
    assert "direct_insert = self.rigid_transit or self.insert_only" in source
    assert 'if direct_insert and self.insert_controller == "policy"' in source
    assert "elif direct_insert and bool(inserting.any())" in source
