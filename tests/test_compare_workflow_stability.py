"""CPU checks for bounded cohort commands and artifact admission."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stability_comparison", ROOT / "scripts/compare_workflow_stability.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_arms_use_the_same_bounded_protocol_and_only_declared_recipe_difference(tmp_path):
    for seed in comparison.SEEDS:
        commands = {
            arm: comparison.workflow_argv(tmp_path, tmp_path / "python.bat", tmp_path / "output", seed, arm)
            for arm in comparison.ARM_FLAGS
        }
        for command in commands.values():
            assert command[command.index("--num_envs") + 1] == "8"
            assert command[command.index("--steps") + 1] == "1900"
            assert command[command.index("--seed") + 1] == str(seed)
            assert "--episodes" not in command
            assert "--video" not in command
            assert "--stable_lighting" not in command
            assert "--rack_retention" in command
        for arm, command in commands.items():
            flags = list(comparison.ARM_FLAGS[arm])
            index = command.index(flags[0], command.index("--seed") + 2)
            assert command[index:index + len(flags)] == flags
            del command[index:index + len(flags)]
        assert commands["legacy_rail"] == commands["refined_fixed_base"]


def archive(path, *, count=8, seed=4070, commit="a" * 40, dirty=False, success=1.0):
    np.savez_compressed(
        path,
        fields=np.array(["success", "control_steps"]),
        rows=np.column_stack((np.full(count, success), np.full(count, 1900.0))),
        metadata=json.dumps({
            "seed": seed, "task": comparison.TASK, "checkpoint_sha256": "b" * 64,
            "source_revision": {"commit": commit, "dirty": dirty},
        }),
    )


def test_eight_complete_failure_rows_are_valid_evidence(tmp_path):
    path = tmp_path / "episodes.npz"
    archive(path, success=0.0)
    result = comparison.read_cohort(path, 4070, "a" * 40)
    assert result["episodes"] == 8
    assert result["successes"] == 0
    assert len(result["sha256"]) == 64


@pytest.mark.parametrize(
    "changes", [{"count": 7}, {"count": 9}, {"seed": 5070}, {"commit": "c" * 40}, {"dirty": True}, {"success": 0.5}],
)
def test_incomplete_or_mismatched_cohorts_are_rejected(tmp_path, changes):
    path = tmp_path / "episodes.npz"
    archive(path, **changes)
    with pytest.raises(ValueError):
        comparison.read_cohort(path, 4070, "a" * 40)
