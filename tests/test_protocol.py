import copy
import json
from pathlib import Path

import pytest

from assembly_recovery.protocol import RunSpec, assess_completion, training_command, validate_run_id, validate_study

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kwargs", [
    {"task": "retired_space_task"}, {"seed": -1}, {"num_envs": 3}, {"num_envs": 65},
    {"num_envs": 1024}, {"epochs": 0}, {"epochs": 1000}, {"max_minutes": 0}, {"max_minutes": 301},
])
def test_invalid_or_unbounded_run_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RunSpec(**kwargs)


@pytest.mark.parametrize("run_id", ["../escape", "", "a/b", "a\\b", "a;echo", "--option", "a b", "a" * 65])
def test_run_id_cannot_escape_output_directory_or_inject_arguments(run_id):
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_plan_keeps_paths_with_spaces_as_single_arguments():
    spec = RunSpec(seed=271, num_envs=64, epochs=2)
    command = training_command(Path("project with spaces"), Path("sim with spaces/python.bat"), spec, "run_271")
    assert command[0] == str(Path("sim with spaces/python.bat"))
    assert command[command.index("--seed") + 1] == "271"
    assert "agent.params.config.full_experiment_name=run_271" in command
    assert spec.requested_transitions == 16384


@pytest.mark.parametrize(("returncode", "timeout", "checks", "expected"), [
    (0, False, {"checkpoint": True}, "completed"),
    (0, True, {"checkpoint": True}, "timed_out"),
    (1, False, {"checkpoint": True}, "process_failed"),
    (0, False, {"checkpoint": True, "config": False}, "artifact_check_failed"),
    (0, False, {}, "artifact_check_failed"),
])
def test_partial_artifacts_never_mask_failed_or_timed_out_jobs(returncode, timeout, checks, expected):
    assert assess_completion(returncode, timeout, checks) == expected


def test_current_study_is_explicitly_not_frozen():
    study = json.loads((ROOT / "configs/study.json").read_text())
    unmet = validate_study(study)
    assert "protocol_frozen" in unmet
    assert "held_part_gravity_resolved" in unmet


def test_training_and_evaluation_seed_leakage_is_rejected():
    study = json.loads((ROOT / "configs/study.json").read_text())
    contaminated = copy.deepcopy(study)
    contaminated["splits"]["test_seeds"].append(study["training"]["seeds"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_study(contaminated)
