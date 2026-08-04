from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_teacher_rl_games_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "src/zero_g_blade_swap/tasks/blade_swap/agents/rl_games_teacher.yaml").read_text(encoding="utf-8")
    )
    params = config["params"]
    assert params["config"]["horizon_length"] == 32
    assert params["config"]["gamma"] == 0.99
    assert params["config"]["tau"] == 0.95
    assert params["config"]["learning_rate"] == 3.0e-4


def test_vision_config_is_memory_conscious() -> None:
    config = yaml.safe_load(
        (ROOT / "src/zero_g_blade_swap/tasks/blade_swap/agents/rl_games_vision.yaml").read_text(encoding="utf-8")
    )
    assert config["params"]["config"]["horizon_length"] == 32


def test_no_large_runtime_artifacts_are_tracked_by_layout() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for expected in (".deps/", "datasets/", "checkpoints/", "videos/", "*.hdf5", "*.onnx"):
        assert expected in ignored
