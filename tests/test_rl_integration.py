"""Pure-Python contracts for the RL integration and launch scripts."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
AGENTS = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/agents"
SCRIPTS = ROOT / "scripts"


def _yaml(name: str) -> dict:
    return yaml.safe_load((AGENTS / name).read_text(encoding="utf-8"))


def test_teacher_ppo_and_asymmetric_critic_contract() -> None:
    params = _yaml("rl_games_teacher.yaml")["params"]
    assert params["network"]["name"] == "blade_swap_teacher"
    assert params["env"] == {
        "clip_observations": 10.0,
        "clip_actions": 1.0,
        "obs_groups": {"obs": ["policy"], "states": ["critic"]},
        "concate_obs_groups": True,
    }
    config = params["config"]
    assert config["horizon_length"] == 32
    assert config["minibatch_size"] == 8192
    assert 1024 * config["horizon_length"] % config["minibatch_size"] == 0
    assert config["central_value_config"]["network"]["name"] == "blade_swap_teacher"


def test_vision_dictionary_actor_and_critic_contract() -> None:
    params = _yaml("rl_games_vision.yaml")["params"]
    assert params["network"]["name"] == "blade_swap_vision"
    assert params["network"]["rgb_key"] == "rgb"
    assert params["network"]["vector_keys"] == ["proprio"]
    assert params["env"]["obs_groups"] == {"obs": ["proprio", "rgb"], "states": ["critic"]}
    assert params["env"]["concate_obs_groups"] is False
    config = params["config"]
    assert config["minibatch_size"] == 2048
    assert 128 * config["horizon_length"] % config["minibatch_size"] == 0
    assert config["central_value_config"]["network"]["name"] == "blade_swap_vision"


@pytest.mark.parametrize(
    "name",
    ("train.py", "play.py", "benchmark.py", "collect_teacher.py", "smoke_env.py"),
)
def test_app_launcher_precedes_isaac_dependent_imports(name: str) -> None:
    source = (SCRIPTS / name).read_text(encoding="utf-8")
    assert source.index("from isaaclab.app import AppLauncher") < source.index("app_launcher = AppLauncher(args)")
    assert source.index("app_launcher = AppLauncher(args)") < source.index("import gymnasium as gym")
    assert "parse_known_args" not in source


def test_smoke_script_has_both_hardware_profiles_and_machine_output() -> None:
    source = (SCRIPTS / "smoke_env.py").read_text(encoding="utf-8")
    assert 'choices=("teacher", "vision", "all")' in source
    assert "--teacher_steps" in source and "default=100" in source
    assert '"--teacher_envs", type=int, default=1' in source
    assert "num_envs=8" in source
    assert "artifacts/smoke_report.json" in source
    assert "sys.argv = [sys.argv[0]]" in source


def test_geometry_and_reward_guards_cover_pilot_failure_modes() -> None:
    assets = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py").read_text(encoding="utf-8")
    commands = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "commands.py").read_text(
        encoding="utf-8"
    )
    randomization = (
        SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "randomization.py"
    ).read_text(encoding="utf-8")
    env_cfg = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "env_cfg.py").read_text(encoding="utf-8")

    assert '"shoulder_pan_joint": 0.35' in assets
    assert "TRANSFER_BLADE_X = 0.18" in assets
    assert "GUIDE_CENTER_OFFSET_Y = 0.08975" in assets
    assert "_handle_goal_from_blade_goal" in commands
    assert "preinsert_blade_goal[:, 0] = TRANSFER_BLADE_X" in commands
    assert "env._last_rewarded_phase[ids] = phase[ids]" in randomization
    assert "self._forces[due, 0]" in randomization
    assert "terminal_failure = RewTerm" in env_cfg


def test_training_and_playback_make_gpu_and_safety_evidence_explicit() -> None:
    train = (SCRIPTS / "train.py").read_text(encoding="utf-8")
    play = (SCRIPTS / "play.py").read_text(encoding="utf-8")

    assert "Simulation and PPO device" in train
    assert "torch.cuda.is_available()" in train
    assert '"termination_counts"' in play
    assert '"maximum_phase_reached"' in play
    assert '"checkpoint_sha256"' in play


def test_benchmark_uses_descending_first_fit_without_aborting_on_failure() -> None:
    source = (SCRIPTS / "benchmark.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEFAULT_CANDIDATES" for target in node.targets)
    )
    assert ast.literal_eval(assignment.value) == {
        "teacher": (1024, 768, 512, 256),
        "vision": (256, 128, 64),
    }
    assert 'if result.get("ok", False) and not args.run_all:' in source
    assert 'if not result.get("ok", False):\n                    break' not in source
    assert '"descending_first_fit"' in source
    assert "--quick" in source and "--run_all" in source
    assert source.index("sys.argv = [sys.argv[0]]") < source.index("app_launcher = AppLauncher(args)")


def test_vision_actor_forward_and_missing_group_error() -> None:
    torch = pytest.importorskip("torch")
    module_path = AGENTS / "network.py"
    spec = importlib.util.spec_from_file_location("blade_swap_network_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    actor = module.VisionActor({"rgb": (64, 64, 3), "proprio": (48,)}, actions_num=7)
    output = actor({"rgb": torch.rand(4, 64, 64, 3), "proprio": torch.rand(4, 48)})
    assert output.shape == (4, 7)
    assert torch.isfinite(output).all()
    with pytest.raises(KeyError, match="missing groups"):
        module.VisionActor({"rgb": (64, 64, 3)}, actions_num=7)
