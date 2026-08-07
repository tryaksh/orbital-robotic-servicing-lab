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


def test_stage_one_insertion_ppo_contract() -> None:
    params = _yaml("rl_games_insertion.yaml")["params"]
    assert params["network"]["name"] == "blade_swap_teacher"
    assert params["env"]["obs_groups"] == {"obs": ["policy"], "states": []}
    config = params["config"]
    assert config["name"] == "zero_g_blade_insertion"
    assert config["horizon_length"] == 32
    assert config["minibatch_size"] == 4096
    assert 512 * config["horizon_length"] % config["minibatch_size"] == 0

    env_source = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "insertion_env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert '"window_size": 2_000' in env_source
    assert '"minimum_level_steps": 1_600' in env_source
    assert '"stage_mixtures": INSERTION_CURRICULUM_MIXTURES' in env_source


def test_stage_one_is_learned_and_has_no_scripted_motion_path() -> None:
    insertion = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "insertion.py").read_text(
        encoding="utf-8"
    )
    env_cfg = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "insertion_env_cfg.py").read_text(encoding="utf-8")

    assert not (SCRIPTS / "scripted_demo.py").exists()
    assert "SecuredBladeConstraint" in insertion
    assert "desired_linear_velocity - blade.data.root_lin_vel_w" in insertion
    assert "wrist_angular_velocity - blade.data.root_ang_vel_w" in insertion
    assert "2.0 * position_damping_ratio * torch.sqrt(position_stiffness * masses)" in insertion
    assert "def insertion_settling_penalty" in insertion
    assert insertion.count("write_root_pose_to_sim") == 1
    assert "def reset_insertion_blade" in insertion
    assert "tool_to_handle" in (SCRIPTS / "play.py").read_text(encoding="utf-8")
    assert "learn insertion and lateral/vertical alignment" in env_cfg.lower()
    assert "TranslationalDifferentialInverseKinematicsActionCfg" in env_cfg
    assert "scale=(0.006, 0.002, 0.002)" in env_cfg
    assert "insertion_success = DoneTerm" in env_cfg


def test_phase_two_profiles_keep_one_six_axis_policy_interface() -> None:
    robust = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "robust_insertion_env_cfg.py").read_text(
        encoding="utf-8"
    )
    registration = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(
        encoding="utf-8"
    )
    params = _yaml("rl_games_robust_insertion.yaml")["params"]

    assert '"tight_six_axis"' in robust
    assert '"mount_wobble"' in robust
    assert "scale=(0.006, 0.002, 0.002, 0.018, 0.018, 0.018)" in robust
    assert '"mass_distribution_params": (5.0, 15.0)' in robust
    assert '"dynamic_friction_range": (0.20, 1.5)' in robust
    assert '"breakaway_force_range": (10.0, 120.0)' in robust
    assert '"force_range": (-30.0, 30.0)' in robust
    assert "make_robust_insertion_robot_cfg" in robust
    assert "Isaac-ZeroG-Blade-Insertion-Robust-v0" in registration
    assert params["config"]["name"] == "zero_g_blade_insertion_robust"
    assert params["env"]["obs_groups"] == {"obs": ["policy"], "states": []}


def test_phase_two_tight_geometry_has_valid_contact_envelope() -> None:
    assets = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py").read_text(encoding="utf-8")

    assert "GUIDE_CENTER_OFFSET_Y = 0.08975" in assets
    assert "ROBUST_INSERTION_BLADE_CFG.spawn.collision_props.contact_offset = 0.0003" in assets
    assert "ROBUST_INSERTION_SLOT_CFG.spawn.collision_props.contact_offset = 0.0004" in assets
    assert "_robust_guide_cfg.spawn.collision_props.contact_offset = 0.0004" in assets
    assert 'ROBUST_INSERTION_BLADE_CFG.spawn.physics_material.friction_combine_mode = "max"' in assets


def test_phase_two_point_five_uses_real_contact_without_changing_policy_shape() -> None:
    assets = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py").read_text(encoding="utf-8")
    contact = (
        SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "contact_insertion_env_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(
        encoding="utf-8"
    )
    params = _yaml("rl_games_contact_insertion.yaml")["params"]

    assert "CONTACT_INSERTION_BLADE_CFG.spawn.handle_collision_enabled = True" in assets
    assert "make_contact_insertion_robot_cfg" in assets
    assert "secured_blade_constraint = None" in contact
    assert "RailStictionForce" in contact
    assert "RobustInsertionActionsCfg" in contact
    assert "contact_insertion_success_mask" in contact
    assert "GraspSettlingDifferentialInverseKinematicsActionCfg" in contact
    assert "settling_time_s=0.30" in contact
    assert "scale=(0.0015, 0.00075, 0.00075, 0.006, 0.006, 0.006)" in contact
    assert "distance = None" in contact
    assert "grasp_retention = None" in contact
    assert "grasp_slip_penalty" in contact
    train = (SCRIPTS / "train.py").read_text(encoding="utf-8")
    assert "standing still must have negative cumulative reward" in train
    assert "blade_pull_distance" in train
    assert "Physical-grasp axial feasibility test" in train
    assert "a successful insertion must be net-positive" in train
    assert "Isaac-ZeroG-Blade-Insertion-Contact-v0" in registration
    assert params["config"]["name"] == "zero_g_blade_insertion_contact"
    assert params["env"]["obs_groups"] == {"obs": ["policy"], "states": []}


def test_play_supports_contact_inspection_views_and_grasp_audit() -> None:
    play = (SCRIPTS / "play.py").read_text(encoding="utf-8")

    assert '"--inspection_view"' in play
    assert '"physical_contact"' in play
    assert '"physx_fixed_joint"' in play
    assert '"Insertion-Contact"' in play


def test_rigid_grasp_task_is_registered_and_uses_physx_joint() -> None:
    assets = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py").read_text(encoding="utf-8")
    registration = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(
        encoding="utf-8"
    )
    train = (SCRIPTS / "train.py").read_text(encoding="utf-8")
    rigid = (
        SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "rigid_grasp_insertion_env_cfg.py"
    ).read_text(encoding="utf-8")
    params = _yaml("rl_games_rigid_grasp.yaml")["params"]

    assert "UsdPhysics.FixedJoint.Define" in assets
    assert "RIGID_GRASP_BLADE_CFG.spawn.handle_collision_enabled = False" in assets
    assert "Isaac-ZeroG-Blade-Insertion-RigidGrasp-v0" in registration
    assert "rl_games_rigid_grasp.yaml" in registration
    assert "Rigid-grasp contract passed" in train
    assert "insertion_axial_progress_reward" in rigid
    assert "insertion_timeout_error_penalty" in rigid
    assert "self.scene.blade_slot.spawn.collision_props.collision_enabled = False" in rigid
    assert "self.events.slot_material = None" in rigid
    assert params["config"]["name"] == "zero_g_blade_insertion_rigid_grasp"


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
    commands = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "commands.py").read_text(encoding="utf-8")
    randomization = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "randomization.py").read_text(
        encoding="utf-8"
    )
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
    assert '"--curriculum_stage"' in play
    assert "force_reset_stage(args.curriculum_stage)" in play
    assert 'result["timeout_failed_conditions"]' in play
    assert '"--robustness_level"' in train
    assert '"--robustness_level"' in play
    assert 'result["randomization_buckets"]' in play


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
