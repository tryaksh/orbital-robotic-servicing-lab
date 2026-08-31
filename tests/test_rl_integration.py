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


def _class_def(source: str, name: str) -> ast.ClassDef:
    return next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef) and node.name == name)


def _declared_fields(source: str, name: str) -> set[str]:
    """Names a config class redeclares, ignoring anything it only inherits."""

    return {
        node.target.id
        for node in _class_def(source, name).body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def test_vision_task_keeps_the_visual_randomizers_reachable() -> None:
    """The camera and both Replicator randomizers must belong to a live task.

    Until 2026-08-10 they were reachable only from the eight-phase swap task.
    Deleting that task without repointing them would have made weeks of
    infrastructure dead code that no smoke test could catch.
    """

    vision = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "vision_insertion_env_cfg.py").read_text(
        encoding="utf-8"
    )
    registration = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(encoding="utf-8")

    assert "Isaac-ZeroG-Blade-Insertion-Vision-v0" in registration
    assert "vision_insertion_env_cfg" in registration
    for term in (
        "make_tiled_camera_cfg",
        "mdp.RackMaterialRandomizer",
        "mdp.OrbitalLightingRandomizer",
        "mdp.camera_rgb_with_radiation_noise",
    ):
        assert term in vision
    # Both randomizers address one prim per environment and raise otherwise.
    assert "replicate_physics=False" in vision
    # The actor must not be handed the pose it is supposed to learn to see.
    actor = vision.split("class InsertionProprioObsCfg")[1].split("class InsertionRgbObsCfg")[0]
    assert "insertion_goal_error" not in actor
    assert "insertion_goal_error" in vision.split("class InsertionCriticObsCfg")[1]


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
    # The guard is "no scripted motion path", not "one call site", and the
    # difference is not academic: this assertion used to count call sites, so
    # adding a second *reset* broke it while nothing about the learned skill
    # changed, and removing that reset again would have let it pass for the
    # wrong reason. A reset writes a pose by construction; what the test is
    # actually about is that nothing *outside* a reset event moves the module.
    # So pin the callers.
    writers = {
        line.strip()
        for line in insertion.splitlines()
        if "write_root_pose_to_sim" in line
    }
    assert writers == {
        "blade.write_root_pose_to_sim(pose, env_ids=ids)",
    }, writers
    assert insertion.count("def reset_") >= 2
    assert "def reset_insertion_blade" in insertion
    # The grasp-abstraction audit moved into the reset-safe terminal metric
    # row that play.py now reports; it must still be measured every episode.
    assert "tool_to_handle_error_m" in (SRC / "zero_g_blade_swap" / "evaluation.py").read_text(encoding="utf-8")
    assert "secured_blade_error_metrics" in (
        SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "terminal_metrics.py"
    ).read_text(encoding="utf-8")
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

    # Derived from the attitude the transit delivers and the attitude a seated
    # module is accepted at, since 2026-08-25; it was derived from the pads
    # before that and inherited from a 160 mm module before that.
    # tests/test_workcell_geometry.py holds the derivation, this only holds that
    # the contact envelope came with it.
    assert "GUIDE_CENTER_OFFSET_Y = 0.085065" in assets
    assert "ROBUST_INSERTION_BLADE_CFG.spawn.collision_props.contact_offset = 0.0001" in assets
    assert "ROBUST_INSERTION_SLOT_CFG.spawn.collision_props.contact_offset = 0.0001" in assets
    assert "_robust_guide_cfg.spawn.collision_props.contact_offset = 0.0001" in assets
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
    ("train.py", "play.py", "benchmark.py", "collect_teacher.py", "smoke_env.py", "grasp_diagnostics.py"),
)
def test_app_launcher_precedes_isaac_dependent_imports(name: str) -> None:
    source = (SCRIPTS / name).read_text(encoding="utf-8")
    assert source.index("from isaaclab.app import AppLauncher") < source.index("app_launcher = AppLauncher(args)")
    assert source.index("app_launcher = AppLauncher(args)") < source.index("import gymnasium as gym")
    assert "parse_known_args" not in source


def test_smoke_script_has_both_hardware_profiles_and_machine_output() -> None:
    source = (SCRIPTS / "smoke_env.py").read_text(encoding="utf-8")
    assert 'choices=("state", "vision", "all")' in source
    assert "--state_steps" in source and "default=100" in source
    assert '"--state_envs", type=int, default=1' in source
    assert "num_envs=8" in source
    assert "artifacts/smoke_report.json" in source
    assert "sys.argv = [sys.argv[0]]" in source
    # A contact sensor that resolves no bodies reports a constant zero, which is
    # indistinguishable from a gentle insertion. Assert it, do not assume it.
    assert 'sensors["blade_contact"].data.net_forces_w is not None' in source


def test_geometry_and_reward_guards_cover_pilot_failure_modes() -> None:
    assets = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py").read_text(encoding="utf-8")
    randomization = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "randomization.py").read_text(
        encoding="utf-8"
    )
    frames = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "grasp_frames.py").read_text(
        encoding="utf-8"
    )

    assert '"shoulder_pan_joint": 0.35' in assets
    # Derived from the attitude the transit delivers and the attitude a seated
    # module is accepted at, since 2026-08-25; it was derived from the pads
    # before that and inherited from a 160 mm module before that.
    # tests/test_workcell_geometry.py holds the derivation, this only holds that
    # the contact envelope came with it.
    assert "GUIDE_CENTER_OFFSET_Y = 0.085065" in assets
    assert "self._forces[due, 0]" in randomization
    # The 2F-85 is symmetric about its closing axis, so a grasp orientation
    # error that ignores the finger swap reports pi where the grip is perfect.
    assert "def equivalent_gripper_orientation" in frames
    assert "flip_x = handle_orientation.new_tensor((0.0, 1.0, 0.0, 0.0))" in frames


def test_insertion_tasks_capture_terminal_metrics_before_auto_reset() -> None:
    registration = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(encoding="utf-8")
    env_module = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "terminal_metrics_env.py").read_text(
        encoding="utf-8"
    )
    collector = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "terminal_metrics.py").read_text(
        encoding="utf-8"
    )
    play = (SCRIPTS / "play.py").read_text(encoding="utf-8")

    # The invariant is that *every* task routes through the pre-reset capture
    # subclass, not that there is a particular number of them. Since the swap
    # task was deleted there is no other entry point left, so assert exactly
    # that rather than counting registrations.
    assert registration.count("entry_point=INSERTION_ENTRY_POINT,") >= 8
    assert "isaaclab.envs:ManagerBasedRLEnv" not in registration
    assert "Isaac-ZeroG-Blade-Insertion-GuidedSlot-v0" in registration
    # Isaac-ZeroG-Blade-CaptureInSlot-v0 was deleted on 2026-08-18: it declared
    # contact_grasp on a blade whose handle collider its parent disables, so it
    # failed the very contract that exists to catch a grasp task gripping nothing,
    # and the certified grapple-pin capture skill does the same job.
    assert "CaptureInSlot" not in registration
    assert "TerminalMetricsManagerBasedRLEnv" in registration
    assert "class TerminalMetricsManagerBasedRLEnv(TerminalMetricsMixin, ManagerBasedRLEnv)" in env_module

    # The collector must read the terminal state, not re-derive it after reset.
    assert "insertion_error_metrics" in collector
    assert "secured_blade_error_metrics" in collector
    assert "attached_blade_velocity" in collector
    assert "env.episode_length_buf" in collector

    assert "enable_terminal_metrics(terminal_metrics)" in play
    assert 'result["terminal_metrics"]' in play
    assert "terminal_metrics_captured_before_reset" in play
    assert '"--episode_metrics"' in play
    # The corrupt post-reset error read must not come back.
    assert "insertion_error_metrics(env.unwrapped)" not in play
    assert '"final_mean_errors"' not in play


def test_force_limited_task_constrains_contact_and_keeps_policy_shape() -> None:
    force = (
        SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "force_limited_insertion_env_cfg.py"
    ).read_text(encoding="utf-8")
    insertion = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "insertion.py").read_text(
        encoding="utf-8"
    )
    registration = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(encoding="utf-8")
    evaluation = (SRC / "zero_g_blade_swap" / "evaluation.py").read_text(encoding="utf-8")

    # Contact load must be both penalised and abortable.
    assert "def contact_force_penalty" in insertion
    assert "def excessive_contact_force" in insertion
    assert "contact_force = RewTerm" in force
    assert "excessive_contact_force = DoneTerm" in force
    assert "ContactSensorCfg" in force
    # The sensor needs real USD clones, and the parent config rebuilds the blade.
    assert "clone_in_fabric=False" in force
    assert "self.scene.spare_blade.spawn.activate_contact_sensors = True" in force
    # The penalty profiles must not redeclare the observation or action fields,
    # so a promoted checkpoint can be fine-tuned rather than retrained.  The
    # force-feedback profile below deliberately breaks this and is excluded.
    for name in (
        "ZeroGBladeForceLimitedInsertionEnvCfg",
        "ZeroGBladeForceLimitedInsertionPlayEnvCfg",
        "ZeroGBladeStrictForceLimitedInsertionEnvCfg",
        "ZeroGBladeStrictForceLimitedInsertionPlayEnvCfg",
    ):
        assert not {"observations", "actions"} & _declared_fields(force, name)
    assert "Isaac-ZeroG-Blade-Insertion-ForceLimited-v0" in registration

    assert "TERMINATION_PRIORITY" in evaluation

    # Reason ids are stored as integers in published evidence, so the first five
    # entries must never move and the new reason must be appended after them.
    from zero_g_blade_swap.evaluation import TERMINATION_PRIORITY, TERMINATION_REASONS

    assert TERMINATION_REASONS[:6] == (
        "non_finite",
        "mount_unstable",
        "insertion_failed",
        "insertion_success",
        "time_out",
        "uncategorized",
    )
    assert TERMINATION_REASONS[6] == "excessive_contact_force"
    # Priority is declared separately from storage order, and a force abort must
    # outrank a geometric success in the same control step.
    assert TERMINATION_PRIORITY.index("excessive_contact_force") < TERMINATION_PRIORITY.index("insertion_success")
    assert set(TERMINATION_PRIORITY).issubset(set(TERMINATION_REASONS))


def test_force_feedback_task_changes_only_the_observation_space() -> None:
    force = (
        SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "force_limited_insertion_env_cfg.py"
    ).read_text(encoding="utf-8")
    insertion = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "insertion.py").read_text(
        encoding="utf-8"
    )
    registration = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(encoding="utf-8")
    train = (SCRIPTS / "train.py").read_text(encoding="utf-8")
    play = (SCRIPTS / "play.py").read_text(encoding="utf-8")

    # The single variable under test: same scene, actions, reward, and
    # terminations as the strict profile, with contact force added to the
    # observation vector and nothing else.
    training_cfg = _class_def(force, "ZeroGBladeForceFeedbackInsertionEnvCfg")
    assert [base.id for base in training_cfg.bases] == ["ZeroGBladeStrictForceLimitedInsertionEnvCfg"]
    assert _declared_fields(force, "ZeroGBladeForceFeedbackInsertionEnvCfg") == {"observations"}

    # The observation term must reuse the same scalar the penalty and abort read.
    assert "class BladeContactWrenchObservation(ManagerTermBase)" in insertion
    assert "blade_contact_force(env, sensor_name).unsqueeze(-1)" in insertion
    assert "def blade_contact_force_vector" in insertion
    assert "forces.sum(dim=1)" in insertion
    # Per-episode filter state must be cleared on reset and mutated in place, or
    # evaluation under inference mode fails on the following reset.
    assert "def reset(self, env_ids: Sequence[int] | None = None) -> None:" in insertion
    assert "self._filtered_force.mul_(1.0 - alpha).add_(tool_force, alpha=alpha)" in insertion
    assert "first_order_filter_alpha" in insertion

    # Evaluation must keep the 60 N limit the earlier force policies were judged
    # under; a 30 N abort here would truncate the distribution being compared.
    assert _declared_fields(force, "ZeroGBladeForceFeedbackInsertionPlayEnvCfg") == {"terminations"}
    assert (
        "terminations: ForceLimitedInsertionTerminationsCfg = ForceLimitedInsertionTerminationsCfg()"
        in force.split("class ZeroGBladeForceFeedbackInsertionPlayEnvCfg")[1]
    )

    # Both arms of the comparison must share one PPO configuration.
    assert registration.count("Isaac-ZeroG-Blade-Insertion-ForceFeedback") == 2
    assert "rl_games_force_feedback" not in registration
    assert '"Insertion-ForceFeedback",' in train
    assert '"Insertion-ForceFeedback",' in play
    # The force-feedback tasks must resolve to the rigid-grasp PPO config.
    agent_tasks = next(line for line in play.splitlines() if line.startswith("RIGID_GRASP_AGENT_TASKS"))
    assert "ForceFeedback" in agent_tasks


def test_pooled_report_records_the_force_limit_without_requiring_it() -> None:
    """Force policies only compare when the abort limit matches, and older runs
    predate the field, so the pooled report must publish it and tolerate its
    absence."""

    np = pytest.importorskip("numpy")
    from zero_g_blade_swap.evaluation import TERMINAL_METRIC_FIELDS, TERMINATION_REASONS

    spec = importlib.util.spec_from_file_location("aggregate_for_test", SCRIPTS / "aggregate_evaluation.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    success_id = float(TERMINATION_REASONS.index("insertion_success"))

    def run(seed: int, limit: float | None) -> dict:
        rows = np.zeros((2, len(TERMINAL_METRIC_FIELDS)), dtype=np.float64)
        rows[:, TERMINAL_METRIC_FIELDS.index("success")] = 1.0
        rows[:, TERMINAL_METRIC_FIELDS.index("termination_reason")] = success_id
        rows[:, TERMINAL_METRIC_FIELDS.index("curriculum_stage")] = [0.0, 1.0]
        metadata = {"checkpoint_sha256": "A" * 64, "seed": seed, "robustness_level": 2, "num_envs": 128}
        if limit is not None:
            metadata["contact_force_limit_n"] = limit
        return {"path": None, "fields": TERMINAL_METRIC_FIELDS, "rows": rows, "metadata": metadata}

    report = module.build_report([run(1065, 60.0), run(2065, 60.0)], "force feedback", 0.95)
    assert report["protocol"]["contact_force_limit_n"] == [60.0]
    assert report["gate"]["passed"] is True

    # A run recorded before the field existed must not break pooling, and a
    # mismatch must stay visible rather than collapsing to one value.
    legacy = module.build_report([run(1065, None), run(2065, 30.0)], "mixed", 0.95)
    assert legacy["protocol"]["contact_force_limit_n"] == [30.0]
    assert module.build_report([run(1065, None)], "legacy", 0.95)["protocol"]["contact_force_limit_n"] is None


def test_pooled_report_preserves_and_validates_clean_source_revision() -> None:
    np = pytest.importorskip("numpy")
    from zero_g_blade_swap.evaluation import TERMINAL_METRIC_FIELDS, TERMINATION_REASONS

    spec = importlib.util.spec_from_file_location("aggregate_provenance_for_test", SCRIPTS / "aggregate_evaluation.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = np.zeros((1, len(TERMINAL_METRIC_FIELDS)), dtype=np.float64)
    rows[:, TERMINAL_METRIC_FIELDS.index("success")] = 1.0
    rows[:, TERMINAL_METRIC_FIELDS.index("termination_reason")] = float(
        TERMINATION_REASONS.index("insertion_success")
    )
    source = {"available": True, "commit": "a" * 40, "branch": "qualification", "dirty": False}

    def run(seed: int, revision: dict | None) -> dict:
        metadata = {
            "checkpoint_sha256": "A" * 64,
            "seed": seed,
            "robustness_level": 0,
            "num_envs": 1,
        }
        if revision is not None:
            metadata["source_revision"] = revision
        return {"path": None, "fields": TERMINAL_METRIC_FIELDS, "rows": rows, "metadata": metadata}

    report = module.build_report([run(1, source), run(2, source)], "clean", 0.95)
    assert report["source_revision"] == source
    with pytest.raises(ValueError, match="mix recorded and missing"):
        module.build_report([run(1, source), run(2, None)], "mixed", 0.95)
    with pytest.raises(ValueError, match="available, clean"):
        module.build_report([run(1, {**source, "dirty": True})], "dirty", 0.95)
    with pytest.raises(ValueError, match="different source commits"):
        module.build_report([run(1, source), run(2, {**source, "commit": "b" * 40})], "split", 0.95)


def test_grasp_diagnostic_measures_the_gate_it_claims_to_measure() -> None:
    source = (SCRIPTS / "grasp_diagnostics.py").read_text(encoding="utf-8")

    # The measurement is only about the grasp if the rails cannot also touch the
    # blade and if a slipping environment is not reset out from under it.
    assert "env_cfg.configure_robustness(0)" in source
    assert "env_cfg.terminations.insertion_failed = None" in source
    assert "env_cfg.events.hold_gripper_closed = None" in source
    # It must refuse to run against the fixed-joint abstraction.
    assert 'if getattr(task.cfg, "rigid_grasp", False):' in source
    assert "there is no grasp to measure" in source
    # The pass mark is the measured Level-2 worst-case contact force, not a
    # threshold invented to make the grasp look adequate.
    assert "LEVEL_2_PEAK_CONTACT_FORCE_N = 66.36" in source
    assert '"required_axial_force_n": required' in source
    assert '"evidence_type": "simulation_physics_characterization"' in source
    assert "It is not learned grasping." in source


def test_guided_slot_is_a_channel_and_capture_happens_inside_it() -> None:
    assets = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py").read_text(encoding="utf-8")
    guided = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "guided_slot_env_cfg.py").read_text(
        encoding="utf-8"
    )
    scene = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "scene_cfg.py").read_text(encoding="utf-8")

    # A channel needs surfaces above the blade, and a funnel needs rotated ones.
    assert "SLOT_UPPER_LIP_CENTER_Z = 0.7435" in assets
    assert "_FLARE_QUAT_LEFT = (0.9945219, 0.0, 0.0, -0.1045285)" in assets
    assert "SLOT_ENTRY_FLARE_DEG = 12.0" in assets
    # The lead-in must be slipperier than the rails or it becomes a catch.
    assert 'friction_combine_mode="min"' in assets
    assert "class ZeroGGuidedSlotSceneCfg(ZeroGRigidGraspInsertionSceneCfg)" in scene

    # The certified geometry must not be edited underneath three evaluations.
    assert "RIGID_GRASP_BLADE_CFG.spawn.handle_size = (0.060, 0.075, 0.030)" in assets
    assert "CONTACT_INSERTION_BLADE_CFG.spawn.handle_offset = GRAPPLE_POST_OFFSET" in assets

    # The channel must survive the parent rebuilding the slot per level.
    assert "_enable_channel(self.scene)" in guided


def test_evaluation_statistics_are_isaac_free_and_gate_is_explicit() -> None:
    evaluation = (SRC / "zero_g_blade_swap" / "evaluation.py").read_text(encoding="utf-8")
    aggregate = (SCRIPTS / "aggregate_evaluation.py").read_text(encoding="utf-8")

    assert "isaaclab" not in evaluation and "import torch" not in evaluation
    assert "def wilson_interval" in evaluation
    assert '"p95"' in evaluation
    assert 'TERMINATION_PRIORITY = (\n    "non_finite",\n    "mount_unstable",' in evaluation
    assert '"zero_instability_terminations"' in aggregate
    assert '"zero_non_finite_metric_episodes"' in aggregate
    assert '"every_stage_meets_minimum"' in aggregate
    assert '"simulation_only"' in aggregate
    assert '"grasp_model": "physx_fixed_joint_already_secured_abstraction"' in aggregate
    # A run pushed outside the training distribution must be labelled as an
    # envelope measurement and must not present a pass/fail certification gate.
    assert '"simulation_capability_envelope"' in aggregate
    assert 'gate["applies"] = False' in aggregate
    play = (SCRIPTS / "play.py").read_text(encoding="utf-8")
    assert '"--pose_noise_scale"' in play
    assert '"--blade_mass_range"' in play
    assert '"out_of_distribution"' in play


def test_training_and_playback_make_gpu_and_safety_evidence_explicit() -> None:
    train = (SCRIPTS / "train.py").read_text(encoding="utf-8")
    play = (SCRIPTS / "play.py").read_text(encoding="utf-8")

    assert "Simulation and PPO device" in train
    assert "torch.cuda.is_available()" in train
    assert '"termination_counts"' in play
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
        "state": (1024, 768, 512, 256),
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
