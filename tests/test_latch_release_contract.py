"""Source contracts for the release-time capture latch.

These tests intentionally avoid importing the Isaac application.  They defend
the ordering boundary between the workflow driver and the interval event: the
driver must not unload the only predicate the event uses before engagement.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_workflow_demo.py"
GRAPPLE = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py"
GRAPPLE_CONFIG = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py"
ASSETS = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/assets.py"
SCENE = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/scene_cfg.py"


def test_release_latch_keeps_loaded_closure_until_engagement() -> None:
    source = DRIVER.read_text(encoding="utf-8")

    assert "release_latch_required=args.latch_on_release" in source
    assert "RELOCATE_TRANSIT_HOLD or self.release_latch_required" in source
    assert "awaiting_latch = transiting & ~latch_engaged" in source
    assert "self.gripper.retain_latch[awaiting_latch] = False" in source


def test_gentle_retain_requires_an_observed_latch_and_precedes_final_leg() -> None:
    source = DRIVER.read_text(encoding="utf-8")

    assert "may_retain = transiting & latch_engaged & (self.waypoint_read > 0)" in source
    assert "self.gripper.retain_latch[may_retain] = True" in source
    assert "self.gripper.retain_latch[due & (self.waypoint_read <= 1)] = False" in source


def test_moving_latch_wrench_is_reexpressed_in_the_current_body_frame() -> None:
    source = GRAPPLE.read_text(encoding="utf-8")

    assert "world_to_blade = quat_inv(blade.data.root_quat_w)" in source
    assert "force_b = quat_apply(world_to_blade, force)" in source
    assert "torque_b = quat_apply(world_to_blade, torque)" in source
    assert "forces=force_b.unsqueeze(1)" in source
    assert "torques=torque_b.unsqueeze(1)" in source
    assert "is_global=False" in source


def test_latch_design_point_is_configurable_and_reported() -> None:
    driver = DRIVER.read_text(encoding="utf-8")
    config = GRAPPLE_CONFIG.read_text(encoding="utf-8")
    implementation = GRAPPLE.read_text(encoding="utf-8")

    for argument in (
        "latch_position_stiffness_n_per_m",
        "latch_position_damping_ratio",
        "latch_rotation_stiffness_nm_per_rad",
        "latch_rotation_damping_ratio",
    ):
        assert f'"--{argument}"' in driver
        assert f"env_cfg.{argument}" in driver
        assert f"args.{argument}" in driver
        assert argument in config

    assert '"position_stiffness_n_per_m": (' in driver
    assert '"rotation_stiffness_nm_per_rad": (' in driver
    assert 'self.events.grapple_latch.params["position_stiffness"]' in config
    assert 'self.events.grapple_latch.params["rotation_stiffness"]' in config
    assert "rotation_damping = 2.0 * rotation_damping_ratio * torch.sqrt(" in implementation
    assert "rotation_stiffness * inertia" in implementation


def test_fixed_release_latch_is_physical_break_rated_and_capture_gated() -> None:
    driver = DRIVER.read_text(encoding="utf-8")
    implementation = GRAPPLE.read_text(encoding="utf-8")
    assets = ASSETS.read_text(encoding="utf-8")
    scene = SCENE.read_text(encoding="utf-8")

    assert 'choices=("compliant", "fixed")' in driver
    assert '"type": (' in driver and '"break_rated_fixed_joint"' in driver
    assert '"reaction_wrench_on_robot_modelled": (' in driver
    assert "class ReleaseLatchJointCfg(FixedGraspJointCfg):" in assets
    assert "joint.CreateBreakForceAttr().Set(cfg.break_force_n)" in assets
    assert "joint.CreateBreakTorqueAttr().Set(cfg.break_torque_nm)" in assets
    assert "enabled: bool = False" in assets
    assert 'prim_path="{ENV_REGEX_NS}/ReleaseLatchJoint"' in scene
    assert "newly_latched = qualified & ~latched" in implementation
    assert 'if joint_mode == "fixed":' in implementation
    # Not on an environment the driver has already softened for mating: that
    # one gave its joint up on purpose and must not have it re-armed.
    assert "self._engage_fixed_joints(env, blade, newly_latched & ~env._grapple_latch_compliant)" in implementation
    assert "joint.GetJointEnabledAttr().Set(True)" in implementation
    assert "blade.data.root_pos_w - wrist_position" in implementation
