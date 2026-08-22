"""Import-free contracts for the relocation-to-insertion hand-off."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_workflow_demo.py"
INSERTION = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/mdp/insertion.py"
GRAPPLE_CONFIG = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py"
ACTIONS = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/mdp/actions.py"


def test_relocation_targets_the_insert_reset_pose_from_the_captured_transform() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "TRANSIT_TARGET_BLADE_POSE" in source
    assert "blade_relative_to_tool = quat_apply(" in source
    assert "blade_relative_rot_to_tool = quat_mul(" in source
    assert "desired_tool_rot = quat_mul(" in source
    assert "desired_blade_rot, quat_inv(blade_relative_rot_to_tool)" in source
    assert "desired_blade_world = self.relocation_staging_pos.unsqueeze(0)" in source
    assert "final_tool_hold = desired_blade_world - quat_apply(" in source
    assert "final_tool_aligned = desired_blade_world - quat_apply(" in source
    assert "approach = final_tool_hold" in source


def test_relocation_separates_translation_from_rack_alignment() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "self.relocation_aligning[start_alignment] = True" in source
    assert "self.relocation_aligned[alignment_complete] = True" in source
    assert "self.waypoint_read[final_alignment_complete] = 0" in source
    assert "stage_translation_complete = (" in source
    assert "due = due & ~self.relocation_aligning" in source
    assert "self.relocation_alignment_tool_pos[start_ids] = blade_position - quat_apply(" in source
    assert "self.relocation_hold_tool_rot[ids]" in source
    assert "self.relocation_final_tool_aligned[ids]" in source
    assert "desired_tool_rot = torch.where(" in source
    assert "RELOCATE_FINAL_LEG_POSITION_AUTHORITY" in source
    assert "TRANSIT_HOLD_ATTITUDE_AUTHORITY" in source
    assert "TRANSIT_ALIGN_ATTITUDE_AUTHORITY" in source
    assert "quat_apply(quat_inv(root_quat), position_error)" in source
    assert "current_tool_rot = quat_mul(inverse_root, tool_rot[ids])" in source


def test_insert_handoff_is_fail_closed_on_pose_motion_and_capture() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    required = (
        "staging_position_error <= INSERT_HANDOFF_POSITION_TOLERANCE_M",
        "staging_orientation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD",
        "staging_linear_speed <= INSERTION_LINEAR_VELOCITY_LIMIT_MPS",
        "staging_angular_speed <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS",
        "secured = secured & grapple_latched(task)",
        "stage_blade_position_error <= INSERT_HANDOFF_POSITION_TOLERANCE_M",
        "stage_blade_orientation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD",
    )
    for expression in required:
        assert expression in source
    assert "self.phase[arrived] = INSERT" in source


def test_handoff_thresholds_share_the_insertion_success_contract() -> None:
    insertion = INSERTION.read_text(encoding="utf-8")
    assert "INSERTION_LATERAL_TOLERANCE_M = 0.0025" in insertion
    assert "INSERTION_ORIENTATION_TOLERANCE_RAD = 0.0523599" in insertion
    assert "INSERTION_LINEAR_VELOCITY_LIMIT_MPS = 0.030" in insertion
    assert "INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS = 0.080" in insertion
    assert "lateral <= INSERTION_LATERAL_TOLERANCE_M" in insertion
    assert "orientation <= INSERTION_ORIENTATION_TOLERANCE_RAD" in insertion


def test_clear_crossing_releases_attitude_only_for_the_fixed_latch() -> None:
    driver = DRIVER.read_text(encoding="utf-8")

    assert "crossing_clear = (" in driver
    assert 'getattr(task.cfg, "latch_joint_mode", "compliant") == "fixed"' in driver
    assert "torch.zeros_like(attitude_authority)" in driver
    assert "& (not self.base_rail_enabled)" in driver
    assert '"unconstrained_for_fixed_latch_only"' in driver
    assert "write_joint_state" not in driver
    assert '"robot_or_payload_state_writes": False' in driver


def test_optional_base_stage_commands_physical_six_dof_drives() -> None:
    driver = DRIVER.read_text(encoding="utf-8")
    config = GRAPPLE_CONFIG.read_text(encoding="utf-8")
    actions = ACTIONS.read_text(encoding="utf-8")

    assert '"--base_rail_on_relocation"' in driver
    assert '"--base_rail_arm_mode"' in driver
    assert "BASE_RAIL_TARGET_STEP_M = 0.0020" in driver
    assert "UsdPhysics.Tokens.transX, UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ" in driver
    assert "self.stage_drive_target_attributes.append(tuple(attributes))" in driver
    assert "UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ" in driver
    assert "self.stage_rotation_drive_target_attributes.append(tuple(rotation_attributes))" in driver
    assert "attribute.Set(float(self.stage_drive_target_m[stage_id, axis]))" in driver
    assert "attribute.Set(float(self.stage_rotation_drive_target_deg[pose_id, axis]))" in driver
    assert "position_action[stage_positioning] = 0.0" in driver
    assert '"physical_payload_shuttle_retreat_cross_align_and_guarded_insert"' in driver
    assert '"kinematic_mount_transform_writes": False' in driver
    assert '"physical_d6_translation_drive_target_writes"' in driver
    assert "self.arm.set_joint_hold_mask(rail_joint_hold)" in driver
    assert "self.payload_stage_joints[index].GetJointEnabledAttr().Set(False)" in driver
    assert "def _engage_payload_stage(self, env_ids: torch.Tensor)" in driver
    assert 'stage.GetPrimAtPath(f"/World/envs/env_{index}/ReleaseLatchJoint/Joint")' in driver
    assert "def set_joint_hold_mask(self, enabled: torch.Tensor)" in actions
    assert "self._asset.set_joint_position_target(" in actions
    assert '"arm_joint_position_target_hold": False if args.base_rail_on_relocation else None' in driver
    assert "begin_stage_alignment = (" in driver
    assert "& self.relocation_aligned" in driver
    assert "command_rack_attitude = command_rack_attitude | (self.waypoint_read[ids] == 1)" in driver
    assert '"robot_or_payload_state_writes": False' in driver
    assert 'robot = task.scene["robot"]' in driver
    assert 'anchor = task.scene["mount_anchor"]' in driver
    assert "robot.data.root_pos_w - anchor.data.root_pos_w" in driver
    assert "env_cfg.base_rail_enabled = args.base_rail_on_relocation" in driver
    assert "base_rail_enabled: bool = False" in config
    assert "def configure_base_rail(self) -> None:" in config
    assert "self.scene.robot = make_grapple_pin_robot_cfg(floating=False)" in config
    assert "spawn=CompliantD6JointCfg(" in config
    assert 'body1_relative_path="SpareBlade"' in config
    assert "relocate_robot_articulation_root=False" in config
    assert "enabled=False" in config
    assert "translation_x_lower_limit=-0.200" in config
    assert "translation_x_upper_limit=0.800" in config
    assert "translation_y_lower_limit=-0.600" in config
    assert "translation_y_upper_limit=0.200" in config
    assert "translation_z_lower_limit=-0.400" in config
    assert "translation_z_upper_limit=0.400" in config
    assert "rotation_limit_deg=20.0" in config
    assert "env_cfg.configure_robustness(0)" in driver
    assert "env_cfg.configure_base_rail()" in driver
    assert driver.index("env_cfg.configure_robustness(0)") < driver.index("env_cfg.configure_base_rail()")
    assert "BASE_STAGE_MIN_TARGET_M = (-0.200, -0.600, -0.400)" in driver
    assert "BASE_STAGE_MAX_TARGET_M = (0.800, 0.200, 0.400)" in driver
    assert "stage_goal[:, 1] += self.relocation_staging_pos[1] - blade_local[:, 1]" in driver
    assert "stage_error = self.stage_goal_target_m[stage_ids] - self.stage_drive_target_m[stage_ids]" in driver
    assert "previous_target + target_delta" in driver
    assert "stage_tracking_error = actual_travel - self.stage_drive_target_m" in driver
    assert '"stage_terminal_drive_target_xyz_m"' in driver
    assert '"stage_terminal_rotation_drive_target_xyz_deg"' in driver
