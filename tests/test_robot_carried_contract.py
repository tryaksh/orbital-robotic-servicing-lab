"""Import-free contracts for the robot-carried relocation.

The whole point of this branch is that the six-axis robot itself carries the
module, so the properties that make that claim true are the ones most worth
defending mechanically. Three of them are structural and can be checked from the
source without a simulator:

* the transit and the insertion command the *robot*, never the module;
* the form lock is released before the module is judged, so a settling check
  cannot be a statement about a joint;
* the world-mounted payload stage is not reachable from the robot-carried path.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_workflow_demo.py"
PRESETS = ROOT / "src/zero_g_blade_swap/service/presets.py"
GRAPPLE = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py"
SCENE = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/scene_cfg.py"
ASSETS = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/assets.py"


def test_the_rigid_carry_path_is_selected_only_by_the_form_lock() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert (
        "self.rigid_transit = release_latch_required and not base_rail_enabled and workflow == \"relocate\""
        in source
    )
    assert "if self.rigid_transit and bool(transiting.any()):" in source
    assert "elif bool(transiting.any()):" in source
    assert "if self.rigid_transit and bool(inserting.any()):" in source
    assert "elif bool(inserting.any()):" in source


def test_the_carried_transit_commands_a_tool_pose_derived_from_a_module_pose() -> None:
    """A module pose is inverted into a tool pose; the module is never written."""

    source = DRIVER.read_text(encoding="utf-8")
    assert "def _rigid_tool_command(" in source
    assert "tool_rot = quat_mul(module_rot, quat_inv(self.relocation_blade_relative_rot_to_tool[ids]))" in source
    # Inverted through the attitude the tool actually has, so a standing
    # attitude error does not become a standing position error.
    assert "tool_pos = module_pos - quat_apply(tool_rot_now, self.relocation_blade_relative_to_tool[ids])" in source
    assert "def _step_rigid_transit(" in source
    assert "self.module_leg_pos[leg, ids]" in source
    assert "self.module_leg_rot[leg, ids]" in source
    # No route from this path to the world-mounted stage.
    assert "self._engage_payload_stage" not in source.split("def _step_rigid_transit(")[1].split("def _front_overhang_x(")[0]


def test_the_retreat_is_derived_from_the_modules_measured_corners() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "def _front_overhang_x(" in source
    assert "measured_clear_centre_x = FLARE_LEADING_X - overhang - TRANSIT_FLARE_CLEARANCE_M" in source
    assert "TRANSIT_FLARE_CLEARANCE_M" in source


def test_the_guarded_insertion_advances_only_on_the_deployed_estimate() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("def _step_guarded_insert(")[1].split("def _front_overhang_x(")[0]
    assert "estimated_position, estimated_orientation, _velocity = self._payload_feedback()" in body
    assert "sensor_ready = estimator.fiducial_current_detection[ids]" in body
    assert "clear_to_advance = sensor_ready & (lateral_error <= lateral_tolerance)" in body
    assert "GUARDED_INSERT_AXIAL_STEP_M" in body
    assert "GUARDED_INSERT_MAX_LEAD_M" in body
    # The module's own pose is never written, only a tool command.
    assert "write_root_pose" not in body
    assert "root_state" not in body


def test_the_form_lock_is_released_before_the_module_is_judged() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("def _step_guarded_insert(")[1].split("def _front_overhang_x(")[0]
    assert "release_grapple_latch(task, due_to_release)" in body
    assert "return fired & ~grapple_latched(task)" in body
    assert "def _begin_guarded_insert(" in source
    assert "GUARDED_INSERT_RELEASE_MARGIN_M" in source
    assert "LATCH_MODULE_FACE_DEPTH_M" in source


def test_the_hand_opens_only_after_the_settling_recheck() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "verified = ripe & self.outcome & self.all_conditions" in source
    assert "self.gripper_released |= verified" in source
    assert "self.actions[finished & self.gripper_released, 6] = -1.0" in source


def test_the_release_disables_the_joint_and_stows_the_visible_jaws() -> None:
    source = GRAPPLE.read_text(encoding="utf-8")
    assert "def release(self, mask: torch.Tensor) -> None:" in source
    assert "self._fixed_joints[index].GetJointEnabledAttr().Set(False)" in source
    assert "self._set_jaw_pose(ids, engaged=False)" in source
    assert "def release_grapple_latch(env, mask: torch.Tensor) -> None:" in source


def test_the_latch_engagement_seeks_the_collar_and_refuses_what_it_cannot_reach() -> None:
    source = GRAPPLE.read_text(encoding="utf-8")
    assert "seek = _collar_shoulder_seek_m(env, blade)" in source
    assert "reachable = (seek >= service_latch.AXIAL_SEEK_RANGE_M[0])" in source
    assert "newly_latched = newly_latched & reachable" in source
    assert "env._grapple_latch_seek_refusals.add_(" in source


def test_the_latch_hardware_is_declared_in_the_scene_and_carries_no_collider() -> None:
    scene = SCENE.read_text(encoding="utf-8")
    assets = ASSETS.read_text(encoding="utf-8")
    assert "service_latch = AssetBaseCfg(" in scene
    assert "spawn=ServiceLatchCfg()" in scene
    assert "def spawn_service_latch(" in assets
    body = assets.split("def spawn_service_latch(")[1].split("class ServiceLatchCfg(")[0]
    # Visual only: the load path is the reported joint, and a collider here
    # would be a second, unreported one.
    assert "define_collision_properties" not in body
    assert "RigidBodyAPI" not in body
    assert "bind_visual_material" in body


def test_the_transit_records_the_tool_to_module_transform_throughout() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "TRANSIT_TRACE_FIELDS" in source
    assert "def _observe_transit_retention(" in source
    assert "TRANSIT_RETENTION_POSITION_LIMIT_M = INSERT_HANDOFF_POSITION_TOLERANCE_M" in source
    assert "TRANSIT_RETENTION_ORIENTATION_LIMIT_RAD = INSERTION_ORIENTATION_TOLERANCE_RAD" in source
    assert '"robot_carried_transit": _transit_retention_report(driver, args)' in source


def test_the_live_preset_does_not_use_the_world_mounted_payload_stage() -> None:
    """The rejected showcase, kept out of the default by a test rather than by memory."""

    source = PRESETS.read_text(encoding="utf-8")
    argv = source.split("argv = (")[1].split(")\n        return ExecutionSpec")[0]
    assert '"--base_rail_on_relocation"' not in argv
    assert '"--latch_on_release"' in argv
    assert '"--latch_joint_mode"' in argv
