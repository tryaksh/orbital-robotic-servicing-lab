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

import pytest

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
    assert "direct_insert = self.rigid_transit or self.insert_only" in source
    assert "elif direct_insert and bool(inserting.any()):" in source
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


def test_the_form_lock_has_three_states_and_ends_in_none_of_them() -> None:
    """Rigid to carry, compliant to mate, released before the module is judged."""

    source = DRIVER.read_text(encoding="utf-8")
    transit = source.split("def _step_rigid_transit(")[1].split("def _front_overhang_x(")[0]
    body = source.split("def _step_guarded_insert(")[1].split("def _front_overhang_x(")[0]
    # Transport ends and mating begins where the module meets the rack -- and
    # that is a *place*, not a leg boundary. Softening on the step the leg
    # changed put the interface's biggest state change on the same control step
    # as a 450 mm step change in the target, and the module went from 14.8 mrad
    # to 65 in sixteen steps. The trigger is now the module's leading face
    # reaching the lead-in, which is what the softening is for.
    assert "module_front_x = module_pos[:, 0] + 0.5 * BLADE_LENGTH_M" in transit
    assert "FLARE_LEADING_X - MATING_SOFTEN_LEAD_M" in transit
    assert "(leg == 0)" in transit
    assert "soften_grapple_latch(task, mating)" in transit
    # The seating re-check is taken on a module the lock is no longer holding.
    assert "release_grapple_latch(task, fired)" in body
    # Rigid mating is a measured alternative, so the gate is on the mode
    # rather than on the lock unconditionally.
    assert "~grapple_latch_rigid(task)" in body
    assert 'MATING_MODE == "rigid"' in body
    # And the geometric interlock still forces a *rigid* lock off.
    assert "release_grapple_latch(task, due_to_release)" in body
    assert "GUARDED_INSERT_RELEASE_MARGIN_M" in source
    assert "LATCH_MODULE_FACE_DEPTH_M" in source


def test_softening_keeps_the_load_path_and_stores_no_energy() -> None:
    source = GRAPPLE.read_text(encoding="utf-8")
    body = source.split("def soften(self, mask")[1].split("def release(self, mask")[0]
    # The joint goes away, the spring takes over from where the module *is*.
    assert "GetJointEnabledAttr().Set(False)" in body
    assert "env._grapple_latch_relative_pos[ids] = quat_apply(" in body
    assert "env._grapple_latch_compliant[ids] = True" in body
    # Still engaged: softening is not releasing.
    assert "_grapple_latched" not in body


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


def test_the_preset_relief_is_the_geometry_the_task_derives_it_from() -> None:
    """The bay's clearance is stated in two places and must not drift.

    ``presets`` cannot import the task's asset module, which needs Isaac, so it
    re-derives the relief from the same simulator-free geometry. Checking the
    derivation here is what keeps the dashboard running the bay that was
    measured rather than one that used to be.
    """

    from zero_g_blade_swap.grapple_geometry import BLADE_LENGTH_M
    from zero_g_blade_swap.service import presets

    assert pytest.approx(
        0.5 * BLADE_LENGTH_M * presets.SETTLED_ATTITUDE_RAD
    ) == presets.DESTINATION_CHANNEL_RELIEF_M
    # And the cap is the stiffness times the stroke, not a round number: past
    # the stroke the compliance is at its hard stop and the cap does nothing.
    assert pytest.approx(40_000.0 * 0.025) == presets.MATING_FORCE_CAP_N


def test_the_report_says_the_insertion_is_scripted_on_the_carried_path() -> None:
    """The one label this project's honesty rests on, keyed on the right thing.

    ``_step_guarded_insert`` is selected by ``rigid_transit`` -- relocate, with
    the form lock, without the payload shuttle -- and on that path the learned
    insert policy is loaded, hashed, and never asked for an action. The label
    branched on ``base_rail_on_relocation`` instead, so every robot-carried
    report claimed ``insert`` as a *learned* phase with an empty unexecuted
    list, for a policy certified at 10.50% pooled and 0.00% in its near stage.
    """

    source = DRIVER.read_text(encoding="utf-8")
    # Keyed on the controller: relocate, with the form lock, without the payload
    # shuttle, and with the guarded advance selected rather than the policy.
    guarded = source.split("guarded_insert = (")[1].split("insert_only =")[0]
    assert 'args.workflow == "relocate"' in guarded
    assert "args.latch_on_release" in guarded
    assert "not args.base_rail_on_relocation" in guarded
    assert 'args.insert_controller == "guarded"' in guarded
    # A checkpoint that is loaded and never consulted has to say so -- and since
    # 2026-08-24 the chain can also decline to load one at all, which is what
    # section 10.2 of the interface specification actually asks for. The label
    # has to survive both cases: named when carried and unused, absent when not
    # carried, and never claimed as a learned phase either way.
    assert 'if guarded_insert and "insert" in policies:' in source
    assert 'loaded_but_not_executed.append("insert")' in source
    assert '"loaded_but_not_executed_policies": loaded_but_not_executed' in source
    assert 'if args.insert_checkpoint is not None:' in source
    assert 'parser.error("--insert_controller policy needs an --insert_checkpoint to run")' in source
    # And the phase it does run is named in the scripted list.
    assert '["transit", "guarded_insert"]' in source


def test_the_learned_insert_arm_exists_and_is_labelled_when_it_runs() -> None:
    """"The policy is not used" has to be a measurement, so it must be runnable.

    With ``--insert_controller policy`` the trained checkpoint drives the seating
    and the labels follow: ``insert`` becomes a learned phase, the scripted list
    drops ``guarded_insert``, and there is no unexecuted policy left to declare.
    Nothing about that is a separate code path for the report to describe --
    ``guarded_insert`` is simply false, and every label already keys on it.
    """

    source = DRIVER.read_text(encoding="utf-8")
    assert '"--insert_controller"' in source
    assert 'choices=("guarded", "policy")' in source
    assert "def _step_policy_insert(" in source
    assert "insert_controller=args.insert_controller," in source
    # The policy arm is not silently guarded: it releases the lock and enforces
    # the geometric interlock, and it does not gate the advance on an envelope.
    body = source.split("def _step_policy_insert(")[1].split("def _step_guarded_insert(")[0]
    assert "release_grapple_latch(task" in body
    assert "clear_to_advance" not in body

# ---------------------------------------------------------------------------
# The scripted legs' controller. Solved inverse kinematics, commanded as
# actuator targets -- which is the one property that makes it a controller
# rather than a pose write, and therefore the one worth defending mechanically.


def test_the_solved_legs_command_actuator_targets_and_nothing_else() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "def _command_solved_tool_pose(" in source
    body = source.split("def _command_solved_tool_pose(")[1].split("def _apply_joint_overrides(")[0]
    assert "batched_solve_ik(" in body
    # Actuator targets only. Not a joint-state write, not a root pose write, and
    # not a module pose write -- the three things this branch exists to not do.
    for forbidden in (
        "write_joint_state_to_sim",
        "write_root_pose_to_sim",
        "write_root_state_to_sim",
        "set_joint_position",
    ):
        assert forbidden not in body, forbidden
    assert "set_joint_target_override" in source


def test_the_solved_setpoint_is_absolute_rather_than_pose_relative() -> None:
    """The whole point: the command is not anchored on the measured tool pose.

    A setpoint advanced from *itself* converges to the leg target and stops. One
    advanced from the tool's current pose plus a delta integrates the joints' lag
    into the command, which is what limit-cycled the squaring leg at about one
    action scale.
    """

    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("def _command_solved_tool_pose(")[1].split("def _apply_joint_overrides(")[0]
    assert "setpoint_pos = setpoint_pos + (target_pos - setpoint_pos).clamp(-scale[:3], scale[:3])" in body
    assert "quat_mul(target_rot, quat_inv(setpoint_rot))" in body
    # Seeded at the pose the arm is in, once per leg, so a leg boundary is not a
    # step change in the command.
    assert "def _seed_solved_setpoints(" in source
    seed_body = source.split("def _seed_solved_setpoints(")[1].split("def _command_solved_tool_pose(")[0]
    assert "self.solved_setpoint_pos[ids] = tool[ids]" in seed_body


def test_the_solved_solve_is_checked_against_the_simulator_before_it_commands() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "def _check_forward_kinematics(" in source
    body = source.split("def _check_forward_kinematics(")[1].split("def _seed_solved_setpoints(")[0]
    assert "raise RuntimeError(" in body
    assert "SOLVED_IK_FK_AGREEMENT_M" in body
    # And it runs before the first solved command, not after it.
    step_body = source.split("def _step_rigid_transit(")[1]
    assert step_body.index("self._check_forward_kinematics(") < step_body.index(
        "self._command_solved_tool_pose("
    )


def test_the_joint_permutation_is_resolved_rather_than_assumed() -> None:
    """``find_joints`` does not promise the order it was given."""

    source = DRIVER.read_text(encoding="utf-8")
    assert "self.arm_dh_permutation = torch.tensor(" in source
    assert "[term_names.index(name) for name in JOINT_ORDER]" in source
    assert "Refusing to solve inverse kinematics for a chain that is not this arm." in source


def test_the_mating_steps_hand_back_to_the_controller_that_has_a_trim() -> None:
    """A softened lock is not a weld, so the exact inversion stops being exact."""

    source = DRIVER.read_text(encoding="utf-8")
    step_body = source.split("def _step_rigid_transit(")[1].split("def _begin_guarded_insert(")[0]
    assert "solved = grapple_latch_rigid(task)[ids] & ~self.latch_softened[ids]" in step_body
    assert "follower = ~solved" in step_body
    assert "self._rigid_tool_command(" in step_body


def test_a_per_environment_attitude_authority_is_broadcast_per_row() -> None:
    """The missing axis that kept every rigid-transit result at n = 1.

    ``rotation_error`` is (environments, 3); a per-environment authority is
    (environments,). ``torch.clamp`` broadcasts from the trailing axis, so the
    two align only at one or three environments and raise at any other count --
    mid-transit, whenever the tenth or thirty-second environment entered a leg.
    """

    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("def _drive_tool_to(")[1].split("def _check_forward_kinematics(")[0]
    assert "if attitude_authority.dim() == 1:" in body
    assert "attitude_authority = attitude_authority.unsqueeze(-1)" in body


def test_the_hand_off_depth_condition_is_one_sided() -> None:
    """Deeper than the staging pose is progress, and must not stall the chain."""

    source = DRIVER.read_text(encoding="utf-8")
    assert (
        "pose_ready = (position_error[:, 0] <= INSERTION_AXIAL_DEPTH_TOLERANCE_M) & (" in source
    )


def test_a_vision_run_refuses_to_close_its_loop_on_simulator_truth() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("def _payload_feedback(")[1].split("def _trace_state(")[0]
    assert 'if "Vision" in args.task:' in body
    assert "run a perception claim on an oracle." in body
    # The fall-through to simulator truth is still there for the state task,
    # which is what a state task is -- it is reached only after that guard.
    assert body.index('if "Vision" in args.task:') < body.index("attached_blade_pose_world(self.task)")


def test_the_report_says_which_controller_flew_the_legs() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert '"scripted_leg_controller"' in source
    assert '"solved_inverse_kinematics_enabled": TRANSIT_SOLVED_IK,' in source
    assert '"actuator_targets_only": True,' in source


def test_the_report_says_whether_the_base_mount_carries_anything() -> None:
    """A zero deflection is what a stiff mount and an absent one both look like."""

    source = DRIVER.read_text(encoding="utf-8")
    assert "def _base_mount_compliance_report(" in source
    body = source.split("def _base_mount_compliance_report(")[1].split("def _transit_retention_report(")[0]
    assert '"in_load_path": in_load_path,' in body
    assert '"claim_supported_about_base_compliance_tolerance": in_load_path,' in body
    # And a contradiction between the configuration and the measurement stops
    # the report rather than appearing in it.
    assert "raise RuntimeError(" in body


def test_the_guarded_envelope_is_the_flare_and_says_why() -> None:
    """A depth-dependent bound was built, measured against, and refuted.

    ``2c/l`` is the law, but ``c`` is the lead-in gap near the mouth and the
    channel gap deeper in -- 8.00 mm against 12.61 mm on this rack -- so a bound
    built from the lead-in gap alone gives 35.6 mrad at full engagement, and the
    chain seats modules at 46.7. The bound would have held exactly the runs that
    succeed. The envelope stays what the entry flare catches, and the report
    carries the reason so it is not rediscovered.
    """

    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("def _step_guarded_insert(")[1].split("def _front_overhang_x(")[0]
    assert "admissible" not in body
    assert '"why_not_depth_dependent"' in source
    # The hand-off gate keeps the lead-in number, deliberately conservative.
    assert "RELOCATION_HANDOFF_ATTITUDE_RAD = 2.0 * LEAD_IN_VERTICAL_HALF_GAP_M / BLADE_SIZE[0]" in source


def test_the_seating_controller_carries_its_own_action_scale() -> None:
    """The insert policy's scale belongs to the policy, not to the phase."""

    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("def _apply_scales(")[1].split("def _set_stage_arm_servo(")[0]
    assert "self._guarded_receiver()" in body
    assert "self.scales[TRANSIT]" in body


def test_the_last_transit_leg_leaves_the_module_at_the_mouth_for_both_controllers() -> None:
    """Both controllers get the same pose, and that is what makes them comparable.

    They did not always. The guarded advance has no reset pose -- its
    precondition is that the module is inside the bay's lead-in catch, already
    true at the mouth -- while the learned insert policy had exactly one, deep
    inside the slot, so the leg had to choose. Choosing wrong is what made the
    transit perform the insertion and left the phase labelled "insert" with
    0.7 mm of it.

    Since ``mdp.reset_grapple_insert_stroke``, the policy's reset spans the whole
    stroke and the mouth is inside its distribution too. So the leg stops
    choosing, both controllers start from the same place, and
    ``--insert_controller`` compares two controllers rather than two hand-offs.
    The old behaviour stays reachable for replaying an archived checkpoint, and
    it is off.

    ``_guarded_receiver`` still has to be read at call time, because it still
    selects the seating *scale*: ``MATING_MODE`` is overwritten from the command
    line long after import, so a module constant computed from it is a constant
    computed from the default.
    """

    source = DRIVER.read_text(encoding="utf-8")
    assert "GUARDED_RECEIVER" not in source
    assert "self.module_leg_pos[0, ids] = staging if LEG_ZERO_AT_POLICY_RESET else crossed" in source
    assert 'LEG_ZERO_AT_POLICY_RESET = os.environ.get("LEG_ZERO_AT_POLICY_RESET", "0") == "1"' in source
    body = source.split("def _guarded_receiver(")[1].split("def _apply_scales(")[0]
    assert 'self.insert_controller == "guarded"' in body
    assert 'MATING_MODE == "compliant"' in body
