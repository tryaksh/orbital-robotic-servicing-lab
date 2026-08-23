"""MDP terms for the three head-on grapple-pin skills.

Grasp, extract, and insert are separate skills with separate gates, so they get
separate terms rather than one shared reward with mode flags. What they have in
common is the frame they are scored in: the tool frame sits at the centre of the
measured 105-to-162 mm pad span, and the blade carries a matching point at the
centre of the length of pin the pads close on.

Two things here differ from the insertion terms in ``insertion.py`` and are easy
to get wrong.

*The blade is not welded to the tool.* Insertion's `tool_to_handle_error_m` is a
tautology on the rigid-grasp task, because a fixed joint holds the blade at the
frame the metric compares against. Here nothing holds the blade but contact, so
the same quantity is a real measurement of whether the grip is still there.

*The workspace predicate is inverted for extraction.* `insertion_failure` treats
a blade at x below 0.45 as an escape, which is exactly where a successful
extraction ends. Extraction therefore carries its own bounds.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import ActionTerm, EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    compute_pose_error,
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from pxr import Gf, UsdGeom, UsdPhysics

from zero_g_blade_swap import service_latch
from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    EXTRACTED_BLADE_CENTRE_X,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_PIN_COLLAR_X,
    GRAPPLE_PIN_GRIP_OFFSET,
    SLOT_MOUTH_X,
)
from zero_g_blade_swap.tasks.blade_swap.assets import (
    SERVICE_LATCH_JAWS,
    SERVICE_LATCH_PRIM,
    service_latch_jaw_translation,
)

from .actions import (
    ROBOTIQ_2F85_COUPLING_SIGNS,
    ROBOTIQ_2F85_JOINT_NAMES,
    RobotiqBinaryAction,
    RobotiqBinaryActionCfg,
    robotiq_2f85_coupled_targets,
)
from .insertion import (
    INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS,
    INSERTION_AXIAL_DEPTH_TOLERANCE_M,
    INSERTION_LATERAL_TOLERANCE_M,
    INSERTION_LINEAR_VELOCITY_LIMIT_MPS,
    INSERTION_ORIENTATION_TOLERANCE_RAD,
    attached_blade_pose_world,
    attached_blade_velocity,
    insertion_error_metrics,
)
from .observations import end_effector_pose_world

# A grasp counts as formed only when the drive torque rises off its noise floor,
# which sits at 1e-5 N-m. This threshold is the same one grasp_diagnostics.py
# gates on, so the trained skill and the physics characterisation agree on what
# the word means.
GRIP_TORQUE_THRESHOLD_NM = 0.05

#: Seconds the chained workflow holds still before re-checking that a phase
#: really succeeded. ``run_workflow_demo.SETTLE_STEPS`` is this at 30 Hz.
WORKFLOW_SETTLE_S = 0.70
#: Stroke of the modelled mating compliance, in metres.
#:
#: An assembly compliance device is a spring *and a hard stop*. 25 mm is what
#: this one needs to do its job -- the rack's lead-in has to be able to push the
#: module about 3 mm off the tool's line to square it, and the module lags the
#: tool by a few millimetres under the insertion's own contact reaction -- with
#: room left over rather than a limit sized at the measurement.
MATING_TRAVEL_LIMIT_M = 0.025
#: The same, in rotation. 0.2 rad is the attitude tolerance section 8 of the
#: interface specification requires the interface to hold through a pull, so a
#: compliance that gave up more than that would be giving up the requirement.
MATING_ROTATION_LIMIT_RAD = 0.2

#: Grip attitude a capture is allowed, and the margin an extraction has to hand
#: over inside it. ``capture_established`` keys on the first.
CAPTURE_ATTITUDE_TOLERANCE_RAD = 0.20
CAPTURE_POSITION_TOLERANCE_M = 0.020
#: Grip error the chained workflow requires before it will hand a captured
#: module to the next skill. The capture skill's own certification tolerance has
#: to be at least this strict or the two disagree about what "captured" means --
#: and they did. Certified on 20 mm while the chain waited for 10, capture
#: overran its 6 s budget in 77 of the chain's 113 failures, 68% of them, while
#: scoring 96.10% alone. ``run_workflow_demo`` reads this rather than restating
#: it.
WORKFLOW_HANDOVER_GRIP_M = 0.010

#: How fast a "removed" module may still be moving, **derived** from the two
#: numbers above rather than chosen.
#:
#: The chain stops commanding the moment a phase succeeds and re-checks the same
#: condition ``WORKFLOW_SETTLE_S`` later. In zero gravity nothing damps what is
#: left moving, so a module declared removed at velocity ``v`` drifts ``v * t``
#: before it is judged. With the old limits -- 0.30 rad/s and 0.10 m/s -- that
#: is 0.21 rad and 70 mm of drift against a 0.20 rad and 20 mm tolerance, so the
#: skill was allowed to declare success in states the chain could not possibly
#: confirm. Measured exactly there: 191 of 192 chained removals fired the
#: predicate and none survived the re-check.
#:
#: Half the remaining margin is spent on drift, which leaves the other half for
#: the attitude the pull itself ends with.
EXTRACTION_ANGULAR_VELOCITY_LIMIT = 0.5 * CAPTURE_ATTITUDE_TOLERANCE_RAD / WORKFLOW_SETTLE_S
EXTRACTION_LINEAR_VELOCITY_LIMIT = 0.5 * CAPTURE_POSITION_TOLERANCE_M / WORKFLOW_SETTLE_S


def grapple_grip_pose_error(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tool-to-grip-point vector in world axes and the orientation error.

    The orientation error is measured against the head-on capture attitude
    rather than against the blade, because the pin's axis is the blade's own
    x axis and the gripper has to arrive along it.
    """

    tool_position, tool_orientation = end_effector_pose_world(env)
    blade_position, blade_orientation = attached_blade_pose_world(env)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    # The capture attitude is expressed relative to the blade, so a blade that
    # has been knocked askew moves the target rather than hiding the error.
    desired = quat_mul(blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation))
    angle = torch.linalg.vector_norm(axis_angle_from_quat(quat_mul(desired, quat_inv(tool_orientation))), dim=-1)
    return tool_position - grip_position, angle


def grapple_grip_error_metrics(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tool-to-grip distance and capture-attitude error magnitudes."""

    vector, angle = grapple_grip_pose_error(env)
    return torch.linalg.vector_norm(vector, dim=-1), angle


def grapple_grip_attitude_error_world(env) -> torch.Tensor:
    """Signed world-frame rotation vector between the tool and the capture attitude.

    ``grapple_grip_pose_error`` returns this vector's *magnitude*. The latch
    needs its direction as well, because a restoring torque has to know which
    way to push.
    """

    _, tool_orientation = end_effector_pose_world(env)
    _, blade_orientation = attached_blade_pose_world(env)
    desired_tool = quat_mul(
        blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation)
    )
    return axis_angle_from_quat(quat_mul(desired_tool, quat_inv(tool_orientation)))


class GrappleLatch(ManagerTermBase):
    """A bounded compliant model of a form-locking capture latch.

    **Why this exists.** A parallel-jaw grip on a passive feature cannot resist a
    moment about the closing axis: the pads' contact normals lie along that axis
    and a normal force cannot oppose a moment about its own direction. This
    project measured that four independent ways -- extraction certifying at
    0.00% while holding grip *position* for a full 15 s pull, 93% of insertion
    failures ending outside the grip-attitude tolerance, 3.8% of chained
    removals ending inside it, and an anti-yaw yoke recovering only 12% of the
    rotation while costing the insert skill 67 points. That yoke has since been
    deleted; its measurements are kept in docs/status.md.

    Flight servicing hardware does not solve this with friction either. The
    SSRMS latching end effector snares a grapple fixture and then *rigidizes*
    it; Dextre's ORU Tool Changeout Mechanism grips a standardised fixture and
    carries a powered socket drive. The load path after capture is form closure
    through a latch, not friction on a passive feature. So the interface
    specification's answer to yaw is a latch, and this is its model.

    **What it does, exactly, so nothing is hidden.** It engages after a capture
    qualifies and, when ``require_armed`` is set, only after the workflow says
    the module is clear of the rails. At engagement it records the actual
    tool-to-module transform, then applies bounded, damped restoring force and
    torque to preserve it. This is the load path a form-closing mechanism
    supplies after the fingers have acquired the pin.

    **What it deliberately does not do.**

    * It has a finite rating, so "the latch holds" is a number rather than an
      assumption, and the force/torque rating the workflow actually needs is the
      specification deliverable this term exists to produce.
    * It engages only on a capture the policy achieved. A policy that never
      arrives never gets one.
    * The equal and opposite reaction on the wrist is **not** modelled. The arm
      is position-controlled through differential IK with stiff drives, which
      would absorb most of it, but this is a simplification and not a free
      lunch: a real latch loads the wrist and the joint torques it implies are
      unmeasured here.

    With ``rated_force_n=0`` the term is the earlier torque-only experiment.
    That mode is retained so its negative evidence remains reproducible.
    """

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        env._grapple_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        #: Whether a driver has said the module is free of the rack yet. Only
        #: read when ``require_armed`` is set, and ``True`` otherwise, so a task
        #: that does not ask for it is bit-identical to before this existed.
        env._grapple_latch_armed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._grapple_latch_relative_pos = torch.zeros((env.num_envs, 3), device=env.device)
        env._grapple_latch_relative_quat = torch.zeros((env.num_envs, 4), device=env.device)
        env._grapple_latch_relative_quat[:, 0] = 1.0
        # Runtime evidence for the modelled interface.  Configuration proves
        # only that the term exists; these values prove whether it engaged,
        # whether it preserved the transform it captured, and whether the
        # requested motion exceeded either finite load rating.
        env._grapple_latch_first_engagement_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        #: Episode step the driver released the form lock, or -1. A latch that
        #: is never released is not a latch, it is a weld, and the difference
        #: has to be in the record rather than in the prose.
        env._grapple_latch_release_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        #: Carriage travel each engagement needed to find the collar shoulder.
        env._grapple_latch_seek_travel_m = torch.zeros(env.num_envs, device=env.device)
        #: Engagements refused because the shoulder was outside that travel.
        env._grapple_latch_seek_refusals = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        #: Whether this environment's lock has been **softened**: still engaged,
        #: still the load path, but a spring instead of a weld.
        #:
        #: A one-state lock cannot do this job. Rigid, it carries the module
        #: through free space -- 2.3 mm of drift against 808 mm on the pads
        #: alone -- and then cannot mate, because a lead-in aligns a part by
        #: pushing it and a part welded to a stiff arm will not be pushed:
        #: measured, 0.3 mm of advance in a thirty-second seating budget.
        #: Released instead, the module slides out of the bay sideways, because
        #: the pads do not resist lateral load. Both are measured in this
        #: repository and neither is a controller problem.
        #:
        #: Industrial assembly has had the answer since the 1970s and flight
        #: servicing uses it too: rigidize to carry, de-rigidize to mate. The
        #: SSRMS latching end effector snares, rigidizes, and releases rigidity
        #: to berth. This flag is that second state.
        env._grapple_latch_compliant = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        #: Whether the compliance ever ran out of stroke, which is this
        #: interface's version of dropping the module.
        env._grapple_latch_overtravel = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        # So a driver can ask the term to let go without reaching into it.
        env._grapple_latch_term = self
        env._grapple_latch_position_error_m = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_orientation_error_rad = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_applied_force_n = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_applied_torque_nm = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_max_position_error_m = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_max_orientation_error_rad = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_max_applied_force_n = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_max_applied_torque_nm = torch.zeros(env.num_envs, device=env.device)
        env._grapple_latch_force_saturated = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._grapple_latch_torque_saturated = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._grapple_latch_force_saturation_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._grapple_latch_torque_saturation_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        body_ids, body_names = env.scene["robot"].find_bodies(["wrist_3_link"], preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(f"GrappleLatch expected wrist_3_link, resolved {body_names}")
        self._wrist_body_id = int(body_ids[0])
        self._fixed_joints: list[UsdPhysics.FixedJoint] = []
        #: The compliant state's joint, one per environment. Absent on tasks
        #: that do not declare it, in which case softening falls back to the
        #: explicit wrench and says so by leaving this empty.
        self._mating_joints: list[UsdPhysics.Joint] = []
        stage_for_mating = get_current_stage()
        for env_path in env.scene.env_prim_paths:
            prim = stage_for_mating.GetPrimAtPath(f"{env_path}/MatingComplianceJoint/Joint")
            if not prim.IsValid():
                self._mating_joints = []
                break
            self._mating_joints.append(UsdPhysics.Joint(prim))
        # Translate ops of the visible jaw carriages, when the scene carries the
        # latch hardware. Empty on tasks that do not, so every task that existed
        # before the mechanism did behaves exactly as it did.
        self._jaw_translate_ops: list[tuple[object, object]] = []
        stage = get_current_stage()
        for env_path in env.scene.env_prim_paths:
            ops = []
            for jaw in SERVICE_LATCH_JAWS:
                prim = stage.GetPrimAtPath(
                    f"{env_path}/Robot/wrist_3_link/{SERVICE_LATCH_PRIM}/{jaw}"
                )
                if not prim.IsValid():
                    ops = []
                    break
                translate = next(
                    (
                        op
                        for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    ),
                    None,
                )
                if translate is None:
                    raise RuntimeError(f"Service latch jaw '{prim.GetPath()}' has no translate op")
                ops.append(translate)
            if not ops:
                self._jaw_translate_ops = []
                break
            self._jaw_translate_ops.append(tuple(ops))
        if self.cfg.params.get("joint_mode", "compliant") == "fixed":
            stage = get_current_stage()
            for env_path in env.scene.env_prim_paths:
                joint = UsdPhysics.FixedJoint(
                    stage.GetPrimAtPath(f"{env_path}/ReleaseLatchJoint/Joint")
                )
                if not joint or not joint.GetPrim().IsValid():
                    raise RuntimeError(
                        f"Fixed release latch is missing at '{env_path}/ReleaseLatchJoint/Joint'"
                    )
                self._fixed_joints.append(joint)

    def reset(self, env_ids=None) -> None:
        ids = torch.arange(self._env.num_envs, device=self._env.device) if env_ids is None else env_ids
        self._env._grapple_latched[ids] = False
        if bool(self.cfg.params.get("require_armed", False)):
            self._env._grapple_latch_armed[ids] = False
        self._env._grapple_latch_relative_pos[ids] = 0.0
        self._env._grapple_latch_relative_quat[ids] = 0.0
        self._env._grapple_latch_relative_quat[ids, 0] = 1.0
        self._env._grapple_latch_first_engagement_step[ids] = -1
        self._env._grapple_latch_release_step[ids] = -1
        self._env._grapple_latch_seek_travel_m[ids] = 0.0
        self._env._grapple_latch_seek_refusals[ids] = 0
        self._env._grapple_latch_compliant[ids] = False
        self._env._grapple_latch_overtravel[ids] = False
        self._env._grapple_latch_position_error_m[ids] = 0.0
        self._env._grapple_latch_orientation_error_rad[ids] = 0.0
        self._env._grapple_latch_applied_force_n[ids] = 0.0
        self._env._grapple_latch_applied_torque_nm[ids] = 0.0
        self._env._grapple_latch_max_position_error_m[ids] = 0.0
        self._env._grapple_latch_max_orientation_error_rad[ids] = 0.0
        self._env._grapple_latch_max_applied_force_n[ids] = 0.0
        self._env._grapple_latch_max_applied_torque_nm[ids] = 0.0
        self._env._grapple_latch_force_saturated[ids] = False
        self._env._grapple_latch_torque_saturated[ids] = False
        self._env._grapple_latch_force_saturation_count[ids] = 0
        self._env._grapple_latch_torque_saturation_count[ids] = 0
        if self._fixed_joints:
            for index in ids.detach().cpu().tolist():
                self._fixed_joints[index].GetJointEnabledAttr().Set(False)
        if self._mating_joints:
            for index in ids.detach().cpu().tolist():
                self._mating_joints[index].GetJointEnabledAttr().Set(False)
        self._set_jaw_pose(ids, engaged=False)
        zeros = torch.zeros((len(ids), 1, 3), device=self._env.device)
        self._env.scene["spare_blade"].permanent_wrench_composer.set_forces_and_torques(
            forces=zeros, torques=zeros, env_ids=ids, is_global=True
        )

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rated_torque_nm: float = 5.0,
        rotation_stiffness: float = 10.0,
        rotation_damping_ratio: float = 0.9,
        rated_force_n: float = 0.0,
        position_stiffness: float = 2_500.0,
        position_damping_ratio: float = 0.9,
        joint_mode: str = "compliant",
        require_armed: bool = False,
        mating_force_cap_n: float = 400.0,
        mating_torque_cap_nm: float = 40.0,
    ) -> None:
        del env_ids
        if joint_mode not in ("compliant", "fixed"):
            raise ValueError(f"joint_mode must be 'compliant' or 'fixed', got {joint_mode!r}")
        latched = env._grapple_latched
        # **When the latch engages is a physical question, and this project has
        # measured half the answer.** Swept from 10 to 160 N-m against the
        # unchanged extract policy, a latch engaged on capture never moved the
        # rotation it was aimed at and collapsed extraction travel from 465 mm
        # to about 25 mm: a restoring torque on a module the rails still hold
        # jams the module in the rails.
        #
        # That refutes engaging it on capture. It says nothing about engaging it
        # the instant the rails let go, which is the phase that actually needs
        # it -- the module is unconstrained, nothing else opposes a moment about
        # the closing axis, and there are no rails left to jam it against. With
        # ``require_armed`` the driver picks that moment.
        qualified = capture_established(env)
        if require_armed:
            qualified = qualified & env._grapple_latch_armed
        newly_latched = qualified & ~latched
        blade = env.scene[asset_cfg.name]
        tool_position, tool_orientation = end_effector_pose_world(env)
        # **The carriage seeks the collar; it does not assume it.** Where the
        # pin sits along the approach axis is set by where its taper thickness
        # equals the pad opening, so it moves with the closure command -- the
        # interface specification measures about 12.5 mm of self-seating travel
        # on every capture. A latch authored at the nominal collar depth would
        # close on the collar's rim. Measure the shoulder, and refuse the
        # engagement when it is outside the carriage's travel rather than
        # pretending the mechanism can reach it.
        seek = _collar_shoulder_seek_m(env, blade)
        reachable = (seek >= service_latch.AXIAL_SEEK_RANGE_M[0]) & (
            seek <= service_latch.AXIAL_SEEK_RANGE_M[1]
        )
        env._grapple_latch_seek_refusals.add_((newly_latched & ~reachable).to(torch.long))
        newly_latched = newly_latched & reachable
        qualified = qualified & (latched | reachable)
        if bool(newly_latched.any()):
            env._grapple_latch_seek_travel_m[newly_latched] = seek[newly_latched]
            self._set_jaw_pose(
                torch.nonzero(newly_latched, as_tuple=False).squeeze(-1),
                engaged=True,
                seek=seek,
            )
            inverse_tool = quat_inv(tool_orientation[newly_latched])
            env._grapple_latch_relative_pos[newly_latched] = quat_apply(
                inverse_tool,
                blade.data.root_pos_w[newly_latched] - tool_position[newly_latched],
            )
            env._grapple_latch_relative_quat[newly_latched] = quat_mul(
                inverse_tool,
                blade.data.root_quat_w[newly_latched],
            )
            env._grapple_latch_first_engagement_step[newly_latched] = env.episode_length_buf[
                newly_latched
            ].to(torch.long)
            if joint_mode == "fixed":
                # A softened environment has already given its joint up; do not
                # re-arm it behind the driver's back.
                self._engage_fixed_joints(env, blade, newly_latched & ~env._grapple_latch_compliant)
        latched |= qualified

        desired_position = tool_position + quat_apply(tool_orientation, env._grapple_latch_relative_pos)
        desired_orientation = quat_mul(tool_orientation, env._grapple_latch_relative_quat)
        # Use the same pose-error convention as the already exercised rigid-
        # grasp insertion fixture.  The first translational prototype formed
        # this difference by hand and paired it with an unrelated canonical
        # capture-attitude error.  In a relocation that inconsistency injected
        # energy until the module escaped the tool; the apparent hand-off to
        # insertion was therefore a false positive, not a successful latch.
        position_error, rotation_error = compute_pose_error(
            blade.data.root_pos_w,
            blade.data.root_quat_w,
            desired_position,
            desired_orientation,
            rot_error_type="axis_angle",
        )
        position_error_norm = torch.linalg.vector_norm(position_error, dim=-1)
        rotation_error_norm = torch.linalg.vector_norm(rotation_error, dim=-1)
        env._grapple_latch_position_error_m.copy_(
            torch.where(latched, position_error_norm, torch.zeros_like(position_error_norm))
        )
        env._grapple_latch_orientation_error_rad.copy_(
            torch.where(latched, rotation_error_norm, torch.zeros_like(rotation_error_norm))
        )
        env._grapple_latch_max_position_error_m.copy_(
            torch.maximum(env._grapple_latch_max_position_error_m, env._grapple_latch_position_error_m)
        )
        env._grapple_latch_max_orientation_error_rad.copy_(
            torch.maximum(
                env._grapple_latch_max_orientation_error_rad,
                env._grapple_latch_orientation_error_rad,
            )
        )
        # Which environments the spring is responsible for. On a compliant task
        # that is all of them; on a fixed task it is the ones the driver has
        # softened, and for the rest the joint is the load path and applying the
        # wrench as well would double-count it.
        # When the compliant state is a joint, the joint is the load path and
        # the wrench would double-count it -- the same rule the fixed state
        # already follows.
        if self._mating_joints and joint_mode != "compliant":
            return
        spring = latched if joint_mode == "compliant" else (latched & env._grapple_latch_compliant)
        if not bool(spring.any()):
            return
        latched = spring
        # **Finite travel, and it is not a numerical guard.**
        #
        # A compliance device has a stroke. Past it there is a hard stop, and
        # past *that* the part is no longer held. Without the limit the spring
        # is a formula rather than a mechanism, and a formula with no cap feeds
        # itself: measured, the module left the cell at 15 km when a contact
        # transient outran the damping at the 30 Hz command rate.
        #
        # Exceeding it is recorded as what it is -- the interface losing the
        # module -- rather than smoothed away.
        overtravelled = latched & (position_error_norm > MATING_TRAVEL_LIMIT_M)
        env._grapple_latch_overtravel |= overtravelled
        position_error = position_error.clamp(-MATING_TRAVEL_LIMIT_M, MATING_TRAVEL_LIMIT_M)
        rated_force_n = min(rated_force_n, mating_force_cap_n)
        rated_torque_nm = min(rated_torque_nm, mating_torque_cap_nm)
        robot = env.scene["robot"]
        wrist_position = robot.data.body_pos_w[:, self._wrist_body_id]
        wrist_linear_velocity = robot.data.body_lin_vel_w[:, self._wrist_body_id]
        wrist_angular_velocity = robot.data.body_ang_vel_w[:, self._wrist_body_id]
        wrist_to_blade = desired_position - wrist_position
        desired_linear_velocity = wrist_linear_velocity + torch.cross(
            wrist_angular_velocity,
            wrist_to_blade,
            dim=-1,
        )
        force = torch.zeros_like(position_error)
        force_demand_norm = torch.zeros(env.num_envs, device=env.device)
        if rated_force_n > 0.0:
            if position_stiffness <= 0.0:
                raise ValueError(f"position_stiffness must be positive, got {position_stiffness}")
            if not 0.0 < position_damping_ratio <= 2.0:
                raise ValueError(
                    f"position_damping_ratio must be in (0, 2], got {position_damping_ratio}"
                )
            masses = _blade_mass(env, asset_cfg)
            position_damping = 2.0 * position_damping_ratio * torch.sqrt(position_stiffness * masses)
            force = position_stiffness * position_error + position_damping.unsqueeze(-1) * (
                desired_linear_velocity - blade.data.root_lin_vel_w
            )
            force_norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True).clamp_min(1.0e-9)
            force_demand_norm = force_norm.squeeze(-1)
            force = force / (force_norm / rated_force_n).clamp_min(1.0)
        force = force * latched.unsqueeze(-1)

        torque = torch.zeros_like(rotation_error)
        torque_demand_norm = torch.zeros(env.num_envs, device=env.device)
        if rated_torque_nm > 0.0:
            if rotation_stiffness <= 0.0:
                raise ValueError(f"rotation_stiffness must be positive, got {rotation_stiffness}")
            if not 0.0 < rotation_damping_ratio <= 2.0:
                raise ValueError(
                    f"rotation_damping_ratio must be in (0, 2], got {rotation_damping_ratio}"
                )
            inertia = _blade_yaw_inertia(env, asset_cfg)
            rotation_damping = 2.0 * rotation_damping_ratio * torch.sqrt(
                rotation_stiffness * inertia
            )
            torque = rotation_stiffness * rotation_error + rotation_damping.unsqueeze(-1) * (
                wrist_angular_velocity - blade.data.root_ang_vel_w
            )
            # Compliance and load rating are different physical properties.
            # Coupling them made a larger claimed rating a stiffer controller
            # and catapulted the module.  These gains match the exercised
            # secured-blade fixture; the rating is solely a norm cap.
            torque_norm = torch.linalg.vector_norm(torque, dim=-1, keepdim=True).clamp_min(1.0e-9)
            torque_demand_norm = torque_norm.squeeze(-1)
            torque = torque / (torque_norm / rated_torque_nm).clamp_min(1.0)
        torque = torque * latched.unsqueeze(-1)

        applied_force = torch.linalg.vector_norm(force, dim=-1)
        applied_torque = torch.linalg.vector_norm(torque, dim=-1)
        force_saturated = latched & (rated_force_n > 0.0) & (force_demand_norm > rated_force_n)
        torque_saturated = latched & (rated_torque_nm > 0.0) & (torque_demand_norm > rated_torque_nm)
        env._grapple_latch_applied_force_n.copy_(applied_force)
        env._grapple_latch_applied_torque_nm.copy_(applied_torque)
        env._grapple_latch_max_applied_force_n.copy_(
            torch.maximum(env._grapple_latch_max_applied_force_n, applied_force)
        )
        env._grapple_latch_max_applied_torque_nm.copy_(
            torch.maximum(env._grapple_latch_max_applied_torque_nm, applied_torque)
        )
        env._grapple_latch_force_saturated.copy_(force_saturated)
        env._grapple_latch_torque_saturated.copy_(torque_saturated)
        env._grapple_latch_force_saturation_count.add_(force_saturated.to(torch.long))
        env._grapple_latch_torque_saturation_count.add_(torque_saturated.to(torch.long))
        # WrenchComposer converts a global wrench through a link pose cached on
        # its first call and then applies the composed wrench as body-local on
        # every physics step.  That is correct for a fixed link and unstable for
        # this rotating payload: a nominally restoring world force rotated with
        # the module until it accelerated the module away (measured 109 m grip
        # error under a continuously saturated 150 N command). Convert from the
        # desired world wrench with the *current* module attitude ourselves on
        # every interval event, then store an explicitly local wrench.
        world_to_blade = quat_inv(blade.data.root_quat_w)
        force_b = quat_apply(world_to_blade, force)
        torque_b = quat_apply(world_to_blade, torque)
        blade.permanent_wrench_composer.set_forces_and_torques(
            forces=force_b.unsqueeze(1),
            torques=torque_b.unsqueeze(1),
            is_global=False,
        )

    def soften(self, mask: torch.Tensor) -> None:
        """Rigid to compliant, without letting go and without a snap.

        The fixed joint is disabled and the spring takes over from the transform
        the module is in **at this instant**, not the one recorded at
        engagement, so the change of state stores no energy and moves nothing.
        The lock is still the load path and still resists the lateral load the
        pads cannot; what it stops doing is refusing to yield to the rack.
        """

        env = self._env
        ids = mask.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        if self._fixed_joints:
            for index in ids.detach().cpu().tolist():
                self._fixed_joints[index].GetJointEnabledAttr().Set(False)
        tool_position, tool_orientation = end_effector_pose_world(env)
        blade = env.scene["spare_blade"]
        inverse_tool = quat_inv(tool_orientation[ids])
        env._grapple_latch_relative_pos[ids] = quat_apply(
            inverse_tool, blade.data.root_pos_w[ids] - tool_position[ids]
        )
        env._grapple_latch_relative_quat[ids] = quat_mul(inverse_tool, blade.data.root_quat_w[ids])
        env._grapple_latch_compliant[ids] = True
        if self._mating_joints:
            # The spring's rest pose is the wrist-to-module transform *now*, so
            # switching states stores no energy and moves nothing. Written to
            # the joint's own frames rather than commanded, because a drive
            # target would have to chase a moving wrist.
            # **The compliance centre goes at the module's leading face, not at
            # the wrist, and that is what makes it a *remote* centre of
            # compliance rather than a spring.**
            #
            # With the centre at the tool, a contact force on the entering tip
            # acts through a 225 mm arm and rotates the module *into* the wall
            # it just touched. That is the jamming mode of a clearance fit, and
            # it is what the measurements show: pushed at 20 kN the module moved
            # 0.1 mm, pushed at 400 N into a channel opened to 4.5 mm it moved
            # 2.4 mm and rotated further off square as it went, 17.2 mrad to
            # 22.3 mrad. Opening the channel does not fix it and neither does
            # pushing harder, because the jam condition is a ratio of moment to
            # force rather than a clearance.
            #
            # Put the centre at the tip and the same contact force translates
            # the module instead of rotating it. This is the entire reason
            # assembly compliance devices are built with a remote centre, and
            # the distance is not a parameter to tune: it is where the part
            # touches the hole.
            robot = env.scene["robot"]
            wrist_position = robot.data.body_pos_w[:, self._wrist_body_id]
            wrist_orientation = robot.data.body_quat_w[:, self._wrist_body_id]
            inverse_wrist = quat_inv(wrist_orientation)
            tip_in_blade = blade.data.root_pos_w.new_tensor((0.5 * BLADE_LENGTH_M, 0.0, 0.0)).expand(
                env.num_envs, -1
            )
            tip_world = blade.data.root_pos_w + quat_apply(blade.data.root_quat_w, tip_in_blade)
            local_position = quat_apply(inverse_wrist, tip_world - wrist_position)
            local_orientation = quat_mul(inverse_wrist, blade.data.root_quat_w)
            for index in ids.detach().cpu().tolist():
                position = local_position[index].detach().cpu().tolist()
                orientation = local_orientation[index].detach().cpu().tolist()
                joint = self._mating_joints[index]
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
                joint.GetLocalRot0Attr().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
                joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.5 * BLADE_LENGTH_M, 0.0, 0.0))
                joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                joint.GetJointEnabledAttr().Set(True)

    def release(self, mask: torch.Tensor) -> None:
        """Let the module go: disable the joint, drop the wrench, stow the jaws.

        Called by the driver at the instant the transit hands over, because the
        form lock is for *transport*. Driving the module back between rails is
        the certified insert policy's job and it was trained on pad contact; a
        joint left enabled through insertion would be carrying the seating
        contact, and the settling check at the end would then be measuring a
        weld rather than a seated module.
        """

        env = self._env
        ids = mask.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        newly_released = mask & env._grapple_latched
        env._grapple_latch_release_step[newly_released] = env.episode_length_buf[newly_released].to(
            torch.long
        )
        env._grapple_latched[ids] = False
        env._grapple_latch_armed[ids] = False
        env._grapple_latch_compliant[ids] = False
        if self._mating_joints:
            for index in ids.detach().cpu().tolist():
                self._mating_joints[index].GetJointEnabledAttr().Set(False)
        if self._fixed_joints:
            for index in ids.detach().cpu().tolist():
                self._fixed_joints[index].GetJointEnabledAttr().Set(False)
        zeros = torch.zeros((ids.numel(), 1, 3), device=env.device)
        env.scene["spare_blade"].permanent_wrench_composer.set_forces_and_torques(
            forces=zeros, torques=zeros, env_ids=ids, is_global=True
        )
        self._set_jaw_pose(ids, engaged=False)

    def _set_jaw_pose(self, ids, *, engaged: bool, seek: torch.Tensor | None = None) -> None:
        """Move the visible carriages between their two commanded positions."""

        if not self._jaw_translate_ops:
            return
        indices = ids.detach().cpu().tolist() if hasattr(ids, "detach") else list(ids)
        travel = None if seek is None else seek.detach().cpu().tolist()
        for index in indices:
            for op, sign in zip(self._jaw_translate_ops[index], (1.0, -1.0), strict=True):
                op.Set(
                    Gf.Vec3d(
                        *service_latch_jaw_translation(
                            engaged=engaged,
                            seek_m=0.0 if travel is None else float(travel[index]),
                            sign=sign,
                        )
                    )
                )

    def _engage_fixed_joints(self, env, blade, mask: torch.Tensor) -> None:
        """Capture the current wrist-to-payload frame, then enable the joint."""

        if len(self._fixed_joints) != env.num_envs:
            raise RuntimeError(
                f"Expected {env.num_envs} fixed release joints, found {len(self._fixed_joints)}"
            )
        robot = env.scene["robot"]
        wrist_position = robot.data.body_pos_w[:, self._wrist_body_id]
        wrist_orientation = robot.data.body_quat_w[:, self._wrist_body_id]
        inverse_wrist = quat_inv(wrist_orientation)
        local_position = quat_apply(
            inverse_wrist,
            blade.data.root_pos_w - wrist_position,
        )
        local_orientation = quat_mul(inverse_wrist, blade.data.root_quat_w)
        for index in mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
            position = local_position[index].detach().cpu().tolist()
            orientation = local_orientation[index].detach().cpu().tolist()
            joint = self._fixed_joints[index]
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
            joint.GetLocalRot0Attr().Set(
                Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:]))
            )
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.GetJointEnabledAttr().Set(True)


def _blade_yaw_inertia(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Payload inertia about the axis the latch resists, per environment.

    Read from PhysX rather than assumed, because Level 2 randomizes the module's
    mass over 5 to 15 kg and recomputes its inertia; a damping constant sized
    against a fixed 10 kg would be wrong at both ends of that range.
    """

    cached = getattr(env, "_grapple_latch_inertia", None)
    if cached is not None:
        return cached
    view = env.scene[asset_cfg.name].root_physx_view
    # Diagonal of the inertia tensor, largest principal moment: the latch
    # resists rotation of a long thin module, so this is the conservative pick.
    inertia = view.get_inertias().to(env.device).reshape(env.num_envs, -1, 3, 3)[:, 0]
    diagonal = torch.stack((inertia[:, 0, 0], inertia[:, 1, 1], inertia[:, 2, 2]), dim=-1)
    env._grapple_latch_inertia = diagonal.amax(dim=-1).clamp_min(1e-4)
    return env._grapple_latch_inertia


def _blade_mass(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Payload mass for critical damping of the translational latch."""

    cached = getattr(env, "_grapple_latch_mass", None)
    if cached is not None:
        return cached
    env._grapple_latch_mass = (
        env.scene[asset_cfg.name].root_physx_view.get_masses().to(env.device)[:, 0].clamp_min(1.0e-4)
    )
    return env._grapple_latch_mass


def _collar_shoulder_seek_m(env, blade) -> torch.Tensor:
    """Carriage travel needed to put the jaw lips behind the collar shoulder.

    The depth of the module's collar shoulder from the flange, along the wrist's
    own approach axis, minus the depth the latch is authored at. Zero means the
    module is seated exactly where the geometry assumed.
    """

    robot = env.scene["robot"]
    wrist_body = getattr(env, "_grapple_wrist_body_id", None)
    if wrist_body is None:
        body_ids, _names = robot.find_bodies(["wrist_3_link"], preserve_order=True)
        wrist_body = int(body_ids[0])
        env._grapple_wrist_body_id = wrist_body
    wrist_position = robot.data.body_pos_w[:, wrist_body]
    wrist_orientation = robot.data.body_quat_w[:, wrist_body]
    shoulder_local = blade.data.root_pos_w.new_tensor((GRAPPLE_PIN_COLLAR_X[1], 0.0, 0.0)).expand(
        env.num_envs, -1
    )
    shoulder_world = blade.data.root_pos_w + quat_apply(blade.data.root_quat_w, shoulder_local)
    depth = quat_apply(quat_inv(wrist_orientation), shoulder_world - wrist_position)[:, 2]
    return depth - service_latch.COLLAR_SHOULDER_FROM_FLANGE_M


def soften_grapple_latch(env, mask: torch.Tensor) -> None:
    """Switch the form lock from rigid to compliant. No-op where none is fitted."""

    term = getattr(env, "_grapple_latch_term", None)
    if term is None:
        return
    term.soften(mask)


def grapple_latch_rigid(env) -> torch.Tensor:
    """Per environment: is the lock currently a weld rather than a spring."""

    latched = getattr(env, "_grapple_latched", None)
    if latched is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    compliant = getattr(env, "_grapple_latch_compliant", None)
    return latched if compliant is None else (latched & ~compliant)


def release_grapple_latch(env, mask: torch.Tensor) -> None:
    """Tell the form lock to let the module go. A no-op where none is fitted."""

    term = getattr(env, "_grapple_latch_term", None)
    if term is None:
        return
    term.release(mask)


def arm_grapple_latch(env, mask: torch.Tensor) -> None:
    """Tell a ``require_armed`` latch that these modules are free of the rack.

    Called by the chained driver at the same instant it retains the grip, which
    is the extract skill's own success mask: the module is out, still gripped,
    and no longer moving. A no-op when no latch is configured, so a driver may
    call it unconditionally.
    """

    state = getattr(env, "_grapple_latch_armed", None)
    if state is None:
        return
    state[mask] = True


def grapple_latched(env) -> torch.Tensor:
    """Per environment: has a qualifying capture engaged the latch this episode."""

    state = getattr(env, "_grapple_latched", None)
    if state is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return state


def grapple_latch_diagnostics(env) -> dict[str, torch.Tensor]:
    """Read behavior-neutral runtime evidence for the compliant form lock.

    A task without the latch event returns an all-zero, never-engaged record so
    callers can instrument latch-on and latch-off workflows through one path.
    Values are per environment and are updated by :class:`GrappleLatch` at the
    30 Hz control rate.
    """

    zeros = torch.zeros(env.num_envs, device=env.device)
    false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    first = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def read(name: str, fallback: torch.Tensor) -> torch.Tensor:
        value = getattr(env, name, None)
        return fallback if value is None else value

    return {
        "engaged": read("_grapple_latched", false),
        "first_engagement_episode_step": read("_grapple_latch_first_engagement_step", first),
        "release_episode_step": read("_grapple_latch_release_step", first),
        "seek_travel_m": read("_grapple_latch_seek_travel_m", zeros),
        "seek_refusals": read("_grapple_latch_seek_refusals", counts),
        "compliant": read("_grapple_latch_compliant", false),
        "overtravel": read("_grapple_latch_overtravel", false),
        "position_error_m": read("_grapple_latch_position_error_m", zeros),
        "orientation_error_rad": read("_grapple_latch_orientation_error_rad", zeros),
        "applied_force_n": read("_grapple_latch_applied_force_n", zeros),
        "applied_torque_nm": read("_grapple_latch_applied_torque_nm", zeros),
        "max_position_error_m": read("_grapple_latch_max_position_error_m", zeros),
        "max_orientation_error_rad": read("_grapple_latch_max_orientation_error_rad", zeros),
        "max_applied_force_n": read("_grapple_latch_max_applied_force_n", zeros),
        "max_applied_torque_nm": read("_grapple_latch_max_applied_torque_nm", zeros),
        "force_saturated": read("_grapple_latch_force_saturated", false),
        "torque_saturated": read("_grapple_latch_torque_saturated", false),
        "force_saturation_count": read("_grapple_latch_force_saturation_count", counts),
        "torque_saturation_count": read("_grapple_latch_torque_saturation_count", counts),
    }


def grapple_grip_attitude_axes(env) -> torch.Tensor:
    """Decompose the capture-attitude error into the tool's own three axes.

    ``grapple_grip_error_metrics`` reports the *magnitude* of that rotation, and
    a magnitude cannot say which axis it is about. That distinction decides
    whether an anti-yaw feature can help at all, and it is why the two that were
    built could not: both opposed rotation about the **closing** axis alone,
    because that is the axis whose moment a pair of flat pad normals cannot
    resist, while the rotation is split almost evenly between the closing axis
    at 0.198 rad and the transverse one at 0.199 rad.

    Measured on the yoked pin, extraction still ended at 0.279 rad against a
    0.125 rad geometric prediction, so wall clearance was never the term that
    dominates. That yoke's code was deleted on 2026-08-18; this decomposition is
    what refuted it, and it is kept because the next interface proposal will
    need the same test. See docs/status.md.

    Returns one row per environment as ``(closing, third, approach)`` in the
    tool frame: the measured 2F-85 closes along its own x and approaches along
    its own z (``evidence/gripper_collision_envelope.json``).
    """

    tool_position, tool_orientation = end_effector_pose_world(env)
    _, blade_orientation = attached_blade_pose_world(env)
    desired = quat_mul(
        blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation)
    )
    # Expressed in the tool's frame rather than the world's, so the components
    # are the gripper's own closing/approach axes at whatever attitude the arm
    # happens to hold.
    relative = quat_mul(quat_inv(tool_orientation), desired)
    return axis_angle_from_quat(relative).abs()


def grapple_grip_error_observation(env) -> torch.Tensor:
    """Six values: the tool-to-grip offset in tool axes, and the attitude error."""

    tool_position, tool_orientation = end_effector_pose_world(env)
    blade_position, blade_orientation = attached_blade_pose_world(env)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    desired = quat_mul(blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation))
    relative_position, relative_quat = subtract_frame_transforms(
        tool_position, tool_orientation, grip_position, desired
    )
    return torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)


def _finger_joint_ids(env) -> list[int]:
    ids = getattr(env, "_grapple_finger_joint_ids", None)
    if ids is None:
        ids, names = env.scene["robot"].find_joints(list(ROBOTIQ_2F85_JOINT_NAMES), preserve_order=True)
        if tuple(names) != ROBOTIQ_2F85_JOINT_NAMES:
            raise RuntimeError(f"unexpected Robotiq joint order: {names}")
        env._grapple_finger_joint_ids = ids
    return ids


def grip_drive_torque(env) -> torch.Tensor:
    """Return the 2F-85 drive torque, the signal that a grip is actually loaded."""

    robot = env.scene["robot"]
    return robot.data.applied_torque[:, _finger_joint_ids(env)[0]].abs()


def grip_finger_angle(env) -> torch.Tensor:
    """Return the 2F-85 drive joint angle in radians. Zero is fully open.

    The convention is the measured one -- pad separation falls monotonically with
    the command over the joint's whole range, 87.08 mm at 0 to 0 mm at 0.8203 --
    and it is stated here because this project carried it inverted for three
    sessions. See evidence/gripper_collision_envelope.json.
    """

    return env.scene["robot"].data.joint_pos[:, _finger_joint_ids(env)[0]]


def gripper_state_observation(env) -> torch.Tensor:
    """Return the finger command's reached angle and the drive torque it develops.

    Both are needed: the angle alone cannot distinguish fingers closed on a pin
    from fingers closed on nothing, which is the failure this project spent
    three sessions not seeing.
    """

    robot = env.scene["robot"]
    joint_ids = _finger_joint_ids(env)
    angle = robot.data.joint_pos[:, joint_ids[0]].unsqueeze(-1)
    torque = grip_drive_torque(env).unsqueeze(-1)
    return torch.cat((angle / 0.8203, torque / 10.0), dim=-1)


def capture_established(
    env,
    torque_threshold: float = GRIP_TORQUE_THRESHOLD_NM,
    position_tolerance: float = 0.020,
    orientation_tolerance: float = 0.20,
) -> torch.Tensor:
    """True where the pads are loaded against the pin at the capture attitude."""

    position, orientation = grapple_grip_error_metrics(env)
    return (
        (grip_drive_torque(env) > torque_threshold)
        & (position <= position_tolerance)
        & (orientation <= orientation_tolerance)
    )


class TwoStageRobotiqAction(RobotiqBinaryAction):
    """Capture gently, then firm up once the grip is loaded.

    This is not a refinement, it is the difference between passing the gate and
    failing it. A wedge converts closing force into thrust along the pull axis,
    so a hard capture drives the payload away before it has been taken; but
    holding wants all the force the drive can produce. Measured axial capacity
    against a single command was 59 N; capturing at 0.48 rad and firming to
    0.68 once the grip is established gives 69 N, against a 66.4 N gate.

    The window is not wide, and it is asymmetric. Capturing at 0.44 gives 63 N
    and at 0.52 gives 68 N, but 0.56 collapses to 26 N. Bias low.
    See evidence/grapple_pin_capture_plateau.json.
    """

    cfg: TwoStageRobotiqActionCfg

    def __init__(self, cfg: TwoStageRobotiqActionCfg, env) -> None:
        super().__init__(cfg, env)
        #: Per environment: "the part is taken, stop reconsidering".
        #:
        #: :func:`capture_established` goes false as soon as grip error passes
        #: 20 mm, which rail contact does routinely, and this term would then
        #: drop back to the gentler capture closure and open the fingers about
        #: 21 mm mid-task. Right while capturing, catastrophic afterwards. A
        #: driver that has decided the hand-off happened sets this for the
        #: environments it applies to.
        #:
        #: It is a tensor rather than a configuration value because a config
        #: field is one scalar shared by every environment: overwriting
        #: ``cfg.closed_position`` at run time, which is what the workflow driver
        #: used to do, latches all of them at once and is only correct when there
        #: is exactly one. Nothing in a training task sets this, so every trained
        #: policy still sees the behaviour its certification was produced under.
        self.hold_latch = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        #: Per environment: "the job is done, stop squeezing".
        #:
        #: Holding and *retaining* are as different as capturing and holding, and
        #: for the same reason. Both the capture and hold commands drive far past
        #: the 0.223 rad the pads come to rest at on the wedge, so the drive
        #: saturates at its 10 N-m limit either way, and a wedge converts that
        #: into thrust along the pull axis. On a railed module that is harmless
        #: because the rails absorb it. On a module that has just been pulled
        #: free it is a continuous push with nothing in zero gravity to oppose
        #: it: traced through the chain's settling window, an extraction that
        #: fires its predicate at 0.008 m/s accelerates monotonically to
        #: 0.103 m/s over 0.70 s at a constant 10 N-m, with the grip never
        #: slipping. That is why every chained removal this project has run has
        #: fired its predicate and failed the re-check.
        #:
        #: ``retain_position`` sits just above the seated angle, so the pads stay
        #: on the wedge and the drive is barely loaded. It takes priority over
        #: the hold latch, and nothing in a training task sets it.
        self.retain_latch = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Robotiq action must have shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}."
            )
        self._raw_actions.copy_(actions)
        closing = actions > self.cfg.threshold if self.cfg.close_on_positive else actions < self.cfg.threshold
        established = capture_established(self._env).unsqueeze(-1) | self.hold_latch
        target = torch.where(
            closing & self.retain_latch,
            torch.full_like(actions, self.cfg.retain_position),
            torch.where(
                closing & established,
                torch.full_like(actions, self.cfg.hold_position),
                torch.where(
                    closing,
                    torch.full_like(actions, self.cfg.closed_position),
                    torch.full_like(actions, self.cfg.open_position),
                ),
            ),
        )
        self._processed_actions.copy_(robotiq_2f85_coupled_targets(target))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # A new episode has not taken anything yet, and has certainly not
        # finished with it. Both latches must clear: leaving ``retain_latch`` set
        # carries a finished episode's gentle closure into the next episode's
        # capture, which measured as exactly half the workflows failing, because
        # each environment's second episode began already relaxed.
        ids = env_ids if env_ids is not None else slice(None)
        self.hold_latch[ids] = False
        self.retain_latch[ids] = False


@configclass
class TwoStageRobotiqActionCfg(RobotiqBinaryActionCfg):
    """Configuration for :class:`TwoStageRobotiqAction`."""

    class_type: type[ActionTerm] = TwoStageRobotiqAction
    #: Commanded once the grip is loaded. ``closed_position`` is the capture
    #: command, which is deliberately gentler.
    hold_position: float = 0.68
    #: Commanded once a driver latches ``retain_latch``: enough to keep the pads
    #: on the wedge, not enough to drive against it. The pads come to rest at
    #: 0.223 rad on the wedge (``GRAPPLE_FINGER_SEATED_RAD``), so 0.25 leaves
    #: 0.027 rad of overdrive against the 0.457 rad that ``hold_position``
    #: applies, and the drive stops saturating.
    #:
    #: Defaults to ``hold_position``'s value so that a task which never sets
    #: ``retain_latch`` -- which is every training task -- is bit-identical.
    retain_position: float = 0.25


def _hold_counter(env, name: str, active: torch.Tensor, hold_time_s: float) -> torch.Tensor:
    """Count consecutive steps a condition has held, once per environment step."""

    counter = getattr(env, name, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, name, counter)
    step_name = f"{name}_step"
    current_step = int(env.common_step_counter)
    if getattr(env, step_name, -1) != current_step:
        counter.copy_(torch.where(active, counter + 1, torch.zeros_like(counter)))
        setattr(env, step_name, current_step)
    return counter >= max(1, int(round(hold_time_s / float(env.step_dt))))


def _potential_reward(
    env, key: str, cost: torch.Tensor, scale: float, clamp: float, settle_s: float = 0.30
) -> torch.Tensor:
    """Pay for the measured reduction in a cost since the previous step.

    Potential-based, so a policy cannot farm it by oscillating: every metre
    gained is paid once and every metre lost is charged once.

    The validity flag is not optional. A reset moves the arm and the blade
    discontinuously, and without it the first step of every episode is charged
    the whole jump from the previous episode's final cost, which shows up as a
    large spurious reward of either sign. The insertion terms carry the same
    flag for the same reason.
    """

    previous = getattr(env, f"_grapple_prev_{key}", None)
    valid = getattr(env, f"_grapple_valid_{key}", None)
    reward = getattr(env, f"_grapple_reward_{key}", None)
    if previous is None or previous.shape[0] != env.num_envs:
        previous = torch.zeros_like(cost)
        valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        reward = torch.zeros_like(cost)
        setattr(env, f"_grapple_prev_{key}", previous)
        setattr(env, f"_grapple_valid_{key}", valid)
        setattr(env, f"_grapple_reward_{key}", reward)
    step_name = f"_grapple_step_{key}"
    current_step = int(env.common_step_counter)
    if getattr(env, step_name, -1) != current_step:
        # A reset writes joint positions but leaves the previous episode's
        # actuator targets in place, so the arm springs for a few steps before
        # it holds. Paying for that motion is paying for nothing the policy did.
        settled = env.episode_length_buf.to(torch.float32) * float(env.step_dt) >= settle_s
        reward.copy_(
            torch.where(valid & settled, ((previous - cost) / scale).clamp(-clamp, clamp) / float(env.step_dt), 0.0)
        )
        previous.copy_(cost)
        valid.fill_(True)
        setattr(env, step_name, current_step)
    return reward


def reset_grapple_progress(env, env_ids: torch.Tensor | None) -> None:
    """Clear every per-episode counter this module keeps on the environment."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    for name in ("_grapple_capture_hold", "_grapple_extract_hold", "_grapple_insert_hold"):
        value = getattr(env, name, None)
        if value is not None and value.shape[0] == env.num_envs:
            value[ids] = 0
    # The holding closure latches for the episode, so it has to be released here
    # or a new episode would start already believing it holds the module.
    latched = getattr(env, "_grapple_hold_latched", None)
    if latched is not None and latched.shape[0] == env.num_envs:
        latched[ids] = False
    # Invalidate rather than zero the potentials: zeroing would charge the next
    # step the entire distance from the goal.
    for key in ("capture", "extract"):
        valid = getattr(env, f"_grapple_valid_{key}", None)
        if valid is not None and valid.shape[0] == env.num_envs:
            valid[ids] = False


def capture_success_mask(
    env,
    hold_time_s: float = 0.30,
    position_tolerance: float = WORKFLOW_HANDOVER_GRIP_M,
) -> torch.Tensor:
    """Terminate a grasp episode once a loaded capture has been held.

    ``position_tolerance`` defaults to what the *chain* demands, not to
    ``capture_established``'s looser 20 mm. A skill certified more loosely than
    the workflow it belongs to will pass alone and stall in the chain, which is
    exactly what happened here. The looser default stays on
    ``capture_established`` itself because extraction and insertion are
    certified against it and their numbers must not move underneath them.
    """

    established = capture_established(env, position_tolerance=position_tolerance)
    return _hold_counter(env, "_grapple_capture_hold", established, hold_time_s)


def capture_success_reward(env, hold_time_s: float = 0.30) -> torch.Tensor:
    return capture_success_mask(env, hold_time_s).to(torch.float32) / float(env.step_dt)


def capture_approach_reward(env, distance_scale: float = 0.050) -> torch.Tensor:
    """Pay for closing on the grip point; standing still earns nothing.

    Potential-based, so a policy cannot farm this by oscillating: the reward is
    the measured reduction in a pose cost since the previous step.
    """

    placed = _blade_reset_pose(env)
    if placed is None:
        return torch.zeros(env.num_envs, device=env.device)
    blade_position, blade_orientation = placed
    tool_position, tool_orientation = end_effector_pose_world(env)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    desired = quat_mul(
        blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation)
    )
    position = torch.linalg.vector_norm(tool_position - grip_position, dim=-1)
    orientation = torch.linalg.vector_norm(
        axis_angle_from_quat(quat_mul(desired, quat_inv(tool_orientation))), dim=-1
    )
    cost = position / max(distance_scale, 1.0e-6) + 0.5 * orientation / 0.20
    return _potential_reward(env, "capture", cost, scale=1.0, clamp=0.25)


def blade_disturbance_penalty(env, free_m: float = 0.018) -> torch.Tensor:
    """Penalize shoving the blade around while approaching it.

    In zero gravity a blade knocked off its rest pose does not come back, and a
    capture that has to chase a moving target is not a capture.

    The free band has to clear the capture's own seating feed. Closing on the
    wedge drives the pin along it until the collar catches, which moves the
    blade about 12.5 mm every single time a capture succeeds. A 5 mm band
    charged that feed at the same rate as a genuine ram, so the penalty could
    not tell the two apart and the gradient pointed at "approach less" rather
    than "approach more gently". 18 mm sits above the feed and well below the
    60 mm at which the episode is failed outright.
    """

    placed = _blade_reset_pose(env)
    if placed is None:
        return torch.zeros(env.num_envs, device=env.device)
    blade = env.scene["spare_blade"]
    displacement = torch.linalg.vector_norm(blade.data.root_pos_w - placed[0], dim=-1)
    return ((displacement - free_m) / 0.010).clamp_min(0.0).square().clamp(max=25.0)


def record_blade_reset_pose(env, env_ids: torch.Tensor | None) -> None:
    """Remember where the blade was placed.

    Two terms need this. Disturbance is measured against it, and so is the
    approach reward: scoring the approach against the *live* blade pays a policy
    for the blade drifting toward the tool, which in zero gravity it can cause
    by shoving it. Scoring against the placed pose means only the arm's own
    motion earns anything, and blade motion is charged separately.
    """

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    blade = env.scene["spare_blade"]
    pose = getattr(env, "_grapple_blade_reset_pose", None)
    if pose is None or pose.shape[0] != env.num_envs:
        pose = blade.data.root_state_w[:, :7].clone()
        env._grapple_blade_reset_pose = pose
    pose[ids] = blade.data.root_state_w[ids, :7]


def _blade_reset_pose(env) -> tuple[torch.Tensor, torch.Tensor] | None:
    pose = getattr(env, "_grapple_blade_reset_pose", None)
    if pose is None or pose.shape[0] != env.num_envs:
        return None
    return pose[:, :3], pose[:, 3:7]


def capture_failure(
    env,
    position_limit: float = 0.120,
    orientation_limit: float = 0.60,
    blade_displacement_limit: float = 0.060,
) -> torch.Tensor:
    """End a grasp episode that can no longer succeed."""

    position, orientation = grapple_grip_error_metrics(env)
    blade = env.scene["spare_blade"]
    placed = _blade_reset_pose(env)
    displaced = (
        torch.linalg.vector_norm(blade.data.root_pos_w - placed[0], dim=-1) > blade_displacement_limit
        if placed is not None
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    conditions = {
        "tool_lost": position > position_limit,
        "attitude_lost": orientation > orientation_limit,
        "blade_shoved": displaced,
    }
    env._grapple_latest_failure_conditions = conditions
    return torch.stack(tuple(conditions.values()), dim=-1).any(dim=-1)


def capture_failure_reward(env) -> torch.Tensor:
    return env.termination_manager.get_term("capture_failed").to(torch.float32) / float(env.step_dt)


def grapple_insertion_conditions(
    env,
    command_name: str = "insertion_goal",
    grip_position_tolerance: float = 0.020,
    grip_orientation_tolerance: float = 0.20,
) -> dict[str, torch.Tensor]:
    """Insertion success, with the grip judged in the head-on convention.

    The shared ``insertion_success_conditions`` measures grasp orientation with
    ``gripper_handle_orientation_error``, which is written against the top-down
    ``GRIPPER_GRASP_ROT``. A head-on capture sits a quarter turn away from that,
    so the check reports about 2.1 rad of grasp error against a 0.15 rad limit
    and can never pass, whatever the policy does.

    That is not hypothetical. The first insert policy drove the blade to 0.1 mm
    of the goal with the grip holding at 8.3 mm and still timed out in 1024 of
    1024 episodes, because this one condition could not be satisfied. The
    geometric thresholds below are the promoted task's, unchanged.
    """

    axial, lateral, orientation = insertion_error_metrics(env, command_name)
    grip_position, grip_orientation = grapple_grip_error_metrics(env)
    velocity = attached_blade_velocity(env)
    return {
        "axial_depth": axial <= INSERTION_AXIAL_DEPTH_TOLERANCE_M,
        "lateral_alignment": lateral <= INSERTION_LATERAL_TOLERANCE_M,
        "orientation": orientation <= INSERTION_ORIENTATION_TOLERANCE_RAD,
        "linear_velocity": (
            torch.linalg.vector_norm(velocity[:, :3], dim=-1) <= INSERTION_LINEAR_VELOCITY_LIMIT_MPS
        ),
        "angular_velocity": (
            torch.linalg.vector_norm(velocity[:, 3:], dim=-1) <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS
        ),
        # Keyed with the shared names on purpose. The curriculum term walks
        # INSERTION_SUCCESS_CONDITION_NAMES to attribute timeouts, so renaming
        # these to "grip_*" crashed training with KeyError: 'grasp_position'.
        "grasp_position": grip_position <= grip_position_tolerance,
        "grasp_orientation": grip_orientation <= grip_orientation_tolerance,
    }


def grapple_insertion_success_mask(env, command_name: str = "insertion_goal", hold_time_s: float = 0.20) -> torch.Tensor:
    """Terminate once a correctly inserted blade has been held, grip intact."""

    conditions = grapple_insertion_conditions(env, command_name)
    geometric = torch.stack(tuple(conditions.values()), dim=-1).all(dim=-1)
    held = _hold_counter(env, "_grapple_insert_hold", geometric, hold_time_s)
    conditions["hold_duration"] = held
    env._insertion_latest_success_conditions = conditions
    return held


def grapple_insertion_success_reward(env, command_name: str = "insertion_goal") -> torch.Tensor:
    return grapple_insertion_success_mask(env, command_name).to(torch.float32) / float(env.step_dt)


def blade_centre_x(env) -> torch.Tensor:
    """Blade centre along the extraction axis, in the environment frame."""

    position, _ = attached_blade_pose_world(env)
    return position[:, 0] - env.scene.env_origins[:, 0]


def extraction_error(env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
    """How much further the blade still has to travel to clear the slot."""

    return (blade_centre_x(env) - target_x).clamp_min(0.0)


def extraction_remaining_observation(env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
    """The same distance as a column, because observation terms are 2-D."""

    return extraction_error(env, target_x).unsqueeze(-1)


def extraction_progress_reward(env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
    """Pay for measured travel toward clear, not for holding position."""

    return _potential_reward(env, "extract", extraction_error(env, target_x), scale=0.030, clamp=1.0)


def extraction_settling_penalty(
    env,
    target_x: float = EXTRACTED_BLADE_CENTRE_X,
    engage_m: float = 0.060,
    linear_velocity_limit: float = EXTRACTION_LINEAR_VELOCITY_LIMIT,
    angular_velocity_limit: float = EXTRACTION_ANGULAR_VELOCITY_LIMIT,
) -> torch.Tensor:
    """Charge residual module motion over the last stretch of the pull.

    :func:`extraction_success_mask` asks for a module that is clear *and*
    settled, and until this term existed nothing paid for the second half:
    velocity entered the objective only through a sparse terminal predicate, so
    there was no gradient toward it anywhere in the pull.

    The measured consequence is extract v10, which trained to a *higher* reward
    than its predecessors -- 158.7 against v8's 148.4 -- and certified at 0.00%,
    with 8,988 of 9,010 episodes ending on grip loss rather than on the clock. A
    progress term weighted 12 paid for travel, nothing charged for arriving at
    speed, and the policy barrelled through the line and out of the workspace.
    That is the policy optimising exactly what it was asked to, again.

    The limits are the ones the success predicate keys on, **read rather than
    restated**, so a policy cannot be trained to a standard the criterion does
    not ask for. Below them this term is exactly zero: the goal is to arrive
    settled, not to crawl.

    The gate ramps over the last ``engage_m`` rather than switching, because the
    module has to *decelerate into* the line and a step change would only charge
    it once it were already too late.
    """

    remaining = extraction_error(env, target_x)
    gate = (1.0 - remaining / engage_m).clamp(0.0, 1.0)
    velocity = attached_blade_velocity(env)
    linear_excess = (
        torch.linalg.vector_norm(velocity[:, :3], dim=-1) / linear_velocity_limit - 1.0
    ).clamp_min(0.0)
    angular_excess = (
        torch.linalg.vector_norm(velocity[:, 3:], dim=-1) / angular_velocity_limit - 1.0
    ).clamp_min(0.0)
    return gate * (linear_excess.square() + angular_excess.square()).clamp(max=9.0)


def extraction_success_mask(
    env,
    target_x: float = EXTRACTED_BLADE_CENTRE_X,
    hold_time_s: float = 0.30,
    linear_velocity_limit: float = EXTRACTION_LINEAR_VELOCITY_LIMIT,
    angular_velocity_limit: float = EXTRACTION_ANGULAR_VELOCITY_LIMIT,
) -> torch.Tensor:
    """Clear of the slot, still gripped, and genuinely settled.

    The velocity limits are **derived from the chain's settling window**, not
    chosen, and that is what makes this criterion consistent rather than
    convenient. See :data:`EXTRACTION_ANGULAR_VELOCITY_LIMIT`.
    """

    velocity = attached_blade_velocity(env)
    active = (
        (blade_centre_x(env) <= target_x)
        & capture_established(env)
        & (torch.linalg.vector_norm(velocity[:, :3], dim=-1) <= linear_velocity_limit)
        & (torch.linalg.vector_norm(velocity[:, 3:], dim=-1) <= angular_velocity_limit)
    )
    return _hold_counter(env, "_grapple_extract_hold", active, hold_time_s)


def extraction_success_reward(env) -> torch.Tensor:
    return extraction_success_mask(env).to(torch.float32) / float(env.step_dt)


def extraction_failure(
    env,
    grip_position_limit: float = 0.030,
    grip_orientation_limit: float = 0.35,
) -> torch.Tensor:
    """End an extraction that has dropped the blade or left the workspace.

    The workspace bound along x is deliberately different from the insertion
    task's, which treats anything below 0.45 as an escape. That is where a
    successful extraction finishes.
    """

    position, orientation = grapple_grip_error_metrics(env)
    blade_position, _ = attached_blade_pose_world(env)
    local = blade_position - env.scene.env_origins
    conditions = {
        "grip_lost": position > grip_position_limit,
        "grip_attitude_lost": orientation > grip_orientation_limit,
        "workspace_x": (local[:, 0] < -0.10) | (local[:, 0] > 1.10),
        "workspace_yz": (local[:, 1].abs() > 0.30) | (local[:, 2] < 0.40) | (local[:, 2] > 1.10),
    }
    env._grapple_latest_failure_conditions = conditions
    return torch.stack(tuple(conditions.values()), dim=-1).any(dim=-1)


def extraction_failure_reward(env) -> torch.Tensor:
    return env.termination_manager.get_term("extraction_failed").to(torch.float32) / float(env.step_dt)


def grip_retention_penalty(
    env,
    free_m: float = 0.004,
    free_rad: float = 0.08,
    orientation_scale: float = 0.15,
    orientation_weight: float = 0.25,
    max_penalty: float = 25.0,
) -> torch.Tensor:
    """Penalize the grip drifting, without paying a policy to stand still.

    The two orientation parameters exist because the defaults are too soft for
    extraction and must not change for insertion, whose certification was
    produced under them. Measured on extract v6: 7,971 of 8,093 failures end with
    the grip attitude at the 0.350 rad failure limit while grip *position* holds
    at 12.5 mm, and under the defaults that attitude costs about 0.16 per step at
    the 0.20 rad success limit against a progress reward weighted 12. The policy
    was trading attitude for travel because attitude was very nearly free until
    the episode died on it, which is the policy optimising exactly what it was
    asked to.
    """

    position, orientation = grapple_grip_error_metrics(env)
    position_excess = ((position - free_m) / 0.010).clamp_min(0.0)
    orientation_excess = ((orientation - free_rad) / orientation_scale).clamp_min(0.0)
    return (position_excess.square() + orientation_weight * orientation_excess.square()).clamp(max=max_penalty)


def reset_grapple_blade_pose(
    env,
    env_ids: torch.Tensor | None,
    pose: tuple[float, ...],
    position_noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("spare_blade"),
) -> None:
    """Place the blade at one fixed pose, with optional isotropic position noise."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    blade = env.scene[asset_cfg.name]
    target = blade.data.root_state_w.new_tensor(pose).expand(len(ids), -1).clone()
    if position_noise > 0.0:
        target[:, :3] += (2.0 * torch.rand((len(ids), 3), device=env.device) - 1.0) * position_noise
    target[:, :3] += env.scene.env_origins[ids]
    blade.write_root_pose_to_sim(target, env_ids=ids)
    blade.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


def hold_two_stage_grip(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    capture_positions: tuple[float, ...],
    hold_positions: tuple[float, ...],
) -> None:
    """Command the capture closure until the grip loads, then the holding one.

    Extract and insert start already captured, but "already captured" cannot be
    faked by writing the fingers to their seated angle: that places the pads
    around the wedge without pressing the pin against the collar, so the first
    pull travels the whole seating gap before anything takes load. Measured that
    way, a 40 mm pull moved the blade 0.1 mm and opened a 12.9 mm grip error.

    Letting the capture actually happen, inside the action term's settling
    window, produces the same preloaded state the pull gate measured 69 N on.

    **The holding closure latches, and that matters more than it looks.** This
    term used to command the capture closure again whenever
    :func:`capture_established` went false, which is correct while capturing and
    destructive afterwards: the predicate fails as soon as grip error passes
    20 mm, and rail contact during an insertion pushes it past that routinely.
    The fingers then open from the hold command to the gentler capture one, about
    21 mm of travel, and drop the module mid-task. It is the most likely single
    cause of the insert skill losing its grip in 6,023 of 9,014 episodes, and of
    the module being released mid-transit when the three skills were first
    chained. A real servicer does not relax its grip because the part shifted.
    """

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    capture = robot.data.joint_pos.new_tensor(capture_positions).expand(len(ids), -1)
    hold = robot.data.joint_pos.new_tensor(hold_positions).expand(len(ids), -1)
    latched = getattr(env, "_grapple_hold_latched", None)
    if latched is None or latched.shape[0] != env.num_envs:
        latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._grapple_hold_latched = latched
    latched[ids] |= capture_established(env)[ids]
    robot.set_joint_position_target(
        torch.where(latched[ids].unsqueeze(-1), hold, capture), joint_ids=asset_cfg.joint_ids, env_ids=ids
    )


def reset_grapple_fingers(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    finger_joint: float,
) -> None:
    """Write the finger joints to one command and hold them there."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    signs = robot.data.joint_pos.new_tensor(ROBOTIQ_2F85_COUPLING_SIGNS)
    targets = (signs * finger_joint).expand(len(ids), -1).clone()
    robot.write_joint_state_to_sim(targets, torch.zeros_like(targets), joint_ids=asset_cfg.joint_ids, env_ids=ids)
    robot.set_joint_position_target(targets, joint_ids=asset_cfg.joint_ids, env_ids=ids)


__all__ = [
    "EXTRACTED_BLADE_CENTRE_X",
    "GRIP_TORQUE_THRESHOLD_NM",
    "SLOT_MOUTH_X",
    "TwoStageRobotiqAction",
    "TwoStageRobotiqActionCfg",
    "arm_grapple_latch",
    "blade_centre_x",
    "blade_disturbance_penalty",
    "capture_approach_reward",
    "capture_established",
    "capture_failure",
    "capture_failure_reward",
    "capture_success_mask",
    "capture_success_reward",
    "extraction_error",
    "extraction_failure",
    "extraction_failure_reward",
    "extraction_progress_reward",
    "extraction_remaining_observation",
    "extraction_settling_penalty",
    "extraction_success_mask",
    "extraction_success_reward",
    "GrappleLatch",
    "grapple_grip_attitude_axes",
    "grapple_grip_attitude_error_world",
    "grapple_grip_error_metrics",
    "grapple_latch_diagnostics",
    "grapple_latched",
    "grapple_grip_error_observation",
    "grapple_grip_pose_error",
    "grapple_insertion_conditions",
    "grapple_insertion_success_mask",
    "grapple_insertion_success_reward",
    "grip_drive_torque",
    "grip_finger_angle",
    "grip_retention_penalty",
    "gripper_state_observation",
    "hold_two_stage_grip",
    "record_blade_reset_pose",
    "reset_grapple_blade_pose",
    "reset_grapple_fingers",
    "reset_grapple_progress",
]
