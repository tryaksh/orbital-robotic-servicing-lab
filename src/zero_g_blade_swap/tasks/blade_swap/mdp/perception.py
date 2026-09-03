"""Camera-derived module state for the deployed servicing workflow.

The three workflow policies were trained with four views of the module state:
the tool-to-grip error, extraction travel remaining, insertion-goal error, and
module velocity. Replacing only the first of those is not a vision policy; the
other three values still reveal the simulator's answer. This module replaces
all four while preserving their original widths and ordering.

``ModuleStateEstimator`` is shared by every perception observation term on one
vectorized environment. It caches by ``common_step_counter``, so all three
observation groups consume one pose-head evaluation per control step. Position
and attitude come from the camera. Linear and angular velocity are finite
differences of consecutive camera estimates passed through a first-order
temporal filter; simulator body velocity is never consulted in deployment mode.

There are three deliberately explicit modes:

* ``deployment`` requires a checkpoint and only reads the camera;
* ``oracle`` reads simulator pose for an evaluation upper bound;
* ``blind`` reads a configured nominal pose for an evaluation lower bound and
  neither renders an image nor reads the live module state.

The exact module pose is otherwise read only by collection labels and diagnostic
error metrics. In particular, diagnostic error is not computed as a side effect
of producing a policy observation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from zero_g_blade_swap.fiducial import estimate_fiducial_pose
from zero_g_blade_swap.grapple_geometry import (
    EXTRACTED_BLADE_CENTRE_X,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_PIN_GRIP_OFFSET,
)
from zero_g_blade_swap.pose_head import MODULE_POSE_DIM, ModulePoseHead, load_pose_head

from ..assets import SLOT_CENTRE_Y, SLOT_UPPER_LIP_HALF_WIDTH_Y
from .observations import camera_rgb_with_radiation_noise, end_effector_pose_world

PERCEPTION_DEPLOYMENT = "deployment"
PERCEPTION_ORACLE = "oracle"
PERCEPTION_BLIND = "blind"
PERCEPTION_MODES = frozenset((PERCEPTION_DEPLOYMENT, PERCEPTION_ORACLE, PERCEPTION_BLIND))
#: Where the module-velocity channel comes from.
#:
#: **The camera is never the best available estimate of this quantity, at any
#: point in the task, and that is measured rather than argued.** Before capture
#: the module is held by its rails and is not moving, so a finite difference of
#: consecutive camera poses reports the estimator's own residual -- 17 mm/s at
#: the deployed filter against a seated module's 0.69 -- as motion. After capture
#: the module is on the form lock and moves with the wrist, whose velocity the
#: robot knows from its own encoders to a precision no camera approaches.
#:
#: The measurement that makes this worth an option: on an unchanged checkpoint,
#: noising the pose channels and leaving the velocity exact costs 8.33 points,
#: noising the velocity and leaving the pose exact costs 10.21, and noising both
#: costs 41.15. The interaction is larger than the sum of the parts, so restoring
#: *either* channel recovers most of the loss, and this is the one that can be
#: restored without a camera.
#:
#: ``camera`` is the shipped path and stays the default; every published RGB-D
#: number was measured on it.
MODULE_VELOCITY_FROM_CAMERA = "camera"
MODULE_VELOCITY_FROM_KINEMATICS = "kinematics"
MODULE_VELOCITY_SOURCES = frozenset((MODULE_VELOCITY_FROM_CAMERA, MODULE_VELOCITY_FROM_KINEMATICS))

PERCEPTION_BACKEND_POSE_HEAD = "pose_head"
PERCEPTION_BACKEND_FIDUCIAL_PNP = "fiducial_pnp"
PERCEPTION_BACKENDS = frozenset((PERCEPTION_BACKEND_POSE_HEAD, PERCEPTION_BACKEND_FIDUCIAL_PNP))

# These functions all expose the live module pose or velocity. They are valid
# for state-only policies, rewards, terminations, and metrics, but never as
# active terms in a deployed vision policy group.
FORBIDDEN_DEPLOYMENT_OBSERVATION_FUNCTIONS = frozenset(
    (
        "attached_blade_velocity",
        "attached_blade_pose_world",
        "extraction_remaining_observation",
        "grapple_grip_error_observation",
        "grapple_grip_error_metrics",
        "insertion_goal_error",
        "module_pose_label",
        "slot_occupancy_label",
    )
)


def module_pose_label(env, asset_name: str = "spare_blade") -> torch.Tensor:
    """Return simulator module pose for supervised labels and diagnostics only.

    The position is environment-local, so the label cannot encode which tile of
    the cloned grid an environment occupies. This function must not be wired to
    a deployed policy observation group; ``audit_vision_deployment_observations``
    enforces that boundary at configuration time.
    """

    blade = env.scene[asset_name]
    local_position = blade.data.root_pos_w - env.scene.env_origins
    return torch.cat((local_position, axis_angle_from_quat(blade.data.root_quat_w)), dim=-1)


def slot_occupancy_label(env, asset_name: str = "spare_blade") -> torch.Tensor:
    """Return exact per-bay occupancy labels for offline supervision.

    These are two independent indicators because a module in transit occupies
    neither bay. Like ``module_pose_label``, this is privileged label data and
    must not be connected to a deployed policy observation.
    """

    blade = env.scene[asset_name]
    local = blade.data.root_pos_w - env.scene.env_origins
    centres = local.new_tensor(SLOT_CENTRE_Y)
    inside_mouth = local[:, 0] > EXTRACTED_BLADE_CENTRE_X
    within_channel = (local[:, 1].unsqueeze(-1) - centres).abs() <= SLOT_UPPER_LIP_HALF_WIDTH_Y
    return (within_channel & inside_mouth.unsqueeze(-1)).to(torch.float32)


def occupancy_from_module_pose(module_pose: torch.Tensor) -> torch.Tensor:
    """Infer rack occupancy from an estimated pose, without simulator state."""

    centres = module_pose.new_tensor(SLOT_CENTRE_Y)
    inside_mouth = module_pose[:, 0] > EXTRACTED_BLADE_CENTRE_X
    within_channel = (module_pose[:, 1].unsqueeze(-1) - centres).abs() <= SLOT_UPPER_LIP_HALF_WIDTH_Y
    return (within_channel & inside_mouth.unsqueeze(-1)).to(torch.float32)


def _orientation_from_module_pose(module_pose: torch.Tensor) -> torch.Tensor:
    """Convert the pose head's rotation vector to a numerically safe quaternion."""

    rotation_vector = module_pose[:, 3:]
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
    fallback_axis = rotation_vector.new_zeros(rotation_vector.shape)
    fallback_axis[:, 0] = 1.0
    axis = torch.where(
        (angle > 1.0e-8).unsqueeze(-1),
        rotation_vector / angle.clamp_min(1.0e-8).unsqueeze(-1),
        fallback_axis,
    )
    return quat_from_angle_axis(angle, axis)


def grip_error_from_module_pose(env, module_pose: torch.Tensor) -> torch.Tensor:
    """Convert an estimated local module pose to the policy's six-value grip error.

    The tool pose is encoder/forward-kinematics information available on a real
    robot. No live module state is read here.
    """

    tool_position, tool_orientation = end_effector_pose_world(env)
    blade_position = module_pose[:, :3] + env.scene.env_origins
    blade_orientation = _orientation_from_module_pose(module_pose)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    desired = quat_mul(
        blade_orientation,
        blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation),
    )
    relative_position, relative_quat = subtract_frame_transforms(
        tool_position,
        tool_orientation,
        grip_position,
        desired,
    )
    return torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)


def extraction_remaining_from_module_pose(
    module_pose: torch.Tensor,
    target_x: float = EXTRACTED_BLADE_CENTRE_X,
) -> torch.Tensor:
    """Return camera-estimated extraction travel as the original one-column term."""

    return (module_pose[:, :1] - target_x).clamp_min(0.0)


def insertion_goal_error_from_module_pose(
    env,
    module_pose: torch.Tensor,
    command_name: str = "insertion_goal",
) -> torch.Tensor:
    """Return goal-minus-estimated-module pose with the original six-value layout."""

    goal = env.command_manager.get_command(command_name)
    relative_position, relative_quat = subtract_frame_transforms(
        module_pose[:, :3],
        _orientation_from_module_pose(module_pose),
        goal[:, :3],
        goal[:, 3:7],
    )
    return torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)


def _resolve_perception_mode(cfg) -> str:
    """Resolve the explicit mode while retaining the existing evaluation CLI flags."""

    configured = str(getattr(cfg, "perception_mode", PERCEPTION_DEPLOYMENT)).lower()
    if configured not in PERCEPTION_MODES:
        choices = ", ".join(sorted(PERCEPTION_MODES))
        raise ValueError(f"perception_mode must be one of {choices}; got {configured!r}")

    # Existing evaluation scripts spell the two controls this way. Preserve the
    # flags, but reject interpolation: a partly privileged policy is neither a
    # deployment result nor an interpretable oracle comparison.
    blind = bool(getattr(cfg, "pose_head_blind", False))
    oracle_blend = float(getattr(cfg, "pose_head_oracle_blend", 0.0))
    if oracle_blend not in (0.0, 1.0):
        raise ValueError(
            "pose_head_oracle_blend no longer permits mixed estimates. Use perception_mode='deployment' "
            "or perception_mode='oracle' so privileged state cannot leak partially into a policy."
        )
    if blind and oracle_blend == 1.0:
        raise ValueError("blind and oracle perception controls are mutually exclusive")

    legacy_mode = PERCEPTION_BLIND if blind else PERCEPTION_ORACLE if oracle_blend == 1.0 else None
    if configured != PERCEPTION_DEPLOYMENT and legacy_mode is not None and legacy_mode != configured:
        raise ValueError(f"conflicting perception modes: perception_mode={configured!r}, legacy flag={legacy_mode!r}")
    return legacy_mode if configured == PERCEPTION_DEPLOYMENT and legacy_mode is not None else configured


class ModuleStateEstimator:
    """One cached camera estimator shared by every module observation term.

    The object belongs to the vectorized environment (stored under
    ``_module_state_estimator``), not to an individual observation term. A call
    from grasp, extract, insert, or diagnostics in the same control step returns
    the same cached tensors. ``cnn_evaluation_count`` is exposed for runtime
    audits and throughput tests.
    """

    def __init__(self, env) -> None:
        self._env = env
        self._mode = _resolve_perception_mode(env.cfg)
        self._backend = str(getattr(env.cfg, "perception_backend", PERCEPTION_BACKEND_POSE_HEAD)).lower()
        if self._backend not in PERCEPTION_BACKENDS:
            choices = ", ".join(sorted(PERCEPTION_BACKENDS))
            raise ValueError(f"perception_backend must be one of {choices}; got {self._backend!r}")
        checkpoint = getattr(env.cfg, "pose_head_checkpoint", None)
        if self._mode == PERCEPTION_DEPLOYMENT and self._backend == PERCEPTION_BACKEND_POSE_HEAD and checkpoint is None:
            raise ValueError(
                "deployment perception requires env.cfg.pose_head_checkpoint; refusing to substitute "
                "simulator module state. Use perception_mode='oracle' or 'blind' only for an explicit ablation."
            )
        self._head = (
            load_pose_head(checkpoint, env.device)
            if self._mode == PERCEPTION_DEPLOYMENT and self._backend == PERCEPTION_BACKEND_POSE_HEAD
            else None
        )
        self._sensor_cfg = SceneEntityCfg("camera")
        self._velocity_source = str(getattr(env.cfg, "module_velocity_source", MODULE_VELOCITY_FROM_CAMERA)).lower()
        if self._velocity_source not in MODULE_VELOCITY_SOURCES:
            choices = ", ".join(sorted(MODULE_VELOCITY_SOURCES))
            raise ValueError(f"module_velocity_source must be one of {choices}; got {self._velocity_source!r}")
        self._filter_time_constant_s = float(getattr(env.cfg, "perception_velocity_filter_time_constant_s", 0.10))
        self._fiducial_sensor_names = [self._sensor_cfg.name]
        if "camera_insert" in env.scene.sensors:
            self._fiducial_sensor_names.append("camera_insert")
        if not math.isfinite(self._filter_time_constant_s) or self._filter_time_constant_s < 0.0:
            raise ValueError("perception_velocity_filter_time_constant_s must be finite and non-negative")

        self._cached_step: int | None = None
        self._last_update_step: int | None = None
        self._pose = torch.zeros((env.num_envs, MODULE_POSE_DIM), device=env.device)
        self._velocity = torch.zeros((env.num_envs, 6), device=env.device)
        self._occupancy_probabilities: torch.Tensor | None = None
        self._previous_position = torch.zeros((env.num_envs, 3), device=env.device)
        self._previous_orientation = torch.zeros((env.num_envs, 4), device=env.device)
        self._previous_orientation[:, 0] = 1.0
        self._history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._cnn_evaluation_count = 0
        self._fiducial_evaluation_count = 0
        self._fiducial_detection_count = 0
        self._fiducial_failure_count = 0
        self._fiducial_consecutive_failures = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._fiducial_max_consecutive_failures = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._confidence = torch.zeros(env.num_envs, device=env.device)
        self._reprojection_error_px = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._fiducial_detection_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._fiducial_current_detection = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._fiducial_detections_by_marker: dict[int, int] = {}
        self._payload_stage_engaged = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Encoder propagation is valid only after the robot has physically
        # captured the module. Before that event the tool moves and the module
        # does not; treating their transform as rigid turns one missed camera
        # frame during approach into a moving fictitious target.
        self._module_tool_attached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._module_in_tool_position = torch.zeros((env.num_envs, 3), device=env.device)
        self._module_in_tool_orientation = torch.zeros((env.num_envs, 4), device=env.device)
        self._module_in_tool_orientation[:, 0] = 1.0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def cnn_evaluation_count(self) -> int:
        return self._cnn_evaluation_count

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def fiducial_detection_statistics(self) -> dict[str, float | int]:
        """Return measured detector availability without exposing sim truth."""

        attempts = self._fiducial_detection_count + self._fiducial_failure_count
        return {
            "attempts": attempts,
            "detections": self._fiducial_detection_count,
            "failures": self._fiducial_failure_count,
            "detection_rate": self._fiducial_detection_count / attempts if attempts else 0.0,
            "max_consecutive_failures": int(self._fiducial_max_consecutive_failures.max().item()),
            # Which flush plate carried each detection. The module now has two,
            # and a run that only ever reads one of them has not exercised the
            # reason there are two.
            "detections_by_datum": dict(sorted(self._fiducial_detections_by_marker.items())),
        }

    @property
    def fiducial_current_detection(self) -> torch.Tensor:
        """Return which environments have a valid datum in the current frame."""

        self.estimate()
        return self._fiducial_current_detection

    def mark_payload_stage_engaged(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Switch missed-frame propagation from robot grasp to safe hold-last.

        Before handoff, robot encoders and forward kinematics propagate the
        last camera pose through brief line-of-sight occlusions. Once the
        physical shuttle owns the payload, that rigid tool relationship is no
        longer valid and missed frames hold the last estimate instead.
        """

        self._payload_stage_engaged[env_ids] = True
        self._module_tool_attached[env_ids] = False

    def mark_robot_capture_established(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Permit tool-kinematic propagation after a verified physical capture.

        The relative pose is frozen from the current camera estimate and robot
        forward kinematics. No simulator module state enters this handoff.
        """

        self.estimate()
        tool_position, tool_orientation = end_effector_pose_world(self._env)
        module_position_world = self._pose[:, :3] + self._env.scene.env_origins
        module_orientation_world = _orientation_from_module_pose(self._pose)
        relative_position, relative_orientation = subtract_frame_transforms(
            tool_position,
            tool_orientation,
            module_position_world,
            module_orientation_world,
        )
        self._module_in_tool_position[env_ids] = relative_position[env_ids]
        self._module_in_tool_orientation[env_ids] = relative_orientation[env_ids]
        self._module_tool_attached[env_ids] = True

    @property
    def confidence(self) -> torch.Tensor:
        """Return the latest bounded detector quality score."""

        self.estimate()
        return self._confidence

    @property
    def reprojection_error_px(self) -> torch.Tensor:
        """Return the latest geometric reprojection RMS in pixels."""

        self.estimate()
        return self._reprojection_error_px

    def pose_wxyz(self) -> torch.Tensor:
        """Return the cached local pose as position plus a unit quaternion.

        This is a reporting interface, not a second estimator call: ``estimate``
        is step-cached, so policy observations and the exported terminal evidence
        refer to the same camera result.
        """

        pose, _ = self.estimate()
        return torch.cat((pose[:, :3], _orientation_from_module_pose(pose)), dim=-1)

    def occupancy_probabilities(self) -> torch.Tensor | None:
        """Return the cached per-bay occupancy scores when the head supports them.

        These sigmoid outputs are useful planning scores, not calibrated
        confidence.  Calling this method shares the same camera pass as the pose
        observations; it never invokes a second network evaluation.
        """

        self.estimate()
        return self._occupancy_probabilities

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Invalidate reset environments so velocity never crosses episode boundaries."""

        ids = slice(None) if env_ids is None else env_ids
        self._history_valid[ids] = False
        self._fiducial_detection_valid[ids] = False
        self._fiducial_current_detection[ids] = False
        self._payload_stage_engaged[ids] = False
        self._module_tool_attached[ids] = False
        self._velocity[ids] = 0.0
        # A partial reset can happen after this step's observations were cached.
        # Invalidate the whole batch; inference remains one pass on the next read.
        self._cached_step = None

    def estimate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(pose, filtered_velocity)`` for the current control step."""

        step = int(self._env.common_step_counter)
        if self._cached_step == step:
            return self._pose, self._velocity

        if self._mode == PERCEPTION_DEPLOYMENT:
            pose = self._estimate_deployment()
        elif self._mode == PERCEPTION_ORACLE:
            pose = self._estimate_oracle()
        else:
            pose = self._estimate_blind()

        if pose.shape != (self._env.num_envs, MODULE_POSE_DIM):
            raise RuntimeError(
                f"module pose estimator returned {tuple(pose.shape)}, expected "
                f"({self._env.num_envs}, {MODULE_POSE_DIM})"
            )
        if not bool(torch.isfinite(pose).all()):
            raise RuntimeError("module pose estimator produced a non-finite policy observation")

        pose = pose.to(device=self._pose.device, dtype=self._pose.dtype)
        orientation = _orientation_from_module_pose(pose)
        elapsed_steps = 1 if self._last_update_step is None else max(1, step - self._last_update_step)
        dt = float(self._env.step_dt) * elapsed_steps

        linear = (pose[:, :3] - self._previous_position) / dt
        rotation_delta = quat_mul(orientation, quat_inv(self._previous_orientation))
        angular = axis_angle_from_quat(rotation_delta) / dt
        raw_velocity = torch.cat((linear, angular), dim=-1)
        raw_velocity = torch.where(self._history_valid.unsqueeze(-1), raw_velocity, torch.zeros_like(raw_velocity))

        tau = self._filter_time_constant_s
        alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
        self._velocity.mul_(1.0 - alpha).add_(raw_velocity, alpha=alpha)
        self._velocity.masked_fill_(~self._history_valid.unsqueeze(-1), 0.0)
        self._pose.copy_(pose)
        self._previous_position.copy_(pose[:, :3])
        self._previous_orientation.copy_(orientation)
        self._history_valid.fill_(True)
        self._last_update_step = step
        self._cached_step = step
        if self._velocity_source == MODULE_VELOCITY_FROM_KINEMATICS:
            self._velocity.copy_(self._kinematic_velocity())
        return self._pose, self._velocity

    def _kinematic_velocity(self) -> torch.Tensor:
        """Report the module's velocity from the robot instead of from the camera.

        Two regimes, and neither reads the module's own state:

        * **before capture** the module is held by its rails and is not moving,
          so its velocity is zero. That is an assumption about the scene rather
          than a measurement of it, and it is the same assumption the whole
          workflow already makes when it plans a capture against a bay it
          believes is holding a module still;
        * **after capture** the module is on the form lock and moves with the
          wrist, so the wrist's velocity is the module's. The transit already
          rests on exactly that assumption and reports what it costs -- 1.05 mm
          and 3.27 mrad of maximum tool-to-module drift in the continuous
          episode.

        The wrist's velocity is encoder and forward-kinematics information, which
        a real servicer has. ``audit_vision_deployment_observations`` forbids a
        deployed group from reading the *module's* live state and this does not:
        it reads the robot's.
        """

        robot = self._env.scene["robot"]
        body_id = self._wrist_body_id(robot)
        velocity = torch.cat(
            (robot.data.body_lin_vel_w[:, body_id], robot.data.body_ang_vel_w[:, body_id]), dim=-1
        )
        return torch.where(self._module_tool_attached.unsqueeze(-1), velocity, torch.zeros_like(velocity))

    def _wrist_body_id(self, robot) -> int:
        """Resolve and cache the wrist body index the tool frame hangs off."""

        cached = getattr(self, "_cached_wrist_body_id", None)
        if cached is not None:
            return cached
        names = list(robot.body_names)
        if "wrist_3_link" not in names:
            raise RuntimeError(
                "module_velocity_source='kinematics' needs the wrist body the tool frame is defined "
                f"from; the robot exposes {names}"
            )
        self._cached_wrist_body_id = names.index("wrist_3_link")
        return self._cached_wrist_body_id

    def _estimate_deployment(self) -> torch.Tensor:
        """Infer pose from one camera frame, without touching live module state."""

        if self._backend == PERCEPTION_BACKEND_FIDUCIAL_PNP:
            # A deterministic geometric detector consumes the calibrated sensor
            # stream directly.  The Gaussian "radiation" transform is training
            # augmentation for the learned head, not a property of the camera
            # calibration and would needlessly corrupt sub-pixel corners.
            image = self._env.scene.sensors[self._sensor_cfg.name].data.output["rgb"][..., :3]
            if image.dtype == torch.uint8:
                image = image.to(dtype=torch.float32).mul_(1.0 / 255.0)
            else:
                image = image.to(dtype=torch.float32).clamp_(0.0, 1.0)
            return self._estimate_fiducial_pnp(image)
        image = camera_rgb_with_radiation_noise(self._env, sensor_cfg=self._sensor_cfg)
        if self._head is None:  # constructor invariant, kept explicit at the privileged boundary
            raise RuntimeError("deployment perception has no loaded pose head")
        with torch.inference_mode():
            if self._head.occupancy is None:
                pose = self._head(image)
                self._occupancy_probabilities = None
            else:
                pose, logits = self._head.forward_with_occupancy(image)
                if logits.ndim != 2 or logits.shape[0] != self._env.num_envs:
                    raise RuntimeError(
                        f"occupancy head returned shape {tuple(logits.shape)}, expected ({self._env.num_envs}, slots)"
                    )
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("occupancy head produced non-finite planning scores")
                self._occupancy_probabilities = torch.sigmoid(logits)
        self._cnn_evaluation_count += 1
        self._confidence.fill_(float("nan"))
        self._reprojection_error_px.fill_(float("nan"))
        return pose

    def _estimate_fiducial_pnp(self, image: torch.Tensor) -> torch.Tensor:
        """Recover module poses from RGB fiducials and calibrated cameras."""

        camera_candidates = []
        for sensor_index, sensor_name in enumerate(self._fiducial_sensor_names):
            camera = self._env.scene.sensors[sensor_name]
            sensor_image = image if sensor_index == 0 else camera.data.output["rgb"][..., :3]
            if sensor_image.dtype == torch.uint8:
                sensor_image = sensor_image.to(dtype=torch.float32).mul_(1.0 / 255.0)
            else:
                sensor_image = sensor_image.to(dtype=torch.float32).clamp_(0.0, 1.0)
            camera_candidates.append(
                (
                    sensor_image.detach().cpu().numpy(),
                    camera.data.output["distance_to_image_plane"].detach().cpu().numpy(),
                    camera.data.intrinsic_matrices.detach().cpu().numpy(),
                    camera.data.pos_w,
                    matrix_from_quat(camera.data.quat_w_ros),
                )
            )
        pose = self._pose.new_empty((self._env.num_envs, MODULE_POSE_DIM))
        detected_now = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)

        for env_index in range(self._env.num_envs):
            selected = None
            for images, depth_images, intrinsics, camera_position, camera_rotation in camera_candidates:
                try:
                    estimate = estimate_fiducial_pose(
                        images[env_index], intrinsics[env_index], depth_images[env_index]
                    )
                except (RuntimeError, ValueError):
                    continue
                selected = (estimate, camera_position[env_index], camera_rotation[env_index])
                break
            if selected is None:
                self._fiducial_failure_count += 1
                self._fiducial_consecutive_failures[env_index] += 1
                self._fiducial_max_consecutive_failures[env_index] = torch.maximum(
                    self._fiducial_max_consecutive_failures[env_index],
                    self._fiducial_consecutive_failures[env_index],
                )
                # Tiled cameras have no valid frame during the environment's
                # reset observation. Return a drawing-level prior solely to
                # keep that zero-action warm-up step finite. Confidence and
                # occupancy remain zero, so the driver's visual preflight
                # cannot accept or move until a real RGB-D detection arrives.
                if bool(self._fiducial_detection_valid[env_index]):
                    pose[env_index] = self._pose[env_index]
                else:
                    initial = self._env.cfg.scene.spare_blade.init_state
                    pose[env_index, :3] = pose.new_tensor(initial.pos)
                    initial_quat = pose.new_tensor(initial.rot).unsqueeze(0)
                    pose[env_index, 3:] = axis_angle_from_quat(initial_quat)[0]
                self._confidence[env_index] = 0.0
                self._reprojection_error_px[env_index] = float("inf")
                continue
            estimate, selected_camera_position, selected_camera_rotation = selected
            rotation_camera_from_object = self._pose.new_tensor(estimate.rotation_camera_from_object)
            position_camera = self._pose.new_tensor(estimate.position_camera_m)
            rotation_world_from_object = selected_camera_rotation @ rotation_camera_from_object
            position_world = selected_camera_position + selected_camera_rotation @ position_camera
            position_local = position_world - self._env.scene.env_origins[env_index]
            orientation_world = quat_from_matrix(rotation_world_from_object.unsqueeze(0))[0]
            pose[env_index, :3] = position_local
            pose[env_index, 3:] = axis_angle_from_quat(orientation_world.unsqueeze(0))[0]
            self._confidence[env_index] = estimate.confidence
            self._reprojection_error_px[env_index] = estimate.reprojection_error_px
            self._fiducial_detection_valid[env_index] = True
            self._fiducial_detection_count += 1
            self._fiducial_detections_by_marker[int(estimate.marker_id)] = (
                self._fiducial_detections_by_marker.get(int(estimate.marker_id), 0) + 1
            )
            self._fiducial_consecutive_failures[env_index] = 0
            detected_now[env_index] = True

        self._occupancy_probabilities = occupancy_from_module_pose(pose)
        self._occupancy_probabilities[~detected_now] = 0.0
        tool_position, tool_orientation = end_effector_pose_world(self._env)
        module_position_world = pose[:, :3] + self._env.scene.env_origins
        module_orientation_world = _orientation_from_module_pose(pose)
        relative_position, relative_orientation = subtract_frame_transforms(
            tool_position,
            tool_orientation,
            module_position_world,
            module_orientation_world,
        )
        self._module_in_tool_position[detected_now] = relative_position[detected_now]
        self._module_in_tool_orientation[detected_now] = relative_orientation[detected_now]
        propagate = (
            ~detected_now
            & self._fiducial_detection_valid
            & self._module_tool_attached
            & ~self._payload_stage_engaged
        )
        if bool(propagate.any()):
            propagated_position, propagated_orientation = combine_frame_transforms(
                tool_position,
                tool_orientation,
                self._module_in_tool_position,
                self._module_in_tool_orientation,
            )
            pose[propagate, :3] = propagated_position[propagate] - self._env.scene.env_origins[propagate]
            pose[propagate, 3:] = axis_angle_from_quat(propagated_orientation[propagate])
        self._fiducial_current_detection.copy_(detected_now)
        self._fiducial_evaluation_count += 1
        return pose

    def _estimate_oracle(self) -> torch.Tensor:
        """Read exact pose for an explicitly requested evaluation upper bound."""

        self._occupancy_probabilities = slot_occupancy_label(self._env)
        return module_pose_label(self._env)

    def _estimate_blind(self) -> torch.Tensor:
        """Return a configured nominal prior without reading image or live module state."""

        configured_occupancy = getattr(self._env.cfg, "perception_blind_occupancy", None)
        if configured_occupancy is None:
            self._occupancy_probabilities = None
        else:
            occupancy = self._pose.new_tensor(configured_occupancy).flatten()
            if occupancy.numel() != len(SLOT_CENTRE_Y):
                raise ValueError(f"perception_blind_occupancy must contain {len(SLOT_CENTRE_Y)} bay scores")
            if not bool(torch.isfinite(occupancy).all()) or not bool(((occupancy >= 0.0) & (occupancy <= 1.0)).all()):
                raise ValueError("perception_blind_occupancy scores must be finite and in [0, 1]")
            self._occupancy_probabilities = occupancy.expand(self._env.num_envs, -1)

        configured = getattr(self._env.cfg, "perception_blind_module_pose", None)
        if configured is not None:
            pose = self._pose.new_tensor(configured).flatten()
            if pose.numel() == MODULE_POSE_DIM:
                return pose.expand(self._env.num_envs, -1)
            if pose.numel() == 7:
                rotation = axis_angle_from_quat(pose[3:7].expand(self._env.num_envs, -1))
                return torch.cat((pose[:3].expand(self._env.num_envs, -1), rotation), dim=-1)
            raise ValueError("perception_blind_module_pose must contain position+rotation-vector (6) or pose (7)")

        # Asset configuration is a design prior, not a simulator readback. It is
        # intentionally fixed even if a reset event moves the live module.
        initial = self._env.cfg.scene.spare_blade.init_state
        position = self._pose.new_tensor(initial.pos).expand(self._env.num_envs, -1)
        orientation = self._pose.new_tensor(initial.rot).expand(self._env.num_envs, -1)
        return torch.cat((position, axis_angle_from_quat(orientation)), dim=-1)

    def diagnostic_position_error_m(self) -> torch.Tensor:
        """Compare the cached estimate with truth without affecting policy values."""

        estimated, _ = self.estimate()
        truth = module_pose_label(self._env)
        return torch.linalg.vector_norm(estimated[:, :3] - truth[:, :3], dim=-1)


def shared_module_state_estimator(env) -> ModuleStateEstimator:
    """Return the sole module estimator attached to this vectorized environment."""

    estimator = getattr(env, "_module_state_estimator", None)
    if estimator is None:
        estimator = ModuleStateEstimator(env)
        env._module_state_estimator = estimator
    elif not isinstance(estimator, ModuleStateEstimator):
        raise TypeError("env._module_state_estimator is not a ModuleStateEstimator")
    return estimator


class _PerceivedModuleObservation(ManagerTermBase):
    """Base manager term that binds all policy views to the shared estimator."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._estimator = shared_module_state_estimator(env)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        self._estimator.reset(env_ids)


class PerceivedGraspError(_PerceivedModuleObservation):
    """Six-value tool-to-grip error derived from the shared camera pose."""

    def __call__(self, env) -> torch.Tensor:
        pose, _ = self._estimator.estimate()
        return grip_error_from_module_pose(env, pose)

    @property
    def position_error_m(self) -> torch.Tensor:
        """Diagnostic position error; this property is privileged by design."""

        return self._estimator.diagnostic_position_error_m()


class PerceivedExtractionRemaining(_PerceivedModuleObservation):
    """One-value extraction distance derived from the shared camera pose."""

    def __call__(self, env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
        del env
        pose, _ = self._estimator.estimate()
        return extraction_remaining_from_module_pose(pose, target_x)


class PerceivedInsertionGoalError(_PerceivedModuleObservation):
    """Six-value insertion-goal error derived from the shared camera pose."""

    def __call__(self, env, command_name: str = "insertion_goal") -> torch.Tensor:
        pose, _ = self._estimator.estimate()
        return insertion_goal_error_from_module_pose(env, pose, command_name)


class PerceivedModuleVelocity(_PerceivedModuleObservation):
    """Six-value filtered finite-difference velocity from camera pose history."""

    def __call__(self, env) -> torch.Tensor:
        del env
        _, velocity = self._estimator.estimate()
        return velocity


def perceived_module_position_error(env) -> torch.Tensor:
    """Return privileged estimator error for reporting, never for policy input."""

    estimator = getattr(env, "_module_state_estimator", None)
    if estimator is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not isinstance(estimator, ModuleStateEstimator):
        raise TypeError("env._module_state_estimator is not a ModuleStateEstimator")
    return estimator.diagnostic_position_error_m()


_REQUIRED_DEPLOYMENT_TERMS = {
    "grasp": {
        "grip_error": PerceivedGraspError,
        "blade_velocity": PerceivedModuleVelocity,
    },
    "extract": {
        "grip_error": PerceivedGraspError,
        "blade_velocity": PerceivedModuleVelocity,
        "remaining_travel": PerceivedExtractionRemaining,
    },
    "insert": {
        "grip_error": PerceivedGraspError,
        "blade_velocity": PerceivedModuleVelocity,
        "blade_goal_error": PerceivedInsertionGoalError,
    },
}


def audit_vision_deployment_observations(observations) -> None:
    """Fail closed if a deployed vision group exposes exact module state.

    The audit checks both the four required replacements and every other active
    term declared on the three policy groups. It intentionally compares function
    names as well as identities so a rebuilt config cannot evade the contract by
    importing the same forbidden function through another module namespace.
    """

    failures: list[str] = []
    for group_name, required in _REQUIRED_DEPLOYMENT_TERMS.items():
        group = getattr(observations, group_name, None)
        if group is None:
            failures.append(f"missing policy group {group_name!r}")
            continue

        for term_name, expected in required.items():
            term = getattr(group, term_name, None)
            function = getattr(term, "func", None)
            if function is not expected:
                actual = getattr(function, "__name__", repr(function))
                failures.append(f"{group_name}.{term_name} uses {actual}, expected {expected.__name__}")

        for term_name in dir(group):
            if term_name.startswith("_"):
                continue
            term = getattr(group, term_name)
            function = getattr(term, "func", None)
            function_name = getattr(function, "__name__", "")
            if function_name in FORBIDDEN_DEPLOYMENT_OBSERVATION_FUNCTIONS:
                failures.append(f"{group_name}.{term_name} exposes forbidden exact state via {function_name}")

    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(f"vision deployment observation audit failed: {detail}")


def jitter_module_pose(
    env,
    env_ids: torch.Tensor | None,
    position_noise_m: tuple[float, float, float] = (0.0, 0.015, 0.015),
    yaw_noise_rad: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("spare_blade"),
) -> None:
    """Move the module by an episode-random offset that only the camera reveals.

    The pull-axis offset is zero because the rails constrain that direction;
    lateral and vertical offsets remain inside the certified grasp envelope.
    The blind control uses configuration as its prior; this event does not store
    a pre-jitter simulator pose where a policy could later retrieve it.
    """

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    blade = env.scene[asset_cfg.name]
    pose = blade.data.root_state_w[ids, :7].clone()
    noise = blade.data.root_state_w.new_tensor(position_noise_m)
    pose[:, :3] += (2.0 * torch.rand((len(ids), 3), device=env.device) - 1.0) * noise
    if yaw_noise_rad > 0.0:
        yaw = (2.0 * torch.rand(len(ids), device=env.device) - 1.0) * yaw_noise_rad
        axis = pose.new_zeros((len(ids), 3))
        axis[:, 2] = 1.0
        pose[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, axis), pose[:, 3:7])
    blade.write_root_pose_to_sim(pose, env_ids=ids)
    blade.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


__all__ = [
    "FORBIDDEN_DEPLOYMENT_OBSERVATION_FUNCTIONS",
    "MODULE_POSE_DIM",
    "ModulePoseHead",
    "ModuleStateEstimator",
    "PERCEPTION_BLIND",
    "MODULE_VELOCITY_FROM_CAMERA",
    "MODULE_VELOCITY_FROM_KINEMATICS",
    "MODULE_VELOCITY_SOURCES",
    "PERCEPTION_BACKENDS",
    "PERCEPTION_BACKEND_FIDUCIAL_PNP",
    "PERCEPTION_BACKEND_POSE_HEAD",
    "PERCEPTION_DEPLOYMENT",
    "PERCEPTION_MODES",
    "PERCEPTION_ORACLE",
    "PerceivedExtractionRemaining",
    "PerceivedGraspError",
    "PerceivedInsertionGoalError",
    "PerceivedModuleVelocity",
    "audit_vision_deployment_observations",
    "extraction_remaining_from_module_pose",
    "grip_error_from_module_pose",
    "insertion_goal_error_from_module_pose",
    "jitter_module_pose",
    "load_pose_head",
    "module_pose_label",
    "perceived_module_position_error",
    "shared_module_state_estimator",
    "slot_occupancy_label",
]


# Camera miscalibration is applied to the configured sensor mount before the
# environment is constructed. A reset-time ``set_world_poses`` probe previously
# lived here but did not move the tiled camera and was therefore removed.
