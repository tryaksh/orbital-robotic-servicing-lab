"""Where the module is, regressed from a camera instead of read from the sim.

Every certified policy in this repository is handed ``grapple_grip_error_observation``
— the exact vector from the tool frame to the point on the module it has to grip
— computed from simulator ground truth. That is the honest weakness of the whole
project, recorded as such since 2026-08-10. This module replaces that one term
with a value derived from a 64x64 RGB image, and changes nothing else.

**What is perceived and what is not, and why the split is the design.** A real
servicing arm knows its own joint angles, its tool pose, its gripper state and
the forces on it, from encoders and forward kinematics. It does *not* know where
a free-floating module is. So the head regresses the **module's pose in the
world**, which is a pure function of the image, and the tool-to-grip vector the
policy consumes is then computed from that estimate and the arm's own known tool
pose — exactly as a real system would.

Regressing the tool-relative vector directly would have been the easy thing to
write and it would have been wrong: the camera cannot see the gripper from this
mount, so the network would have had to invent the half of the answer that
proprioception already supplies exactly. That mistake was caught by rendering a
frame and projecting the targets into it before any data was collected.

The head is small on purpose. A 64x64 tile at 15 Hz across many environments is
the throughput budget this project measured and kept, and the quantity is a
smooth six-dimensional function of one rigid body's pose, not a semantic
problem. Four strided convolutions take 64x64 to 4x4.
"""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from zero_g_blade_swap.grapple_geometry import (
    EXTRACTED_BLADE_CENTRE_X,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_PIN_GRIP_OFFSET,
)
from zero_g_blade_swap.pose_head import MODULE_POSE_DIM, ModulePoseHead, load_pose_head

from ..assets import SLOT_CENTRE_Y, SLOT_UPPER_LIP_HALF_WIDTH_Y
from .insertion import attached_blade_pose_world
from .observations import camera_rgb_with_radiation_noise, end_effector_pose_world


def module_pose_label(env, asset_name: str = "spare_blade") -> torch.Tensor:
    """The regression target: the module's pose in its own environment's frame.

    Environment-local rather than world, so the label does not encode which tile
    of the cloned grid an environment happens to occupy — a network handed world
    coordinates would learn the grid instead of the module.
    """

    position, orientation = attached_blade_pose_world(env)
    local = position - env.scene.env_origins
    return torch.cat((local, axis_angle_from_quat(orientation)), dim=-1)


def slot_occupancy_label(env, asset_name: str = "spare_blade") -> torch.Tensor:
    """One value per bay: is the module inside that bay's channel?

    The supervision for the pose head's occupancy branch, and the thing that
    turns "the camera locates a part" into "the camera reads the state of the
    rack". With two bays a servicer's first question is which one holds the
    module, and during a relocation the honest answer is sometimes *neither* ---
    the module spends the whole transit outside both. So these are two
    independent indicators rather than a choice between two bays, and the head
    scores them with independent logits for the same reason.

    "Inside the channel" is read off the rack geometry rather than chosen:

    * axially, the module's centre is past ``EXTRACTED_BLADE_CENTRE_X``, which is
      the centre position at which its rear face is level with the mouth --- the
      same line extraction is judged against, so "out" here and "extracted"
      there cannot disagree;
    * laterally, within ``SLOT_UPPER_LIP_HALF_WIDTH_Y`` of that bay's centre,
      which is the physical half-width of the channel the lips define. The bays
      are 0.22 m apart and that half-width is 0.0725 m, so the two indicators
      cannot both be true and a module parked between them sets neither.
    """

    position, _ = attached_blade_pose_world(env)
    local = position - env.scene.env_origins
    centres = local.new_tensor(SLOT_CENTRE_Y)
    inside_mouth = local[:, 0] > EXTRACTED_BLADE_CENTRE_X
    within_channel = (local[:, 1].unsqueeze(-1) - centres).abs() <= SLOT_UPPER_LIP_HALF_WIDTH_Y
    return (within_channel & inside_mouth.unsqueeze(-1)).to(torch.float32)


def grip_error_from_module_pose(env, module_pose: torch.Tensor) -> torch.Tensor:
    """Turn an estimated module pose into the six values the policies consume.

    Every other quantity in here is proprioception: the tool pose comes from
    forward kinematics and the grip offset is a dimension of the interface. This
    is the arithmetic ``grapple_grip_error_observation`` does, with the module's
    pose supplied instead of looked up.
    """

    tool_position, tool_orientation = end_effector_pose_world(env)
    blade_position = module_pose[:, :3] + env.scene.env_origins
    angle = torch.linalg.vector_norm(module_pose[:, 3:], dim=-1, keepdim=True).clamp_min(1e-8)
    blade_orientation = quat_from_angle_axis(angle.squeeze(-1), module_pose[:, 3:] / angle)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    desired = quat_mul(
        blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation)
    )
    relative_position, relative_quat = subtract_frame_transforms(
        tool_position, tool_orientation, grip_position, desired
    )
    return torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)


class PerceivedGraspError(ManagerTermBase):
    """The grip error, from the camera rather than from the simulator.

    Set ``env.cfg.pose_head_checkpoint`` to a trained head. If it is unset the
    term **raises** rather than falling back to ground truth: a vision
    evaluation that quietly used the simulator's own answer is the most
    expensive kind of wrong number this project could produce, because it would
    look like a triumph.

    ``pose_head_oracle_blend`` at 1.0 runs the oracle arm of the comparison
    through this identical code path, so the difference between the two arms
    cannot be an artefact of one of them taking a different route through the
    observation manager.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        checkpoint = getattr(env.cfg, "pose_head_checkpoint", None)
        self._blend = float(getattr(env.cfg, "pose_head_oracle_blend", 0.0))
        self._blind = bool(getattr(env.cfg, "pose_head_blind", False))
        if checkpoint is None and self._blend < 1.0 and not self._blind:
            raise ValueError(
                "PerceivedGraspError needs env.cfg.pose_head_checkpoint. Refusing to fall back to "
                "ground truth: a vision result computed from the simulator's own answer is worse "
                "than no result. Set pose_head_oracle_blend = 1.0 to run the oracle arm deliberately."
            )
        self._head = None if checkpoint is None else load_pose_head(checkpoint, env.device)
        self._sensor_cfg = SceneEntityCfg("camera")
        self._error = torch.zeros(env.num_envs, device=env.device)
        # ``self._blind`` is the control that says whether the camera does any
        # work at all: the robot assumes the module is exactly where the rack
        # nominally presents it, reading no image. If a blind run scores as well
        # as a seeing one, the perception result is measuring nothing.

    def __call__(self, env) -> torch.Tensor:
        truth_pose = module_pose_label(env)
        if self._blind:
            # The pose the rack nominally presents, recorded before the episode's
            # displacement was applied. No image is read at all.
            stored = getattr(env, "_module_nominal_pose", None)
            if stored is None:
                # The observation manager evaluates every term once while the
                # environment is being built, before any reset has run. There is
                # no displacement yet either, so the truth *is* the nominal.
                return grip_error_from_module_pose(env, truth_pose)
            believed = torch.cat(
                (stored[:, :3] - env.scene.env_origins, axis_angle_from_quat(stored[:, 3:7])), dim=-1
            )
            self._error = torch.linalg.vector_norm(believed[:, :3] - truth_pose[:, :3], dim=-1)
            return grip_error_from_module_pose(env, believed)
        if self._blend >= 1.0:
            self._error = torch.zeros_like(self._error)
            return grip_error_from_module_pose(env, truth_pose)
        image = camera_rgb_with_radiation_noise(env, sensor_cfg=self._sensor_cfg)
        with torch.inference_mode():
            predicted = self._head(image).to(truth_pose.dtype)
        # Recorded per step so the evaluation can report how wrong the estimator
        # actually was, rather than only whether the workflow survived it.
        self._error = torch.linalg.vector_norm(predicted[:, :3] - truth_pose[:, :3], dim=-1)
        pose = predicted if self._blend <= 0.0 else torch.lerp(predicted, truth_pose, self._blend)
        return grip_error_from_module_pose(env, pose)

    @property
    def position_error_m(self) -> torch.Tensor:
        """How far the last prediction was from the module's true position."""

        return self._error


def perceived_module_position_error(env) -> torch.Tensor:
    """Per-environment estimator error, for the evaluation to record.

    Zero where there is no estimator -- the oracle arm, and every state-only
    task -- which is the right reading: those have nothing to be wrong by.

    **This function had never run.** It reached for
    ``observation_manager._term_names``, which this Isaac Lab does not have, so
    the first caller got an AttributeError rather than a number. It is written
    against the public ``active_terms`` now, and it is searched across every
    observation group rather than a hard-coded "grasp", because the vision
    profiles put the perceived term in whichever group their parent declared it.
    """

    manager = env.observation_manager
    for group, names in manager.active_terms.items():
        for name, term in zip(names, manager._group_obs_term_cfgs[group], strict=False):
            del name
            if isinstance(term.func, PerceivedGraspError):
                return term.func.position_error_m
    return torch.zeros(env.num_envs, device=env.device)


__all__ = [
    "MODULE_POSE_DIM",
    "ModulePoseHead",
    "PerceivedGraspError",
    "grip_error_from_module_pose",
    "load_pose_head",
    "module_pose_label",
    "perceived_module_position_error",
    "slot_occupancy_label",
]


def jitter_module_pose(
    env,
    env_ids: torch.Tensor | None,
    position_noise_m: tuple[float, float, float] = (0.0, 0.015, 0.015),
    yaw_noise_rad: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("spare_blade"),
) -> None:
    """Present the module at a pose nothing in the observation reveals.

    Without this the vision experiment is hollow. The module's reset pose is one
    of three fixed stage poses, so a head could memorise three positions and
    score beautifully while having learned nothing about seeing. Displacing it by
    an amount drawn per episode makes the image the *only* source of the answer,
    which is the same reasoning that made the pose-belief task move the slot
    physically instead of adding a bias to a reported number.

    The default envelope is deliberately not symmetric. Along the pull axis the
    module is constrained by its rails and cannot move, so ``x`` is zero; across
    the slot and vertically it has room, and 15 mm is comfortably inside the
    20 mm capture tolerance the grasp policy was certified against, so a *perfect*
    estimator would cost nothing and the measurement isolates the estimator.
    """

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    blade = env.scene[asset_cfg.name]
    pose = blade.data.root_state_w[ids, :7].clone()
    # Keep the pose *before* the displacement. That is what a robot working from
    # the drawing rather than from an image would believe, and it is the control
    # the perception result has to be measured against.
    nominal = getattr(env, "_module_nominal_pose", None)
    if nominal is None or nominal.shape[0] != env.num_envs:
        nominal = torch.zeros((env.num_envs, 7), device=env.device)
        env._module_nominal_pose = nominal
    nominal[ids] = pose
    noise = blade.data.root_state_w.new_tensor(position_noise_m)
    pose[:, :3] += (2.0 * torch.rand((len(ids), 3), device=env.device) - 1.0) * noise
    if yaw_noise_rad > 0.0:
        yaw = (2.0 * torch.rand(len(ids), device=env.device) - 1.0) * yaw_noise_rad
        axis = pose.new_zeros((len(ids), 3))
        axis[:, 2] = 1.0
        pose[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, axis), pose[:, 3:7])
    blade.write_root_pose_to_sim(pose, env_ids=ids)
    blade.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


__all__.append("jitter_module_pose")


# ``jitter_camera_pose`` used to live here and was **deleted on 2026-08-15
# because it did nothing**. It called ``set_world_poses`` on the tiled camera
# inside a reset hook; the camera's reported position moved by exactly 0.0 mm
# and two renders 50 mm apart differed by 23.79 levels against a 24.22-level
# camera-noise floor. That is an inert probe, and this project has published one
# of those already.
#
# Camera miscalibration is now applied where it demonstrably takes effect: on
# the sensor's configured mount offset, before the environment is constructed.
# It is constant for a run rather than drawn per episode, which models a
# calibration offset more faithfully anyway -- a mis-mounted camera is
# mis-mounted all day. See ``scripts/sweep_camera_calibration.py``.

