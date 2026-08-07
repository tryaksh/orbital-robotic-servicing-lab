"""Task-specific action terms."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

ROBOTIQ_2F85_JOINT_NAMES = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
)
ROBOTIQ_2F85_COUPLING_SIGNS = (1.0, 1.0, -1.0, 1.0, -1.0, -1.0)


class TranslationalDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Expose base-frame XYZ translation while retaining the reset orientation."""

    def __init__(self, cfg: TranslationalDifferentialInverseKinematicsActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._cartesian_command = torch.zeros((self.num_envs, 6), device=self.device)
        self._reference_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self._reference_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Translational IK action must have shape {tuple(self._raw_actions.shape)}, got {actions.shape}."
            )
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(actions * self._scale)
        ee_position, ee_orientation = self._compute_frame_pose()
        initialize = ~self._reference_valid
        self._reference_orientation[initialize] = ee_orientation[initialize]
        self._reference_valid[initialize] = True

        target_position = ee_position + self._processed_actions
        position_error, orientation_error = math_utils.compute_pose_error(
            ee_position,
            ee_orientation,
            target_position,
            self._reference_orientation,
        )
        self._cartesian_command[:, :3] = position_error.clamp(-0.012, 0.012)
        self._cartesian_command[:, 3:] = orientation_error.clamp(-0.05, 0.05)
        self._ik_controller.set_command(self._cartesian_command, ee_position, ee_orientation)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._reference_valid[ids] = False


@configclass
class TranslationalDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = TranslationalDifferentialInverseKinematicsAction


class GraspSettlingDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Hold the arm briefly while the physical gripper establishes contact."""

    cfg: GraspSettlingDifferentialInverseKinematicsActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.settling_time_s <= 0.0:
            super().process_actions(actions)
            return
        safe_actions = actions.clone()
        settling = self._env.episode_length_buf.to(torch.float32) * float(self._env.step_dt) < self.cfg.settling_time_s
        safe_actions[settling] = 0.0
        super().process_actions(safe_actions)


@configclass
class GraspSettlingDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = GraspSettlingDifferentialInverseKinematicsAction
    settling_time_s: float = 0.30


def robotiq_2f85_coupled_targets(command: torch.Tensor) -> torch.Tensor:
    """Expand ``(..., 1)`` finger motion using the official 2F-85 signs.

    This helper is intentionally pure so its coupling contract can be tested
    without launching Isaac Sim.
    """

    if command.shape[-1] != 1:
        raise ValueError(f"Expected a final command dimension of one, got shape {tuple(command.shape)}.")
    signs = command.new_tensor(ROBOTIQ_2F85_COUPLING_SIGNS)
    return command * signs


class RobotiqBinaryAction(ActionTerm):
    """Map one scalar to the actuated Robotiq 2F-85 finger joint.

    The action changes only the real gripper drive target.  It does not attach
    the blade, teleport it, or create a runtime fixed joint; grasping therefore
    succeeds only through the Robotiq collision geometry and PhysX contacts.
    """

    cfg: RobotiqBinaryActionCfg

    def __init__(self, cfg: RobotiqBinaryActionCfg, env) -> None:
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError(f"RobotiqBinaryAction requires an Articulation, got {type(self._asset).__name__}.")

        joint_ids, joint_names = self._asset.find_joints(list(cfg.joint_names), preserve_order=True)
        if len(joint_ids) != len(ROBOTIQ_2F85_JOINT_NAMES):
            raise ValueError(
                "Expected the six ordered Robotiq 2F-85 coupled joints, "
                f"resolved {joint_names}. The 2F-85 USD joint layout may have changed."
            )
        if tuple(joint_names) != tuple(cfg.joint_names):
            raise ValueError(f"Robotiq joint order mismatch: expected {cfg.joint_names}, resolved {joint_names}.")
        self._joint_ids = joint_ids
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, 6), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Robotiq action must have shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}."
            )
        self._raw_actions.copy_(actions)
        close_mask = actions > self.cfg.threshold if self.cfg.close_on_positive else actions < self.cfg.threshold
        open_target = torch.full_like(actions, self.cfg.open_position)
        closed_target = torch.full_like(actions, self.cfg.closed_position)
        scalar_target = torch.where(close_mask, closed_target, open_target)
        self._processed_actions.copy_(robotiq_2f85_coupled_targets(scalar_target))

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = env_ids if env_ids is not None else slice(None)
        self._raw_actions[ids] = 0.0
        open_command = self._raw_actions[ids].new_full(self._raw_actions[ids].shape, self.cfg.open_position)
        self._processed_actions[ids] = robotiq_2f85_coupled_targets(open_command)


@configclass
class RobotiqBinaryActionCfg(ActionTermCfg):
    """Configuration for :class:`RobotiqBinaryAction`."""

    class_type: type[ActionTerm] = RobotiqBinaryAction
    joint_names: tuple[str, ...] = ROBOTIQ_2F85_JOINT_NAMES
    open_position: float = 0.0
    closed_position: float = 0.45
    threshold: float = 0.0
    close_on_positive: bool = True


__all__ = [
    "ROBOTIQ_2F85_COUPLING_SIGNS",
    "ROBOTIQ_2F85_JOINT_NAMES",
    "GraspSettlingDifferentialInverseKinematicsAction",
    "GraspSettlingDifferentialInverseKinematicsActionCfg",
    "RobotiqBinaryAction",
    "RobotiqBinaryActionCfg",
    "TranslationalDifferentialInverseKinematicsAction",
    "TranslationalDifferentialInverseKinematicsActionCfg",
    "robotiq_2f85_coupled_targets",
]
