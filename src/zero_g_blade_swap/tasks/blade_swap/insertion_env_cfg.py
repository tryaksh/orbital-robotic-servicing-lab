"""Stage-1 PPO environment: insert a blade already secured in the gripper."""

from __future__ import annotations

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from zero_g_blade_swap.math_utils import INSERTION_CURRICULUM_MIXTURES

from . import mdp
from .env_cfg import ARM_JOINTS, make_simulation_cfg
from .mdp.actions import ROBOTIQ_2F85_JOINT_NAMES
from .scene_cfg import ZeroGInsertionSceneCfg

ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)
WRIST_CFG = SceneEntityCfg("robot", body_names=["wrist_3_link"])
GRIPPER_CFG = SceneEntityCfg("robot", joint_names=list(ROBOTIQ_2F85_JOINT_NAMES), preserve_order=True)


@configclass
class InsertionActionsCfg:
    """Learn insertion and lateral/vertical alignment; IK holds orientation."""

    arm = mdp.TranslationalDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="wrist_3_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=mdp.TOOL_OFFSET_POS,
            rot=mdp.TOOL_OFFSET_ROT,
        ),
        # Keep the proven axial authority and use smaller corrections across
        # the 5 mm-clearance guide rails.
        scale=(0.006, 0.002, 0.002),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )


@configclass
class InsertionCommandsCfg:
    insertion_goal = mdp.InsertionGoalCommandCfg()


@configclass
class InsertionPolicyObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ARM_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ARM_CFG})
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    blade_goal_error = ObsTerm(func=mdp.insertion_goal_error)
    blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class InsertionObservationsCfg:
    policy: InsertionPolicyObsCfg = InsertionPolicyObsCfg()


@configclass
class InsertionEventsCfg:
    """Nominal physics only; robustness is introduced after nominal promotion."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_progress = EventTerm(func=mdp.reset_insertion_progress, mode="reset")
    reset_arm = EventTerm(
        func=mdp.reset_insertion_joints,
        mode="reset",
        params={
            "asset_cfg": ARM_CFG,
            "noise_by_stage": (0.001, 0.002, 0.004),
        },
    )
    reset_blade = EventTerm(func=mdp.reset_insertion_blade, mode="reset")
    close_gripper_on_reset = EventTerm(
        func=mdp.hold_gripper_closed,
        mode="reset",
        params={"asset_cfg": GRIPPER_CFG},
    )
    hold_gripper_closed = EventTerm(
        func=mdp.hold_gripper_closed,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={"asset_cfg": GRIPPER_CFG},
    )
    secured_blade_constraint = EventTerm(
        func=mdp.SecuredBladeConstraint,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={
            "position_stiffness": 2_500.0,
            "position_damping": 250.0,
            # The 10 kg blade's long-axis inertia is only about 0.02 kg m^2;
            # larger gains over-rotate it between 120 Hz physics steps.
            "rotation_stiffness": 10.0,
            "rotation_damping": 1.0,
            "maximum_force": 150.0,
            "maximum_torque": 5.0,
        },
    )


@configclass
class InsertionRewardsCfg:
    distance = RewTerm(func=mdp.insertion_distance_reward, weight=2.0)
    progress = RewTerm(func=mdp.insertion_progress_reward, weight=8.0)
    success = RewTerm(func=mdp.insertion_success_reward, weight=20.0)
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.02)
    misalignment = RewTerm(func=mdp.insertion_misalignment_penalty, weight=-0.02)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-5, params={"asset_cfg": ARM_CFG})
    failure = RewTerm(func=mdp.insertion_failure_reward, weight=-10.0)


@configclass
class InsertionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insertion_success = DoneTerm(func=mdp.insertion_success_mask)
    insertion_failed = DoneTerm(func=mdp.insertion_failure)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class InsertionCurriculumCfg:
    pose_noise = CurrTerm(
        func=mdp.InsertionSuccessRateCurriculum,
        params={
            "success_term": "insertion_success",
            "threshold": 0.80,
            "window_size": 2_000,
            "max_stage": 2,
            # Fifty PPO iterations at horizon 32. This is longer than the
            # 360-control-step timeout, so early successes cannot promote a
            # level before its first timeouts have completed.
            "minimum_level_steps": 1_600,
            "stage_mixtures": INSERTION_CURRICULUM_MIXTURES,
        },
    )


@configclass
class ZeroGBladeInsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Fast state-only task used to learn the first reliable PPO skill."""

    scene: ZeroGInsertionSceneCfg = ZeroGInsertionSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    observations: InsertionObservationsCfg = InsertionObservationsCfg()
    actions: InsertionActionsCfg = InsertionActionsCfg()
    commands: InsertionCommandsCfg = InsertionCommandsCfg()
    events: InsertionEventsCfg = InsertionEventsCfg()
    rewards: InsertionRewardsCfg = InsertionRewardsCfg()
    terminations: InsertionTerminationsCfg = InsertionTerminationsCfg()
    curriculum: InsertionCurriculumCfg = InsertionCurriculumCfg()

    sim = make_simulation_cfg(render_interval=4)
    decimation: int = 4
    episode_length_s: float = 12.0
    is_finite_horizon: bool = False

    def __post_init__(self) -> None:
        self.viewer.eye = (-0.5, -1.8, 1.25)
        self.viewer.lookat = (0.55, 0.0, 0.72)
        self.num_rerenders_on_reset = 0


@configclass
class ZeroGBladeInsertionPlayEnvCfg(ZeroGBladeInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


__all__ = ["ZeroGBladeInsertionEnvCfg", "ZeroGBladeInsertionPlayEnvCfg"]
