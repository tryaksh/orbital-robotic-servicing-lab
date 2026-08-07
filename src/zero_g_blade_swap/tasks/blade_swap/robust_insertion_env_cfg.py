"""Phase-2 PPO environment: six-axis insertion under progressive physics gaps."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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
from .assets import CompliantD6JointCfg, make_insertion_robot_cfg, make_robust_insertion_robot_cfg
from .env_cfg import ARM_JOINTS, make_simulation_cfg
from .insertion_env_cfg import ARM_CFG, GRIPPER_CFG, WRIST_CFG
from .scene_cfg import ZeroGRobustInsertionSceneCfg

ROBUSTNESS_PROFILES = (
    "tight_six_axis",
    "pose_error",
    "mass",
    "friction_stiction",
    "mount_wobble",
)


@configclass
class RobustInsertionActionsCfg:
    """Six small relative Cartesian corrections; the gripper remains secured."""

    arm = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="wrist_3_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=mdp.TOOL_OFFSET_POS,
            rot=mdp.TOOL_OFFSET_ROT,
        ),
        # Millimetre-scale translations and roughly one-degree rotations per
        # 30 Hz decision prevent the policy from jumping through tight rails.
        scale=(0.006, 0.002, 0.002, 0.018, 0.018, 0.018),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )


@configclass
class RobustInsertionCommandsCfg:
    insertion_goal = mdp.InsertionGoalCommandCfg()


@configclass
class RobustInsertionPolicyObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ARM_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ARM_CFG})
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    blade_goal_error = ObsTerm(func=mdp.insertion_goal_error)
    blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)
    mount_state = ObsTerm(func=mdp.robot_mount_state)
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class RobustInsertionObservationsCfg:
    policy: RobustInsertionPolicyObsCfg = RobustInsertionPolicyObsCfg()


@configclass
class RobustInsertionEventsCfg:
    """All Phase-2 terms; the selected profile disables later categories."""

    blade_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("spare_blade"),
            "mass_distribution_params": (5.0, 15.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    slot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("blade_slot"),
            "static_friction_range": (0.25, 2.0),
            "dynamic_friction_range": (0.20, 1.5),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    left_guide_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("blade_slot_left_guide"),
            "static_friction_range": (0.25, 2.0),
            "dynamic_friction_range": (0.20, 1.5),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    right_guide_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("blade_slot_right_guide"),
            "static_friction_range": (0.25, 2.0),
            "dynamic_friction_range": (0.20, 1.5),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_progress = EventTerm(func=mdp.reset_insertion_progress, mode="reset")
    reset_arm = EventTerm(
        func=mdp.reset_insertion_joints,
        mode="reset",
        params={"asset_cfg": ARM_CFG, "noise_by_stage": (0.001, 0.002, 0.004)},
    )
    reset_blade = EventTerm(func=mdp.reset_insertion_blade, mode="reset")
    close_gripper_on_reset = EventTerm(
        func=mdp.hold_gripper_closed,
        mode="reset",
        params={"asset_cfg": GRIPPER_CFG},
    )
    randomize_stiction = EventTerm(
        func=mdp.randomize_rail_stiction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("spare_blade"),
            "breakaway_force_range": (10.0, 120.0),
            "viscous_coefficient_range": (2.0, 25.0),
        },
    )
    clear_mount_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
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
            "position_stiffness": 3_000.0,
            "position_damping": 300.0,
            # The secured-blade fixture is a temporary Phase-2 abstraction.
            # Maintain one damping ratio across the 5--15 kg mass range so the
            # abstraction itself does not create a low-mass instability.
            "position_damping_ratio": 0.90,
            "rotation_stiffness": 12.0,
            "rotation_damping": 1.2,
            "maximum_force": 350.0,
            "maximum_torque": 7.5,
        },
    )
    base_wobble = EventTerm(
        func=mdp.MountWobblePulse,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={
            "force_range": (-30.0, 30.0),
            "torque_range": (-6.0, 6.0),
            "duration_range_s": (0.2, 0.8),
            "quiet_range_s": (0.75, 2.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )


@configclass
class RobustInsertionRewardsCfg:
    distance = RewTerm(func=mdp.insertion_distance_reward, weight=2.0)
    progress = RewTerm(func=mdp.insertion_progress_reward, weight=8.0)
    success = RewTerm(func=mdp.insertion_success_reward, weight=20.0)
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.02)
    misalignment = RewTerm(func=mdp.insertion_misalignment_penalty, weight=-0.03)
    settling = RewTerm(func=mdp.insertion_settling_penalty, weight=-0.04)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-5, params={"asset_cfg": ARM_CFG})
    mount_deflection = RewTerm(func=mdp.mount_deflection_penalty, weight=-20.0)
    failure = RewTerm(func=mdp.insertion_failure_reward, weight=-10.0)
    mount_failure = RewTerm(
        func=mdp.undesired_termination_penalty,
        weight=-10.0,
        params={"term_keys": ("mount_unstable",)},
    )


@configclass
class RobustInsertionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insertion_success = DoneTerm(func=mdp.insertion_success_mask)
    insertion_failed = DoneTerm(func=mdp.insertion_failure)
    mount_unstable = DoneTerm(func=mdp.robot_mount_unstable)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class RobustInsertionCurriculumCfg:
    pose_noise = CurrTerm(
        func=mdp.InsertionSuccessRateCurriculum,
        params={
            "success_term": "insertion_success",
            "threshold": 0.80,
            "window_size": 2_000,
            "max_stage": 2,
            "minimum_level_steps": 1_600,
            "stage_mixtures": INSERTION_CURRICULUM_MIXTURES,
        },
    )


@configclass
class ZeroGBladeRobustInsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Insertion task whose physics categories are enabled cumulatively."""

    scene: ZeroGRobustInsertionSceneCfg = ZeroGRobustInsertionSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    observations: RobustInsertionObservationsCfg = RobustInsertionObservationsCfg()
    actions: RobustInsertionActionsCfg = RobustInsertionActionsCfg()
    commands: RobustInsertionCommandsCfg = RobustInsertionCommandsCfg()
    events: RobustInsertionEventsCfg = RobustInsertionEventsCfg()
    rewards: RobustInsertionRewardsCfg = RobustInsertionRewardsCfg()
    terminations: RobustInsertionTerminationsCfg = RobustInsertionTerminationsCfg()
    curriculum: RobustInsertionCurriculumCfg = RobustInsertionCurriculumCfg()

    sim = make_simulation_cfg(render_interval=4)
    decimation: int = 4
    episode_length_s: float = 12.0
    is_finite_horizon: bool = False
    robustness_level: int = 0

    def __post_init__(self) -> None:
        self.viewer.eye = (-0.5, -1.8, 1.25)
        self.viewer.lookat = (0.55, 0.0, 0.72)
        self.num_rerenders_on_reset = 0
        self.configure_robustness(self.robustness_level)

    def configure_robustness(self, level: int) -> None:
        """Select one cumulative profile before constructing the Gym env."""

        if not 0 <= level < len(ROBUSTNESS_PROFILES):
            raise ValueError(f"robustness level must be in [0, {len(ROBUSTNESS_PROFILES) - 1}], got {level}")
        self.robustness_level = level
        # Rebuild so callers may safely change the level after parse_env_cfg.
        self.events = RobustInsertionEventsCfg()
        self.events.reset_arm.params["noise_by_stage"] = (
            (0.001, 0.002, 0.004) if level == 0 else (0.003, 0.006, 0.012)
        )
        if level < 2:
            self.events.blade_mass = None
        if level < 3:
            self.events.slot_material = None
            self.events.left_guide_material = None
            self.events.right_guide_material = None
            self.events.randomize_stiction = None
        if level < 4:
            self.scene.robot = make_insertion_robot_cfg()
            self.scene.base_compliance = None
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None
        else:
            self.scene.robot = make_robust_insertion_robot_cfg()
            self.scene.base_compliance = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/BaseCompliance",
                spawn=CompliantD6JointCfg(),
            )


def configure_insertion_play_presentation(cfg: ManagerBasedRLEnvCfg) -> None:
    """Add a close, well-lit inspection view without changing task physics."""

    # Frame the blade/slot interaction, not the entire arm.
    cfg.viewer.eye = (0.18, -1.05, 1.02)
    cfg.viewer.lookat = (0.76, 0.0, 0.72)

    cfg.scene.showcase_key_light = AssetBaseCfg(
        prim_path="/World/ShowcaseKeyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=4_500.0,
            color=(1.0, 0.94, 0.82),
            angle=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.8924, 0.0990, -0.3696, -0.2391)),
    )
    cfg.scene.showcase_fill_light = AssetBaseCfg(
        prim_path="/World/ShowcaseFillLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=1_500.0,
            color=(0.55, 0.72, 1.0),
            radius=0.20,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, -0.65, 1.15)),
    )

    # High-contrast inspection materials.  Play configs are deep copies, so
    # state-only training scenes retain their original materials.
    blade_material = cfg.scene.spare_blade.spawn.visual_material
    blade_material.diffuse_color = (0.02, 0.48, 0.95)
    blade_material.metallic = 0.35
    blade_material.roughness = 0.28
    cfg.scene.blade_slot.spawn.visual_material.diffuse_color = (0.72, 0.40, 0.03)
    for guide in (cfg.scene.blade_slot_left_guide, cfg.scene.blade_slot_right_guide):
        guide.spawn.visual_material.diffuse_color = (0.62, 0.66, 0.72)
        guide.spawn.visual_material.roughness = 0.25


@configclass
class ZeroGBladeRobustInsertionPlayEnvCfg(ZeroGBladeRobustInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "ROBUSTNESS_PROFILES",
    "configure_insertion_play_presentation",
    "ZeroGBladeRobustInsertionEnvCfg",
    "ZeroGBladeRobustInsertionPlayEnvCfg",
]
