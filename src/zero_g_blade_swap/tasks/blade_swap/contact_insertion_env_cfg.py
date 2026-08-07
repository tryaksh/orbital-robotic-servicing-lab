"""Phase-2.5 PPO environment: insertion through a real finger/handle grasp."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .assets import (
    CONTACT_INSERTION_STAGE_ARM_JOINT_POS,
    CONTACT_INSERTION_STAGE_BLADE_POSE,
    CONTACT_TOOL_OFFSET_POS,
    CompliantD6JointCfg,
    make_contact_insertion_robot_cfg,
)
from .env_cfg import ARM_JOINTS
from .insertion_env_cfg import ARM_CFG, GRIPPER_CFG
from .robust_insertion_env_cfg import (
    ROBUSTNESS_PROFILES,
    RobustInsertionActionsCfg,
    RobustInsertionCommandsCfg,
    RobustInsertionCurriculumCfg,
    RobustInsertionEventsCfg,
    RobustInsertionObservationsCfg,
    RobustInsertionRewardsCfg,
    RobustInsertionTerminationsCfg,
    ZeroGBladeRobustInsertionEnvCfg,
    configure_insertion_play_presentation,
)
from .scene_cfg import ZeroGContactInsertionSceneCfg

CONTACT_GRIPPER_PREGRASP = (0.45, 0.45, -0.45, 0.45, -0.45, -0.45)
# NVIDIA's 2F-85 assembly task uses 0.45--0.69 depending on object width.
# Use a firm, non-saturating close target; actuator effort remains at
# NVIDIA's official 10 N-m setting.
CONTACT_GRIPPER_CLOSED = (0.60, 0.60, -0.60, 0.60, -0.60, -0.60)


@configclass
class ContactInsertionActionsCfg(RobustInsertionActionsCfg):
    """Six PPO corrections gated until the physical grasp has settled."""

    arm = mdp.GraspSettlingDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="wrist_3_link",
        body_offset=mdp.GraspSettlingDifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=CONTACT_TOOL_OFFSET_POS,
            rot=mdp.TOOL_OFFSET_ROT,
        ),
        # 45 mm/s maximum axial motion at 30 Hz is slow enough for frictional
        # grasping but still covers the full 170 mm approach within an episode.
        scale=(0.0015, 0.00075, 0.00075, 0.006, 0.006, 0.006),
        settling_time_s=0.30,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )


@configclass
class ContactInsertionEventsCfg(RobustInsertionEventsCfg):
    """Use PhysX contact for grasp transmission; never attach the blade."""

    reset_arm = EventTerm(
        func=mdp.reset_insertion_joints,
        mode="reset",
        params={
            "asset_cfg": ARM_CFG,
            "noise_by_stage": (0.001, 0.002, 0.004),
            "poses_by_stage": CONTACT_INSERTION_STAGE_ARM_JOINT_POS,
        },
    )
    close_gripper_on_reset = EventTerm(
        func=mdp.reset_contact_gripper,
        mode="reset",
        params={
            "asset_cfg": GRIPPER_CFG,
            "pregrasp_positions": CONTACT_GRIPPER_PREGRASP,
            "closed_positions": CONTACT_GRIPPER_CLOSED,
        },
    )
    reset_blade = EventTerm(
        func=mdp.reset_insertion_blade,
        mode="reset",
        params={"poses_by_stage": CONTACT_INSERTION_STAGE_BLADE_POSE},
    )
    hold_gripper_closed = EventTerm(
        func=mdp.hold_gripper_closed,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={"asset_cfg": GRIPPER_CFG, "closed_positions": CONTACT_GRIPPER_CLOSED},
    )
    # Explicitly remove the Phase-2 spring fixture.  The blade can move only
    # through contacts, rail resistance, and its own inertia.
    secured_blade_constraint = None
    rail_stiction_force = EventTerm(
        func=mdp.RailStictionForce,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={"asset_cfg": SceneEntityCfg("spare_blade")},
    )


@configclass
class ContactInsertionRewardsCfg(RobustInsertionRewardsCfg):
    # Constant distance/retention rewards let PPO earn points by holding still.
    # Contact training instead pays only for reducing insertion error.
    distance = None
    progress = RewTerm(func=mdp.insertion_progress_reward, weight=12.0)
    success = RewTerm(func=mdp.contact_insertion_success_reward, weight=25.0)
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.10)
    grasp_retention = None
    grasp_slip = RewTerm(func=mdp.grasp_slip_penalty, weight=-0.50)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    failure = RewTerm(func=mdp.insertion_failure_reward, weight=-15.0)


@configclass
class ContactInsertionTerminationsCfg(RobustInsertionTerminationsCfg):
    insertion_success = DoneTerm(func=mdp.contact_insertion_success_mask)
    insertion_failed = DoneTerm(
        func=mdp.insertion_failure,
        params={"grasp_position_limit": 0.025, "grasp_orientation_limit": 0.25},
    )


@configclass
class ZeroGBladeContactInsertionEnvCfg(ZeroGBladeRobustInsertionEnvCfg):
    """Fine-tune the six-axis policy while real contacts retain the blade."""

    scene: ZeroGContactInsertionSceneCfg = ZeroGContactInsertionSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    observations: RobustInsertionObservationsCfg = RobustInsertionObservationsCfg()
    actions: ContactInsertionActionsCfg = ContactInsertionActionsCfg()
    commands: RobustInsertionCommandsCfg = RobustInsertionCommandsCfg()
    events: ContactInsertionEventsCfg = ContactInsertionEventsCfg()
    rewards: ContactInsertionRewardsCfg = ContactInsertionRewardsCfg()
    terminations: ContactInsertionTerminationsCfg = ContactInsertionTerminationsCfg()
    curriculum: RobustInsertionCurriculumCfg = RobustInsertionCurriculumCfg()
    contact_grasp: bool = True
    tool_offset_pos: tuple[float, float, float] = CONTACT_TOOL_OFFSET_POS

    def configure_robustness(self, level: int) -> None:
        """Enable the same cumulative gaps without reintroducing the fixture."""

        if not 0 <= level < len(ROBUSTNESS_PROFILES):
            raise ValueError(f"robustness level must be in [0, {len(ROBUSTNESS_PROFILES) - 1}], got {level}")
        self.robustness_level = level
        self.events = ContactInsertionEventsCfg()
        self.events.reset_arm.params["noise_by_stage"] = (
            (0.0005, 0.001, 0.002) if level == 0 else (0.001, 0.002, 0.004)
        )
        if level < 2:
            self.events.blade_mass = None
        if level < 3:
            self.events.slot_material = None
            self.events.left_guide_material = None
            self.events.right_guide_material = None
            self.events.randomize_stiction = None
            self.events.rail_stiction_force = None
        if level < 4:
            self.scene.robot = make_contact_insertion_robot_cfg()
            self.scene.base_compliance = None
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None
        else:
            self.scene.robot = make_contact_insertion_robot_cfg(floating=True)
            self.scene.base_compliance = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/BaseCompliance",
                spawn=CompliantD6JointCfg(),
            )


@configclass
class ZeroGBladeContactInsertionPlayEnvCfg(ZeroGBladeContactInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "ContactInsertionActionsCfg",
    "ContactInsertionEventsCfg",
    "ZeroGBladeContactInsertionEnvCfg",
    "ZeroGBladeContactInsertionPlayEnvCfg",
]
