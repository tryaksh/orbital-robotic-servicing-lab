"""Insertion curriculum with a gap-free PhysX fixed-grasp abstraction."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .assets import (
    CONTACT_TOOL_OFFSET_POS,
    RIGID_GRASP_BLADE_CFG,
    RIGID_GRASP_SLOT_CFG,
    RIGID_GRASP_SLOT_LEFT_GUIDE_CFG,
    RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG,
    RIGID_GRASP_TIGHT_LEFT_GUIDE_CFG,
    RIGID_GRASP_TIGHT_RIGHT_GUIDE_CFG,
    RIGID_GRASP_TIGHT_SLOT_CFG,
    RIGID_GRASP_WIDE_LEFT_GUIDE_CFG,
    RIGID_GRASP_WIDE_RIGHT_GUIDE_CFG,
    RIGID_GRASP_WIDE_SLOT_CFG,
)
from .contact_insertion_env_cfg import (
    ContactInsertionActionsCfg,
    ContactInsertionEventsCfg,
    ContactInsertionRewardsCfg,
    ContactInsertionTerminationsCfg,
    ZeroGBladeContactInsertionEnvCfg,
)
from .robust_insertion_env_cfg import (
    RobustInsertionCommandsCfg,
    RobustInsertionCurriculumCfg,
    RobustInsertionObservationsCfg,
    configure_insertion_play_presentation,
)
from .scene_cfg import ZeroGRigidGraspInsertionSceneCfg

# The finger positions the promoted Level-0/1/2 checkpoints ran with, kept
# verbatim so their certifications keep describing the task in this file.
LEGACY_GRIPPER_PREGRASP = (0.45, 0.45, -0.45, 0.45, -0.45, -0.45)
LEGACY_GRIPPER_CLOSED = (0.60, 0.60, -0.60, 0.60, -0.60, -0.60)


@configclass
class RigidGraspInsertionRewardsCfg(ContactInsertionRewardsCfg):
    """Anti-stall rewards for insertion with an already-secured blade."""

    progress = RewTerm(func=mdp.insertion_progress_reward, weight=12.0)
    axial_progress = RewTerm(
        func=mdp.insertion_axial_progress_reward,
        weight=8.0,
        params={"distance_scale": 0.030},
    )
    success = RewTerm(func=mdp.contact_insertion_success_reward, weight=35.0)
    timeout_error = RewTerm(
        func=mdp.insertion_timeout_error_penalty,
        weight=-5.0,
        params={"distance_scale": 0.030},
    )
    # A fixed joint cannot slip. Penalizing its small frame-calibration error
    # taught the inherited policy to freeze without improving task physics.
    grasp_slip = None
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)


@configclass
class ZeroGBladeRigidGraspInsertionEnvCfg(ZeroGBladeContactInsertionEnvCfg):
    """Train insertion after the gripper has already secured the handle."""

    scene: ZeroGRigidGraspInsertionSceneCfg = ZeroGRigidGraspInsertionSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    observations: RobustInsertionObservationsCfg = RobustInsertionObservationsCfg()
    actions: ContactInsertionActionsCfg = ContactInsertionActionsCfg()
    commands: RobustInsertionCommandsCfg = RobustInsertionCommandsCfg()
    events: ContactInsertionEventsCfg = ContactInsertionEventsCfg()
    rewards: RigidGraspInsertionRewardsCfg = RigidGraspInsertionRewardsCfg()
    terminations: ContactInsertionTerminationsCfg = ContactInsertionTerminationsCfg()
    curriculum: RobustInsertionCurriculumCfg = RobustInsertionCurriculumCfg()
    contact_grasp: bool = False
    rigid_grasp: bool = True
    # Pinned, not inherited. The grasp task moved its tool frame onto the finger
    # pads; this task must not follow. The frame below is what the promoted
    # Level-0/1/2 checkpoints were trained against, it reaches the policy through
    # `end_effector_pose_local`, and it anchors the PhysX fixed joint, so
    # changing it would invalidate three certifications at once.
    tool_offset_pos: tuple[float, float, float] = CONTACT_TOOL_OFFSET_POS

    def configure_robustness(self, level: int) -> None:
        """Add contact and physics gaps cumulatively after free insertion."""

        super().configure_robustness(level)
        # Pinned for the same reason as the tool frame above. The grasp task
        # corrected its finger commands after measuring that the convention runs
        # backwards; this task keeps the values its promoted checkpoints ran
        # with. The fingers here cannot touch the blade anyway, because the
        # fixed joint replaces them and handle collision is disabled.
        self.events.close_gripper_on_reset.params["pregrasp_positions"] = LEGACY_GRIPPER_PREGRASP
        self.events.close_gripper_on_reset.params["closed_positions"] = LEGACY_GRIPPER_CLOSED
        self.events.hold_gripper_closed.params["closed_positions"] = LEGACY_GRIPPER_CLOSED
        self.scene.spare_blade = RIGID_GRASP_BLADE_CFG.copy()
        # In zero gravity the lower shelf is not load-bearing. Its tight
        # contact combined with randomized friction produced a non-physical
        # lateral ejection, so Phase 2 uses the two actual side rails plus the
        # explicit axial stiction model. Bottom contact remains a documented
        # geometry-calibration item, not a hidden training instability.
        self.events.slot_material = None
        if level == 0:
            self.scene.blade_slot = RIGID_GRASP_SLOT_CFG.copy()
            self.scene.blade_slot_left_guide = RIGID_GRASP_SLOT_LEFT_GUIDE_CFG.copy()
            self.scene.blade_slot_right_guide = RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG.copy()
        elif level == 1:
            # Wide side rails first; the floor stays collision-free so PPO
            # learns lateral correction without simultaneous vertical contact.
            self.scene.blade_slot = RIGID_GRASP_WIDE_SLOT_CFG.copy()
            self.scene.blade_slot_left_guide = RIGID_GRASP_WIDE_LEFT_GUIDE_CFG.copy()
            self.scene.blade_slot_right_guide = RIGID_GRASP_WIDE_RIGHT_GUIDE_CFG.copy()
        else:
            self.scene.blade_slot = RIGID_GRASP_TIGHT_SLOT_CFG.copy()
            self.scene.blade_slot_left_guide = RIGID_GRASP_TIGHT_LEFT_GUIDE_CFG.copy()
            self.scene.blade_slot_right_guide = RIGID_GRASP_TIGHT_RIGHT_GUIDE_CFG.copy()
            # Level 2 isolates tight side rails plus mass. Level 3 adds guide
            # friction/stiction; Level 4 adds floating-mount wobble.
            self.scene.blade_slot.spawn.collision_props.collision_enabled = False


@configclass
class ZeroGBladeRigidGraspInsertionPlayEnvCfg(ZeroGBladeRigidGraspInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "RigidGraspInsertionRewardsCfg",
    "ZeroGBladeRigidGraspInsertionEnvCfg",
    "ZeroGBladeRigidGraspInsertionPlayEnvCfg",
]
