"""Insertion curriculum with a gap-free PhysX fixed-grasp abstraction."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .assets import (
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

    def configure_robustness(self, level: int) -> None:
        """Add contact and physics gaps cumulatively after free insertion."""

        super().configure_robustness(level)
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
