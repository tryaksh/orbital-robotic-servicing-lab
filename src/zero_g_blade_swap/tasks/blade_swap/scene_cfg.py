"""Interactive scene configurations for the zero-g blade swap task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .assets import (
    CONTACT_INSERTION_BLADE_CFG,
    GRAPPLE_PIN_BLADE_CFG,
    INSERTION_BLADE_CFG,
    INSERTION_SLOT_CFG,
    INSERTION_SLOT_LEFT_GUIDE_CFG,
    INSERTION_SLOT_RIGHT_GUIDE_CFG,
    MOUNT_ANCHOR_CFG,
    RACK_CFG,
    RIGID_GRASP_BLADE_CFG,
    RIGID_GRASP_SLOT_CFG,
    RIGID_GRASP_SLOT_LEFT_GUIDE_CFG,
    RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG,
    ROBUST_INSERTION_BLADE_CFG,
    ROBUST_INSERTION_SLOT_CFG,
    ROBUST_INSERTION_SLOT_LEFT_GUIDE_CFG,
    ROBUST_INSERTION_SLOT_RIGHT_GUIDE_CFG,
    SLOT_ENTRY_LEFT_FLARE_CFG,
    SLOT_ENTRY_RIGHT_FLARE_CFG,
    SLOT_UPPER_LEFT_LIP_CFG,
    SLOT_UPPER_RIGHT_LIP_CFG,
    CompliantD6JointCfg,
    FixedGraspJointCfg,
    make_contact_insertion_robot_cfg,
    make_grapple_pin_robot_cfg,
    make_insertion_robot_cfg,
)


@configclass
class ZeroGInsertionSceneCfg(InteractiveSceneCfg):
    """Lean Stage-1 scene containing only the assets needed for insertion."""

    robot = make_insertion_robot_cfg()
    spare_blade = INSERTION_BLADE_CFG
    blade_slot = INSERTION_SLOT_CFG
    blade_slot_left_guide = INSERTION_SLOT_LEFT_GUIDE_CFG
    blade_slot_right_guide = INSERTION_SLOT_RIGHT_GUIDE_CFG
    rack = RACK_CFG
    camera: TiledCameraCfg | None = None


@configclass
class ZeroGRobustInsertionSceneCfg(InteractiveSceneCfg):
    """Lean Phase-2 scene with tight rails and a profile-selectable mount."""

    # The anchor remains present at every robustness level so observation
    # dimensions never change when a checkpoint is resumed at a harder level.
    mount_anchor = MOUNT_ANCHOR_CFG
    robot = make_insertion_robot_cfg()
    # Declare this as a concrete asset to preserve creation order (anchor,
    # robot, then joint). Lower robustness profiles replace it with ``None``.
    base_compliance = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BaseCompliance",
        spawn=CompliantD6JointCfg(),
    )
    spare_blade = ROBUST_INSERTION_BLADE_CFG
    blade_slot = ROBUST_INSERTION_SLOT_CFG
    blade_slot_left_guide = ROBUST_INSERTION_SLOT_LEFT_GUIDE_CFG
    blade_slot_right_guide = ROBUST_INSERTION_SLOT_RIGHT_GUIDE_CFG
    rack = RACK_CFG
    # These are populated only by the Play configuration.  Training stays
    # render-free, while live inspection gets deliberate showcase lighting.
    showcase_key_light: AssetBaseCfg | None = None
    showcase_fill_light: AssetBaseCfg | None = None
    camera: TiledCameraCfg | None = None


@configclass
class ZeroGContactInsertionSceneCfg(ZeroGRobustInsertionSceneCfg):
    """Phase-2.5 scene where finger/handle contacts carry the blade."""

    robot = make_contact_insertion_robot_cfg()
    spare_blade = CONTACT_INSERTION_BLADE_CFG


@configclass
class ZeroGRigidGraspInsertionSceneCfg(ZeroGContactInsertionSceneCfg):
    """Insertion scene with a real PhysX fixed joint for a secured grasp."""

    grasp_joint = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GraspJoint",
        spawn=FixedGraspJointCfg(),
    )
    spare_blade = RIGID_GRASP_BLADE_CFG
    blade_slot = RIGID_GRASP_SLOT_CFG
    blade_slot_left_guide = RIGID_GRASP_SLOT_LEFT_GUIDE_CFG
    blade_slot_right_guide = RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG


@configclass
class ZeroGGuidedSlotSceneCfg(ZeroGRigidGraspInsertionSceneCfg):
    """Rigid-grasp scene whose slot is a real channel with a funnelled mouth.

    Adds two overhanging upper lips, which constrain blade pitch and roll that
    the two side walls alone cannot, and two angled lead-in plates at the mouth.
    Declared as a separate scene so the certified Level-0/1/2 geometry is not
    modified underneath three published evaluations.
    """

    blade_slot_upper_left_lip = SLOT_UPPER_LEFT_LIP_CFG
    blade_slot_upper_right_lip = SLOT_UPPER_RIGHT_LIP_CFG
    blade_slot_entry_left_flare = SLOT_ENTRY_LEFT_FLARE_CFG
    blade_slot_entry_right_flare = SLOT_ENTRY_RIGHT_FLARE_CFG


@configclass
class ZeroGGrapplePinSceneCfg(ZeroGContactInsertionSceneCfg):
    """Head-on capture scene: a guided channel and a blade carrying a pin.

    Built on the contact scene rather than the rigid-grasp one on purpose. The
    rigid-grasp scene welds the blade to the tool with a fixed joint and turns
    the slot floor collider off; neither is acceptable for measuring a grasp.
    Here the blade is held only by contact, and the floor is solid, which is
    what forces the gripper to stay outside the rack and therefore what sets the
    pin's length.
    """

    robot = make_grapple_pin_robot_cfg()
    spare_blade = GRAPPLE_PIN_BLADE_CFG
    blade_slot_upper_left_lip = SLOT_UPPER_LEFT_LIP_CFG
    blade_slot_upper_right_lip = SLOT_UPPER_RIGHT_LIP_CFG
    blade_slot_entry_left_flare = SLOT_ENTRY_LEFT_FLARE_CFG
    blade_slot_entry_right_flare = SLOT_ENTRY_RIGHT_FLARE_CFG


def make_tiled_camera_cfg() -> TiledCameraCfg:
    """Create the 64x64, 15 Hz camera used by vision and data collection."""

    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1.0 / 15.0,
        height=64,
        width=64,
        data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.55, -0.65, 1.15),
            rot=(0.9610, -0.0346, 0.1438, 0.2317),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            horizontal_aperture=22.0,
            focus_distance=1.4,
            f_stop=0.0,
            clipping_range=(0.05, 4.0),
        ),
    )


__all__ = [
    "ZeroGInsertionSceneCfg",
    "ZeroGRobustInsertionSceneCfg",
    "ZeroGContactInsertionSceneCfg",
    "ZeroGGrapplePinSceneCfg",
    "ZeroGGuidedSlotSceneCfg",
    "make_tiled_camera_cfg",
]
