"""Interactive scene configurations for the zero-g blade swap task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from zero_g_blade_swap.servicing_camera import (
    CAMERA_CLIPPING_RANGE_M,
    CAMERA_FOCAL_LENGTH_MM,
    CAMERA_FOCUS_DISTANCE_M,
    CAMERA_HEIGHT_PX,
    CAMERA_HORIZONTAL_APERTURE_MM,
    CAMERA_POSITION_M,
    CAMERA_QUATERNION_WXYZ_ROS,
    CAMERA_UPDATE_PERIOD_S,
    CAMERA_WIDTH_PX,
)

from .assets import (
    CONTACT_INSERTION_BLADE_CFG,
    GRAPPLE_MOUNT_ANCHOR_CFG,
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
    SECOND_SLOT_CFG,
    SECOND_SLOT_ENTRY_LEFT_FLARE_CFG,
    SECOND_SLOT_ENTRY_RIGHT_FLARE_CFG,
    SECOND_SLOT_LEFT_GUIDE_CFG,
    SECOND_SLOT_RIGHT_GUIDE_CFG,
    SECOND_SLOT_UPPER_LEFT_LIP_CFG,
    SECOND_SLOT_UPPER_RIGHT_LIP_CFG,
    SERVICE_LATCH_PRIM,
    SLOT_ENTRY_LEFT_FLARE_CFG,
    SLOT_ENTRY_RIGHT_FLARE_CFG,
    SLOT_UPPER_LEFT_LIP_CFG,
    SLOT_UPPER_RIGHT_LIP_CFG,
    CompliantD6JointCfg,
    FixedGraspJointCfg,
    MatingComplianceJointCfg,
    ReleaseLatchJointCfg,
    ServiceLatchCfg,
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
    # The anchor follows the base. See GRAPPLE_MOUNT_ANCHOR_CFG: leaving it
    # behind terminates every episode on mount_unstable before the arm can act.
    mount_anchor = GRAPPLE_MOUNT_ANCHOR_CFG
    spare_blade = GRAPPLE_PIN_BLADE_CFG
    release_latch_joint = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReleaseLatchJoint",
        spawn=ReleaseLatchJointCfg(),
    )
    # The same mechanism's second state. Disabled until the driver says the
    # module has reached the rack; see ``MatingComplianceJointCfg``.
    mating_compliance_joint = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MatingComplianceJoint",
        spawn=MatingComplianceJointCfg(),
    )
    # The hardware that joint represents. Visual geometry on the wrist, with no
    # collider and no rigid body: the load path is the joint, and this is what
    # the joint *is*, so that a viewer can see which mechanism is holding the
    # module and so its clearances can be checked rather than asserted. See
    # ``src/zero_g_blade_swap/service_latch.py`` and
    # ``scripts/check_service_latch_clearance.py``.
    service_latch = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + SERVICE_LATCH_PRIM,
        spawn=ServiceLatchCfg(),
    )
    # Populated only by the relocation workflow after the payload entity has
    # been declared, preserving the required spawn order.
    payload_stage: AssetBaseCfg | None = None
    blade_slot_upper_left_lip = SLOT_UPPER_LEFT_LIP_CFG
    blade_slot_upper_right_lip = SLOT_UPPER_RIGHT_LIP_CFG
    blade_slot_entry_left_flare = SLOT_ENTRY_LEFT_FLARE_CFG
    blade_slot_entry_right_flare = SLOT_ENTRY_RIGHT_FLARE_CFG


@configclass
class ZeroGTwoSlotGrapplePinSceneCfg(ZeroGGrapplePinSceneCfg):
    """The head-on capture scene with a second bay beside the first.

    A rack with one slot can only demonstrate remove-and-replace; a relocation
    needs somewhere else to put the module. The second slot is the certified one
    displaced to y = -0.22 m, part for part, so the two bays cannot drift apart.

    A separate scene, like every geometry change in this project, so the
    single-slot tasks and their certifications describe unchanged scenes.
    """

    blade_slot_two = SECOND_SLOT_CFG
    blade_slot_two_left_guide = SECOND_SLOT_LEFT_GUIDE_CFG
    blade_slot_two_right_guide = SECOND_SLOT_RIGHT_GUIDE_CFG
    blade_slot_two_upper_left_lip = SECOND_SLOT_UPPER_LEFT_LIP_CFG
    blade_slot_two_upper_right_lip = SECOND_SLOT_UPPER_RIGHT_LIP_CFG
    blade_slot_two_entry_left_flare = SECOND_SLOT_ENTRY_LEFT_FLARE_CFG
    blade_slot_two_entry_right_flare = SECOND_SLOT_ENTRY_RIGHT_FLARE_CFG
    # The destination bay's vertical lead-in, installed only by
    # ``configure_service_destination()``. ``None`` here so every task that
    # existed before it describes exactly the scene its certification was taken
    # on, which is the rule this repository applies to every geometry change.
    blade_slot_two_entry_upper_ramp: RigidObjectCfg | None = None
    blade_slot_two_entry_lower_ramp: RigidObjectCfg | None = None
    rack_retention_hardware: AssetBaseCfg | None = None
    rack_retention_joint: AssetBaseCfg | None = None


def make_tiled_camera_cfg() -> TiledCameraCfg:
    """Create the 384x384, 15 Hz servicing RGB-D overview camera.

    Resolution and focal length scale together, preserving the original
    millimetres-per-pixel while expanding the field around the whole transfer
    workspace. The former 64 px / 180 mm design passed scale arithmetic but
    failed the projection gate: the rack mouth landed at u=-19 px and free
    transit was farther outside still.
    """

    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=CAMERA_UPDATE_PERIOD_S,
        height=CAMERA_HEIGHT_PX,
        width=CAMERA_WIDTH_PX,
        # Metric image-plane depth is paired with RGB for the industrial
        # fiducial path.  RGB identifies the service datum; depth removes the
        # weak, noise-sensitive scale/tilt estimate of single-plane RGB PnP.
        data_types=["rgb", "distance_to_image_plane"],
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_POSITION_M,
            # ROS optical frame aimed vertically at the centre of the complete
            # two-bay workflow envelope. The flush top-face datum is therefore
            # seen near-normal rather than at the former 68-degree incidence.
            rot=CAMERA_QUATERNION_WXYZ_ROS,
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            # The 30 mm aperture widens the earlier crop enough to keep the
            # complete two-bay transfer envelope in frame. 384 px preserves
            # (and slightly improves) its measured ground sampling density.
            focal_length=CAMERA_FOCAL_LENGTH_MM,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE_MM,
            focus_distance=CAMERA_FOCUS_DISTANCE_M,
            f_stop=0.0,
            clipping_range=CAMERA_CLIPPING_RANGE_M,
        ),
    )


__all__ = [
    "ZeroGInsertionSceneCfg",
    "ZeroGRobustInsertionSceneCfg",
    "ZeroGContactInsertionSceneCfg",
    "ZeroGGrapplePinSceneCfg",
    "ZeroGTwoSlotGrapplePinSceneCfg",
    "ZeroGGuidedSlotSceneCfg",
    "make_tiled_camera_cfg",
]
