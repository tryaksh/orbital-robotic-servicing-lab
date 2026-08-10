"""Head-on capture on a grapple pin, and the skills it makes possible.

A parallel-jaw grip cannot hold this blade against extraction. That is
structural, not a tuning failure: the gripper closes along one axis while the
blade leaves along another, the rails must leave the extraction axis free, and
flat pads on a smooth post can then oppose the pull with nothing but friction.
Measured axial capacity that way is about 6 N against the 66.4 N the promoted
insertion policy's own contact reaction demands.

Putting geometry on the pull axis means approaching along it, which moves the
interface onto the module. The blade therefore carries a tapered grapple pin on
its ``-x`` face and the gripper takes it head-on. Pulling drags thicker pin into
the pads and forces them apart against the drive, so the holding force comes
from the taper rather than from a friction coefficient.

Every dimension involved is derived in ``assets.py`` from
``evidence/gripper_collision_envelope.json``, which measures the 2F-85 from its
collision meshes rather than from body origins. That distinction matters here:
every 2F-85 body in this asset is collapsed to within 18 mm of the flange, and
reading those origins as pad locations is what produced a retracted claim once
already.

Nothing here touches the geometry the promoted Level-0/1/2 insertion policies
were certified on. This is a separate scene and separate registrations.
"""

from __future__ import annotations

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from . import mdp
from .assets import (
    CONTACT_INSERTION_STAGE_BLADE_POSE,
    GRAPPLE_HEAD_ON_ARM_JOINT_POS,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_TOOL_OFFSET_POS,
    make_grapple_pin_robot_cfg,
)
from .contact_insertion_env_cfg import (
    ContactInsertionActionsCfg,
    ContactInsertionEventsCfg,
    ZeroGBladeContactInsertionEnvCfg,
)
from .env_cfg import ARM_JOINTS
from .insertion_env_cfg import ARM_CFG, GRIPPER_CFG
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .scene_cfg import ZeroGGrapplePinSceneCfg

# Finger commands, in the measured convention: ``finger_joint`` runs 0 to
# 0.8203 rad, the clear opening is 87.08 mm at 0 and closes at 106.2 mm/rad, so
# *zero is fully open*. The names in this file mean what they say only because
# that was measured; the previous constants in this project were inverted.
#
# Approach at 84.9 mm, which clears the wedge's 70 mm free end by 7.5 mm a side.
# Capture commands far past where the pads can actually reach: seated against
# the collar their trailing edges sit on 67.3 mm of wedge, so they stop at
# about 0.186 rad and everything beyond that is converted by the 40 N-m/rad
# drive into grip force rather than motion. 0.55 rad saturates the 10 N-m
# limit, which is the most grip this actuator model can produce.
GRAPPLE_GRIPPER_APPROACH = (0.02, 0.02, -0.02, 0.02, -0.02, -0.02)
GRAPPLE_GRIPPER_CAPTURE = (0.55, 0.55, -0.55, 0.55, -0.55, -0.55)

# The tool points along +x with the closing axis vertical, and each stage places
# the pads around the wedge with their leading faces one closing stroke short of
# the collar, so closing seats them on it. Solved in ``assets.py``.


@configclass
class GrapplePinActionsCfg(ContactInsertionActionsCfg):
    """Six Cartesian corrections about the frame the pads actually grip with."""

    arm = mdp.GraspSettlingDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="wrist_3_link",
        body_offset=mdp.GraspSettlingDifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=GRAPPLE_TOOL_OFFSET_POS,
            rot=mdp.TOOL_OFFSET_ROT,
        ),
        scale=(0.0015, 0.00075, 0.00075, 0.006, 0.006, 0.006),
        settling_time_s=0.30,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )


@configclass
class GrapplePinEventsCfg(ContactInsertionEventsCfg):
    """Reset head-on with the fingers open around the wedge, then capture."""

    reset_arm = EventTerm(
        func=mdp.reset_insertion_joints,
        mode="reset",
        params={
            "asset_cfg": ARM_CFG,
            "noise_by_stage": (0.001, 0.002, 0.004),
            "poses_by_stage": GRAPPLE_HEAD_ON_ARM_JOINT_POS,
        },
    )
    close_gripper_on_reset = EventTerm(
        func=mdp.reset_contact_gripper,
        mode="reset",
        params={
            "asset_cfg": GRIPPER_CFG,
            "pregrasp_positions": GRAPPLE_GRIPPER_APPROACH,
            "closed_positions": GRAPPLE_GRIPPER_CAPTURE,
        },
    )
    hold_gripper_closed = EventTerm(
        func=mdp.hold_gripper_closed,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={"asset_cfg": GRIPPER_CFG, "closed_positions": GRAPPLE_GRIPPER_CAPTURE},
    )


@configclass
class ZeroGBladeGrapplePinCaptureEnvCfg(ZeroGBladeContactInsertionEnvCfg):
    """Close a head-on capture on the pin while the channel still holds the blade.

    Squeezing a free-floating mass in zero gravity ejects it, so capture happens
    on a blade the rails and lips are still constraining, and only then is it
    broken free. The blade reset poses are the contact curriculum's, which park
    it inside the channel with the pin protruding through the slot mouth.
    """

    scene: ZeroGGrapplePinSceneCfg = ZeroGGrapplePinSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    actions: GrapplePinActionsCfg = GrapplePinActionsCfg()
    events: GrapplePinEventsCfg = GrapplePinEventsCfg()
    contact_grasp: bool = True
    tool_offset_pos: tuple[float, float, float] = GRAPPLE_TOOL_OFFSET_POS
    # World orientation the tool frame holds for a head-on capture: the +z
    # approach axis along world +x, and the closing axis vertical.
    tool_target_rot: tuple[float, float, float, float] = GRAPPLE_HEAD_ON_TOOL_ROT

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # The parent rebuilds the event set for the chosen level from the
        # contact task's class, which carries the top-down poses and the old
        # finger commands, so re-assert the head-on ones afterwards.
        self.events = GrapplePinEventsCfg()
        self.events.reset_arm.params["noise_by_stage"] = (
            (0.0005, 0.001, 0.002) if level == 0 else (0.001, 0.002, 0.004)
        )
        self.events.reset_blade.params["poses_by_stage"] = CONTACT_INSERTION_STAGE_BLADE_POSE
        if level < 2:
            self.events.blade_mass = None
        if level < 3:
            self.events.slot_material = None
            self.events.left_guide_material = None
            self.events.right_guide_material = None
            self.events.randomize_stiction = None
            self.events.rail_stiction_force = None
        # The parent swaps in the contact robot, which spawns top-down and keeps
        # the inherited 10 N-m finger drive. Put the head-on, rated-force robot
        # back; it is the whole point of this task.
        self.scene.robot = make_grapple_pin_robot_cfg(floating=level >= 4)
        if level < 4:
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None


@configclass
class ZeroGBladeGrapplePinCapturePlayEnvCfg(ZeroGBladeGrapplePinCaptureEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "GRAPPLE_GRIPPER_APPROACH",
    "GRAPPLE_GRIPPER_CAPTURE",
    "GrapplePinActionsCfg",
    "GrapplePinEventsCfg",
    "ZeroGBladeGrapplePinCaptureEnvCfg",
    "ZeroGBladeGrapplePinCapturePlayEnvCfg",
]
