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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from zero_g_blade_swap.math_utils import INSERTION_CURRICULUM_MIXTURES

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
from .insertion_env_cfg import ARM_CFG, GRIPPER_CFG, WRIST_CFG
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .scene_cfg import ZeroGGrapplePinSceneCfg

# Finger commands, in the measured convention: ``finger_joint`` runs 0 to
# 0.8203 rad, the clear opening is 87.08 mm at 0 and closes at 106.2 mm/rad, so
# *zero is fully open*. The names in this file mean what they say only because
# that was measured; the previous constants in this project were inverted.
#
# Approach at 84.9 mm, which clears the wedge's 70 mm free end by 7.5 mm a side.
#
# Capture and hold are two different commands, and that is what makes the pull
# gate pass. The wedge converts closing force into thrust along the pull axis,
# so a firm capture drives the payload away before it has been taken; holding,
# once the pin is seated, wants everything the drive can produce. Measured on
# the same grid: one command throughout holds 59 N, capturing at 0.48 and
# firming to 0.68 holds 69 N, against a 66.4 N gate.
#
# The window is narrow and asymmetric. 0.44 gives 63 N and 0.52 gives 68 N, but
# 0.56 collapses to 26 N, so the capture command is biased low.
# See evidence/grapple_pin_capture_plateau.json.
GRAPPLE_GRIPPER_APPROACH = (0.02, 0.02, -0.02, 0.02, -0.02, -0.02)
GRAPPLE_GRIPPER_CAPTURE = (0.48, 0.48, -0.48, 0.48, -0.48, -0.48)
GRAPPLE_GRIPPER_HOLD = (0.68, 0.68, -0.68, 0.68, -0.68, -0.68)
# Where the fingers actually come to rest on the wedge, measured. Extract and
# insert start already captured, so writing this avoids replaying a closing
# transient at full holding force, which is the case that measured 26 N.
GRAPPLE_FINGER_SEATED_RAD = 0.223

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
        # Hold the arm still while the capture completes and preloads the
        # collar. The pull gate needed 1.0 s to settle; acting before that is
        # acting on a grip that has not taken load yet.
        settling_time_s=1.0,
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
        params={"asset_cfg": GRIPPER_CFG, "closed_positions": GRAPPLE_GRIPPER_HOLD},
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


# ---------------------------------------------------------------------------
# The three skills a replacement demonstration needs, gated separately.
#
# They share a scene, a tool frame, and a capture attitude, and nothing else.
# Each has its own reward, its own success predicate, and its own failure
# predicate, because "did the grasp form", "is the blade clear of the rack", and
# "is the blade seated in the slot" are three different questions and a single
# blended reward would let a policy trade one against another.


@configclass
class GrappleSkillObsCfg(ObsGroup):
    """What all three skills can see.

    ``gripper_state`` is the term that matters and is new here. It reports the
    finger angle *and* the drive torque, because the angle alone cannot tell
    fingers closed on a pin from fingers closed on nothing, and not seeing that
    difference is what hid the absent grasp in this project for three sessions.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ARM_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ARM_CFG})
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    grip_error = ObsTerm(func=mdp.grapple_grip_error_observation)
    gripper_state = ObsTerm(func=mdp.gripper_state_observation)
    blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GraspObservationsCfg:
    policy: GrappleSkillObsCfg = GrappleSkillObsCfg()


@configclass
class ExtractPolicyObsCfg(GrappleSkillObsCfg):
    remaining_travel = ObsTerm(func=mdp.extraction_remaining_observation)


@configclass
class ExtractObservationsCfg:
    policy: ExtractPolicyObsCfg = ExtractPolicyObsCfg()


@configclass
class InsertPolicyObsCfg(GrappleSkillObsCfg):
    blade_goal_error = ObsTerm(func=mdp.insertion_goal_error)


@configclass
class InsertObservationsCfg:
    policy: InsertPolicyObsCfg = InsertPolicyObsCfg()


@configclass
class GraspActionsCfg(GrapplePinActionsCfg):
    """Six Cartesian corrections plus the decision to close.

    The grasp skill is the only one of the three that commands the gripper. A
    policy that cannot choose when to close is not learning to grasp, it is
    learning to arrive somewhere while a script closes for it.
    """

    arm = mdp.GraspSettlingDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="wrist_3_link",
        body_offset=mdp.GraspSettlingDifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=GRAPPLE_TOOL_OFFSET_POS,
            rot=mdp.TOOL_OFFSET_ROT,
        ),
        # Finer than the insertion scales: the wedge's free end has 8.5 mm of
        # clearance a side inside the pad aperture, so alignment is the task.
        scale=(0.002, 0.001, 0.001, 0.008, 0.008, 0.008),
        # Hold through the reset transient. A reset writes joint positions but
        # leaves the previous episode's actuator targets, so the arm springs for
        # a few steps before it settles.
        settling_time_s=0.30,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )
    gripper = mdp.TwoStageRobotiqActionCfg(
        asset_name="robot",
        open_position=GRAPPLE_GRIPPER_APPROACH[0],
        closed_position=GRAPPLE_GRIPPER_CAPTURE[0],
        hold_position=GRAPPLE_GRIPPER_HOLD[0],
    )


@configclass
class ExtractActionsCfg(GrapplePinActionsCfg):
    """Six corrections; the fingers stay where the capture left them."""

    arm = mdp.GraspSettlingDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="wrist_3_link",
        body_offset=mdp.GraspSettlingDifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=GRAPPLE_TOOL_OFFSET_POS,
            rot=mdp.TOOL_OFFSET_ROT,
        ),
        # 120 mm/s along the pull axis. The blade has to travel 495 mm to clear
        # the mouth, which does not fit an episode at the insertion scale.
        scale=(0.004, 0.001, 0.001, 0.008, 0.008, 0.008),
        # Long enough for the capture to complete and preload the collar before
        # the policy is allowed to pull. The gate needed 1.0 s to settle.
        settling_time_s=1.0,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )


@configclass
class GraspEventsCfg(GrapplePinEventsCfg):
    """Start near the pin with the fingers open and let the policy close them."""

    close_gripper_on_reset = None
    hold_gripper_closed = None
    open_gripper_on_reset = EventTerm(
        func=mdp.reset_grapple_fingers,
        mode="reset",
        params={"asset_cfg": GRIPPER_CFG, "finger_joint": GRAPPLE_GRIPPER_APPROACH[0]},
    )
    remember_blade_pose = EventTerm(func=mdp.record_blade_reset_pose, mode="reset")
    reset_grapple = EventTerm(func=mdp.reset_grapple_progress, mode="reset")


@configclass
class GraspRewardsCfg:
    approach = RewTerm(func=mdp.capture_approach_reward, weight=10.0)
    success = RewTerm(func=mdp.capture_success_reward, weight=30.0)
    disturbance = RewTerm(func=mdp.blade_disturbance_penalty, weight=-0.20)
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.10)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    failure = RewTerm(func=mdp.capture_failure_reward, weight=-15.0)


@configclass
class GraspTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    capture_success = DoneTerm(func=mdp.capture_success_mask)
    capture_failed = DoneTerm(func=mdp.capture_failure)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class GraspCurriculumCfg:
    pose_noise = CurrTerm(
        func=mdp.InsertionSuccessRateCurriculum,
        params={
            "success_term": "capture_success",
            "threshold": 0.80,
            "window_size": 2_000,
            "max_stage": 2,
            "minimum_level_steps": 1_600,
            "stage_mixtures": INSERTION_CURRICULUM_MIXTURES,
        },
    )


@configclass
class ZeroGBladeGrapplePinGraspEnvCfg(ZeroGBladeGrapplePinCaptureEnvCfg):
    """Learn to align on the pin and close a loaded capture.

    The reset pose is the calibrated head-on capture pose carrying much larger
    joint noise than the insertion tasks use, so the policy has to servo back
    onto an 8.5 mm-per-side aperture rather than start inside it. A true
    standoff approach, backing off along the tool axis and flying in, needs a
    second calibrated pose per stage and is deliberately left for after this
    skill is certified.
    """

    observations: GraspObservationsCfg = GraspObservationsCfg()
    actions: GraspActionsCfg = GraspActionsCfg()
    events: GraspEventsCfg = GraspEventsCfg()
    rewards: GraspRewardsCfg = GraspRewardsCfg()
    terminations: GraspTerminationsCfg = GraspTerminationsCfg()
    curriculum: GraspCurriculumCfg = GraspCurriculumCfg()
    episode_length_s: float = 6.0

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        self.events = GraspEventsCfg()
        # Displacing the tool by roughly 10 to 60 mm is what makes this a grasp
        # skill instead of a decision about when to close.
        self.events.reset_arm.params["noise_by_stage"] = (0.010, 0.025, 0.050)
        self.events.reset_blade.params["poses_by_stage"] = CONTACT_INSERTION_STAGE_BLADE_POSE
        if level < 2:
            self.events.blade_mass = None
        if level < 3:
            self.events.slot_material = None
            self.events.left_guide_material = None
            self.events.right_guide_material = None
            self.events.randomize_stiction = None
            self.events.rail_stiction_force = None
        self.scene.robot = make_grapple_pin_robot_cfg(floating=level >= 4)
        if level < 4:
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None


@configclass
class ExtractEventsCfg(GrapplePinEventsCfg):
    """Take the pin for real at the start of the episode, then act.

    "Already captured" cannot be faked by writing the fingers to their seated
    angle. That places the pads around the wedge without pressing the pin
    against the collar, and the first pull then travels the whole seating gap
    before anything takes load: measured, a 40 mm pull moved the blade 0.1 mm
    and opened a 12.9 mm grip error.

    So the episode opens with the fingers apart and the same two-stage capture
    the pull gate measured 69 N on, while the action term holds the arm still.
    The policy takes over with a preloaded grip.
    """

    close_gripper_on_reset = EventTerm(
        func=mdp.reset_grapple_fingers,
        mode="reset",
        params={"asset_cfg": GRIPPER_CFG, "finger_joint": GRAPPLE_GRIPPER_APPROACH[0]},
    )
    hold_gripper_closed = EventTerm(
        func=mdp.hold_two_stage_grip,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={
            "asset_cfg": GRIPPER_CFG,
            "capture_positions": GRAPPLE_GRIPPER_CAPTURE,
            "hold_positions": GRAPPLE_GRIPPER_HOLD,
        },
    )
    remember_blade_pose = EventTerm(func=mdp.record_blade_reset_pose, mode="reset")
    reset_grapple = EventTerm(func=mdp.reset_grapple_progress, mode="reset")


@configclass
class ExtractRewardsCfg:
    progress = RewTerm(func=mdp.extraction_progress_reward, weight=12.0)
    success = RewTerm(func=mdp.extraction_success_reward, weight=30.0)
    retention = RewTerm(func=mdp.grip_retention_penalty, weight=-0.50)
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.10)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    failure = RewTerm(func=mdp.extraction_failure_reward, weight=-15.0)


@configclass
class ExtractTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    extraction_success = DoneTerm(func=mdp.extraction_success_mask)
    extraction_failed = DoneTerm(func=mdp.extraction_failure)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class ExtractCurriculumCfg:
    pose_noise = CurrTerm(
        func=mdp.InsertionSuccessRateCurriculum,
        params={
            "success_term": "extraction_success",
            "threshold": 0.80,
            "window_size": 2_000,
            "max_stage": 2,
            "minimum_level_steps": 1_600,
            "stage_mixtures": INSERTION_CURRICULUM_MIXTURES,
        },
    )


@configclass
class ZeroGBladeGrapplePinExtractEnvCfg(ZeroGBladeGrapplePinCaptureEnvCfg):
    """Break the blade free and pull it fully clear of the slot mouth.

    Success is the blade's rear face passing x = 0.45, which is 495 mm of
    travel from fully inserted. That is the only definition under which the
    module has actually been removed, and it was chosen with the owner over the
    cheaper option of retreating to the insertion staging pose.

    The end of that pull puts the wrist about 200 mm in front of the robot's own
    base with the tool still pointing along +x. That is inside the UR10e's
    reach but folded, and it has not been checked kinematically; the first smoke
    run should confirm the arm can get there before a long run is started.
    """

    observations: ExtractObservationsCfg = ExtractObservationsCfg()
    actions: ExtractActionsCfg = ExtractActionsCfg()
    events: ExtractEventsCfg = ExtractEventsCfg()
    rewards: ExtractRewardsCfg = ExtractRewardsCfg()
    terminations: ExtractTerminationsCfg = ExtractTerminationsCfg()
    curriculum: ExtractCurriculumCfg = ExtractCurriculumCfg()
    episode_length_s: float = 15.0

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        self.events = ExtractEventsCfg()
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
        self.scene.robot = make_grapple_pin_robot_cfg(floating=level >= 4)
        if level < 4:
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None


@configclass
class InsertRewardsCfg:
    progress = RewTerm(func=mdp.insertion_progress_reward, weight=12.0)
    success = RewTerm(func=mdp.insertion_success_reward, weight=30.0)
    retention = RewTerm(func=mdp.grip_retention_penalty, weight=-0.50)
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.10)
    misalignment = RewTerm(func=mdp.insertion_misalignment_penalty, weight=-0.03)
    settling = RewTerm(func=mdp.insertion_settling_penalty, weight=-0.04)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    failure = RewTerm(func=mdp.extraction_failure_reward, weight=-15.0)


@configclass
class InsertTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insertion_success = DoneTerm(func=mdp.insertion_success_mask)
    # Reuses extraction's predicate on purpose: it is the one that asks whether
    # a *physical* grip is still there, rather than measuring a fixed joint
    # against the frame that fixed joint defines.
    extraction_failed = DoneTerm(func=mdp.extraction_failure)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class ZeroGBladeGrapplePinInsertEnvCfg(ZeroGBladeGrapplePinCaptureEnvCfg):
    """Insert a blade the gripper is physically holding, with no fixed joint.

    This is the promoted insertion task's problem without its central
    abstraction. The three certified policies carry the blade on a PhysX fixed
    joint; here the only thing between the tool and the blade is pad against
    pin, so the grip has to survive every rail contact the insertion generates.

    It starts at the certified full-distance staging pose rather than at the end
    of an extraction, because that is the arm pose that has been calibrated.
    Starting from fully clear needs one more calibration run.
    """

    observations: InsertObservationsCfg = InsertObservationsCfg()
    events: ExtractEventsCfg = ExtractEventsCfg()
    rewards: InsertRewardsCfg = InsertRewardsCfg()
    terminations: InsertTerminationsCfg = InsertTerminationsCfg()
    episode_length_s: float = 12.0

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        self.events = ExtractEventsCfg()
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
        self.scene.robot = make_grapple_pin_robot_cfg(floating=level >= 4)
        if level < 4:
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None


@configclass
class ZeroGBladeGrapplePinGraspPlayEnvCfg(ZeroGBladeGrapplePinGraspEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeGrapplePinExtractPlayEnvCfg(ZeroGBladeGrapplePinExtractEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeGrapplePinInsertPlayEnvCfg(ZeroGBladeGrapplePinInsertEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "GRAPPLE_FINGER_SEATED_RAD",
    "GRAPPLE_GRIPPER_APPROACH",
    "GRAPPLE_GRIPPER_CAPTURE",
    "GRAPPLE_GRIPPER_HOLD",
    "ExtractActionsCfg",
    "ExtractEventsCfg",
    "GraspActionsCfg",
    "GraspEventsCfg",
    "GrapplePinActionsCfg",
    "GrapplePinEventsCfg",
    "ZeroGBladeGrapplePinCaptureEnvCfg",
    "ZeroGBladeGrapplePinCapturePlayEnvCfg",
    "ZeroGBladeGrapplePinExtractEnvCfg",
    "ZeroGBladeGrapplePinExtractPlayEnvCfg",
    "ZeroGBladeGrapplePinGraspEnvCfg",
    "ZeroGBladeGrapplePinGraspPlayEnvCfg",
    "ZeroGBladeGrapplePinInsertEnvCfg",
    "ZeroGBladeGrapplePinInsertPlayEnvCfg",
]
