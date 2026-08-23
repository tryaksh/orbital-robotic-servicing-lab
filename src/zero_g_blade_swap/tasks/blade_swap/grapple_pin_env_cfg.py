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

from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
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
from .assets import (
    CONTACT_INSERTION_STAGE_BLADE_POSE,
    GRAPPLE_HEAD_ON_ARM_JOINT_POS,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_TOOL_OFFSET_POS,
    SECOND_SLOT_ENTRY_LOWER_RAMP_CFG,
    SECOND_SLOT_ENTRY_UPPER_RAMP_CFG,
    CompliantD6JointCfg,
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
    # Runs at the control rate, like the grip hold. Inert until a capture
    # qualifies, and inherited by all three skills so none of them can be
    # trained against a different interface than the others.
    grapple_latch = EventTerm(
        func=mdp.GrappleLatch,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg("spare_blade"),
            "rated_torque_nm": 5.0,
            "rotation_stiffness": 10.0,
            "rotation_damping_ratio": 0.9,
            # Zero preserves the original torque-only latch experiment. A
            # workflow may explicitly rate the form-locking translational load
            # path before arming it after rail release.
            "rated_force_n": 0.0,
            "position_stiffness": 2_500.0,
            "position_damping_ratio": 0.9,
            "joint_mode": "compliant",
            "require_armed": False,
        },
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
    # A modelled latch was built next, on the reasoning that flight servicing
    # hardware latches rather than relying on friction, and it is off for a
    # measured reason of its own. Swept from 10 to 160 N-m against the unchanged
    # extract v4 policy, it never moved the rotation it was aimed at -- the
    # transverse component sat at 0.293 to 0.299 rad across the whole range --
    # while extraction travel collapsed from 465 mm to about 25 mm, because a
    # restoring torque on a module the rails still hold jams it in the rails.
    #
    # It stays implemented because the sweep is evidence and because a latch
    # applied *after* the module is free is a different experiment that has not
    # been run. See ``mdp.GrappleLatch`` and docs/status.md.
    latch_enabled: bool = False
    latch_rated_torque_nm: float = 5.0
    latch_rated_force_n: float = 0.0
    latch_position_stiffness_n_per_m: float = 2_500.0
    latch_position_damping_ratio: float = 0.9
    latch_rotation_stiffness_nm_per_rad: float = 10.0
    latch_rotation_damping_ratio: float = 0.9
    latch_joint_mode: str = "compliant"
    #: Load the *mating* compliance is allowed to apply. Separate from
    #: ``latch_rated_*`` because the two states of this mechanism are rated for
    #: different things: the weld has to survive the gripper's own wedge
    #: preload, thousands of newtons, while the spring has to be gentle enough
    #: that a lead-in plate can still push the module off the tool's line.
    mating_force_cap_n: float = 400.0
    #: Translational stiffness of the mating compliance. Soft enough that the
    #: rack's lead-in can push the module off the tool's line, which is the only
    #: way a lead-in aligns anything.
    mating_translation_stiffness_n_per_m: float = 40_000.0
    #: Angular stiffness of the mating compliance, about the module's tip.
    mating_rotation_stiffness_nm_per_rad: float = 200.0
    mating_torque_cap_nm: float = 40.0
    base_rail_enabled: bool = False
    #: Engage the latch only once a driver says the module is free of the rails,
    #: instead of the instant a capture qualifies. The sweep that refuted the
    #: latch engaged it while the module was still railed, which is exactly where
    #: a restoring torque jams it; the transit is where the lock is needed and it
    #: has never been tested there. See ``mdp.GrappleLatch``.
    latch_engages_on_release: bool = False
    #: Per-side clearance added to the destination bay's channel, in metres.
    #:
    #: **Zero, which is the production channel, and that is deliberate.** A
    #: rigidly delivered module needs ``L * theta / 2`` of clearance because it
    #: cannot be straightened by the channel it is entering -- 3.15 mm on this
    #: module at the 0.014 rad the arm delivers. Widening the rack to that is a
    #: real answer and the wrong one to reach for first: the module can instead
    #: be delivered *compliantly*, and then the lead-in aligns it exactly as
    #: section 6 of the interface specification measures it doing. Running with
    #: the relief on by default would let the compliance take credit for the
    #: relief's work, so the default is the rack as built and the relief is
    #: something a run has to ask for.
    service_destination_channel_relief_m: float = 0.0

    def _configure_latch(self) -> None:
        """Apply the latch settings to whichever event set is currently installed.

        ``configure_robustness`` rebuilds the event configuration and each skill
        rebuilds it again afterwards, so this is called from every one of them
        rather than written once.
        """

        if not self.latch_enabled:
            self.events.grapple_latch = None
            return
        self.events.grapple_latch.params["rated_torque_nm"] = self.latch_rated_torque_nm
        self.events.grapple_latch.params["rated_force_n"] = self.latch_rated_force_n
        self.events.grapple_latch.params["position_stiffness"] = self.latch_position_stiffness_n_per_m
        self.events.grapple_latch.params["position_damping_ratio"] = self.latch_position_damping_ratio
        self.events.grapple_latch.params["rotation_stiffness"] = self.latch_rotation_stiffness_nm_per_rad
        self.events.grapple_latch.params["rotation_damping_ratio"] = self.latch_rotation_damping_ratio
        self.events.grapple_latch.params["joint_mode"] = self.latch_joint_mode
        self.events.grapple_latch.params["mating_force_cap_n"] = self.mating_force_cap_n
        # The same number has to reach the *joint*, or the cap that was measured
        # on the wrench silently stops applying when the mechanism became a
        # joint -- which is exactly what happened: the drive pushed at its own
        # 20 kN default and wedged the module where 400 N walks it in.
        mating_joint = getattr(self.scene, "mating_compliance_joint", None)
        if mating_joint is not None:
            mating_joint.spawn.max_force = self.mating_force_cap_n
            mating_joint.spawn.translation_stiffness = self.mating_translation_stiffness_n_per_m
            mating_joint.spawn.translation_damping = 2.0 * 0.9 * (
                self.mating_translation_stiffness_n_per_m * 10.0
            ) ** 0.5
            # **Soft in rotation, now that the centre is at the tip.** With the
            # centre at the wrist a soft angular gain let the module flop, so
            # the first fix was to stiffen it -- and a stiff one cannot be
            # reoriented by the channel it is entering, which is the other half
            # of the jam. A remote centre at the part's own tip is what makes
            # softness safe: a lateral force there translates the module, and a
            # moment there rotates it about the point that is touching. That is
            # the property the device is named for.
            mating_joint.spawn.rotation_stiffness = self.mating_rotation_stiffness_nm_per_rad
            mating_joint.spawn.rotation_damping = 2.0 * 0.9 * (
                self.mating_rotation_stiffness_nm_per_rad * 0.2
            ) ** 0.5
        self.events.grapple_latch.params["mating_torque_cap_nm"] = self.mating_torque_cap_nm
        self.events.grapple_latch.params["require_armed"] = self.latch_engages_on_release

    def configure_service_destination(self) -> None:
        """Install the destination bay's vertical lead-in.

        **A rack requirement, produced by a manipulator measurement.** Section 6
        of ``docs/service_interface_spec.md`` establishes that the lateral flares
        do not assist the insertion, they perform it: removed, two fully trained
        policies insert nothing, at any displacement. That result is about the
        lateral axis, and nothing in this project had ever asked about the
        vertical one, because both insertion skills *reset* with the module
        already inside the channel and so never entered the mouth from outside.

        A relocation does enter from outside, and a six-axis arm carrying a
        450 mm module through free space delivers it to the mouth with a
        measured 0.066 rad of attitude error. That swings the module's leading
        corner 14.7 mm off the channel's centre plane, against the 0.5 mm per
        side the floor plate and the upper lips leave. The module cannot enter,
        and no controller gain fixes a geometric interference.

        So the destination bay gets the same lead-in on the vertical axis it
        already has on the lateral one: the same 80 mm plate at the same 12
        degrees on the same low-friction surface, giving the same 16.6 mm per
        side of catch, which accepts 0.074 rad. Only the destination bay, and
        only when the workflow is carrying the module itself.
        """

        self.scene.blade_slot_two_entry_upper_ramp = SECOND_SLOT_ENTRY_UPPER_RAMP_CFG
        self.scene.blade_slot_two_entry_lower_ramp = SECOND_SLOT_ENTRY_LOWER_RAMP_CFG

        # **And the channel behind it may have to be wider, for a reason that
        # is a general rule rather than a fudge -- but only for a module that
        # arrives rigid.**
        #
        # A module delivered by a manipulator arrives with that manipulator's
        # attitude accuracy, and if it is *rigidly* held it cannot be
        # straightened by the channel it is entering: a lead-in works by pushing
        # a module square, and a module bolted to a wrist will not be pushed.
        # A straight channel of clearance c admits a rigid module of length L at
        # a tilt of at most 2c/L, so the rack requirement is
        #
        #     c >= L * theta / 2
        #
        # with theta the delivered attitude accuracy. Measured on this arm the
        # squaring leg converges to between 0.013 and 0.066 rad depending on how
        # much of the workcell's reach boundary it is standing in; at the worst
        # of that, a 0.45 m module needs 14.8 mm per side and at the best 2.9 mm.
        #
        # The relief below is sized for the measured median rather than the
        # worst case, and the run reports what it actually delivered, so a
        # workflow that arrives outside it fails its seating check rather than
        # being quietly accommodated. Alternatives a designer has, in order of
        # preference: hold the module compliantly for the last 10 mm (this
        # project cannot -- the pads do not resist lateral load, measured), move
        # the arm out of its reach boundary (section 6a), or widen the channel.
        relief = self.service_destination_channel_relief_m
        if relief <= 0.0:
            return
        left_guide = list(self.scene.blade_slot_two_left_guide.init_state.pos)
        right_guide = list(self.scene.blade_slot_two_right_guide.init_state.pos)
        left_guide[1] += relief
        right_guide[1] -= relief
        self.scene.blade_slot_two_left_guide.init_state.pos = tuple(left_guide)
        self.scene.blade_slot_two_right_guide.init_state.pos = tuple(right_guide)
        floor = list(self.scene.blade_slot_two.init_state.pos)
        floor[2] -= relief
        self.scene.blade_slot_two.init_state.pos = tuple(floor)
        for lip in (
            self.scene.blade_slot_two_upper_left_lip,
            self.scene.blade_slot_two_upper_right_lip,
        ):
            position = list(lip.init_state.pos)
            position[2] += relief
            lip.init_state.pos = tuple(position)
        # **The lead-ins stay where they are, and that is measured rather than
        # tidy.** The obvious rule is that a lead-in continues a channel surface,
        # so moving the surface should move the lead-in -- the ramps are authored
        # from ``SLOT_LIP_BOTTOM_Z`` and ``SLOT_FLOOR_TOP_Z``, and section 6
        # places each flare so its inner face meets the rail face exactly at the
        # mouth. Built that way and measured, the module stops dead on the mouth
        # plane at 0.2249 m: the lead-ins at the *nominal* surfaces are what
        # squares a module the arm delivers 67 mrad off, and a lead-in moved out
        # with the relief stops touching it in time to do that. Left in place
        # they present a narrower throat and a longer correcting run, and the
        # module goes in. Recorded here because it is the opposite of what the
        # geometry suggests.

    def configure_base_rail(self) -> None:
        """Install the physical payload shuttle after task setup.

        Every grapple skill overrides ``configure_robustness`` and several of
        those overrides replace ``scene.robot`` after calling ``super``.  A
        rail selected before that chain therefore used to be silently replaced
        by the fixed-root robot, leaving the moving anchor as an unattached
        marker.  This method is deliberately called *after* the complete task
        robustness configuration so its physical topology is the final one.
        """

        if not self.base_rail_enabled:
            raise ValueError("configure_base_rail() requires base_rail_enabled=True")
        # The payload stage is a procedurally authored D6 joint.  PhysX scene
        # replication only copied the first joint, so parallel qualification
        # environments had a stage prim but no usable joint.  Author each
        # environment independently, as the camera workflow already does.
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
        # The arm remains fixed to its workcell after it hands the extracted
        # ORU to the shuttle.  Moving the entire six-axis articulation made its
        # finite-effort joints counteract the base drive and coupled pose axes.
        self.scene.robot = make_grapple_pin_robot_cfg(floating=False)
        self.scene.base_compliance = None
        # Design-for-serviceability destination mouth.  The original funnel
        # converged to the same 0.75 mm per-side clearance as the straight
        # production rails.  A carried module aligned to 10 micrometres and
        # 0.4 milliradians repeatedly stopped on its front contact plane, even
        # after reducing valid PhysX contact envelopes.  The metrology-guided
        # bay therefore omits those two colliders and enters the unchanged
        # 1.5 mm straight channel directly.  Their visuals stay in the model so
        # the design comparison remains obvious; they no longer pretend to be
        # a useful passive funnel for a precision shuttle.
        self.scene.blade_slot_two_entry_left_flare.spawn.collision_props.collision_enabled = False
        self.scene.blade_slot_two_entry_right_flare.spawn.collision_props.collision_enabled = False
        # Open the destination's straight rails from 1.5 mm to 4.5 mm total
        # clearance.  The unmodified guide front faces produced the same exact
        # x=0.225 m stop after the flare colliders were removed, identifying the
        # remaining butt-contact.  A 2.25 mm-per-side key is still a tight
        # service interface, but it has resolvable manufacturing/physics margin
        # and matches the 2.5 mm lateral acceptance envelope.
        rail_relief_m = 0.0015
        left_guide = list(self.scene.blade_slot_two_left_guide.init_state.pos)
        right_guide = list(self.scene.blade_slot_two_right_guide.init_state.pos)
        left_guide[1] += rail_relief_m
        right_guide[1] -= rail_relief_m
        self.scene.blade_slot_two_left_guide.init_state.pos = tuple(left_guide)
        self.scene.blade_slot_two_right_guide.init_state.pos = tuple(right_guide)
        # The old floor top was exactly coincident with the module's lower
        # face.  Its front vertical face therefore formed a second x=0.225 m
        # butt contact even after the side rails were relieved.  Lowering only
        # the destination floor by 2 mm provides a real lead-in clearance; the
        # upper lips and side rails still capture all remaining five motions.
        destination_floor = list(self.scene.blade_slot_two.init_state.pos)
        destination_floor[2] -= 0.002
        self.scene.blade_slot_two.init_state.pos = tuple(destination_floor)
        self.scene.payload_stage = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/PayloadStage",
            # A disabled D6 joint is armed at the measured extraction pose. It
            # then carries the module directly, like a service caddy/hexapod,
            # while the robot opens and clears the work envelope.
            spawn=CompliantD6JointCfg(
                body1_relative_path="SpareBlade",
                relocate_robot_articulation_root=False,
                enabled=False,
                translation_limit=0.8,
                translation_x_lower_limit=-0.200,
                translation_x_upper_limit=0.800,
                translation_y_lower_limit=-0.600,
                translation_y_upper_limit=0.200,
                translation_z_lower_limit=-0.400,
                translation_z_upper_limit=0.400,
                rotation_limit_deg=20.0,
                # This is a positioning stage, not a compliant suspension.
                # The earlier 10 kN/m drive deflected 71 mm when the loaded arm
                # replayed its extraction path, carrying the module outside the
                # receiving channel even though the commanded target was exact.
                # These near-critical gains keep the carriage physical while
                # giving it the authority a machine-tool axis requires.
                translation_stiffness=200_000.0,
                translation_damping=10_000.0,
                rotation_stiffness=500_000.0,
                rotation_damping=15_000.0,
                max_force=100_000.0,
            ),
        )

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # The parent rebuilds the event set for the chosen level from the
        # contact task's class, which carries the top-down poses and the old
        # finger commands, so re-assert the head-on ones afterwards.
        self.events = GrapplePinEventsCfg()
        self.events.reset_arm.params["noise_by_stage"] = (0.0005, 0.001, 0.002) if level == 0 else (0.001, 0.002, 0.004)
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
        # This skill rebuilt the event set above, so re-apply the latch to it.
        self._configure_latch()


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
        # 240 mm/s along the pull axis. The blade has to travel 495 mm to clear
        # the mouth, which is three and a half times the certified insertion
        # distance, and at the insertion scale that is 124 consecutive steps of
        # near-maximum command before the first reward for finishing.
        #
        # Measured at 120 mm/s and 700 epochs: the grip held perfectly, 6.7 mm
        # of error at the median, and the blade still travelled only 71 mm. The
        # training reward was climbing monotonically the whole time and had not
        # levelled off, so that is a horizon too long to credit-assign across,
        # not a policy that cannot hold the pin. Halving the step count is the
        # cheaper half of the fix; the other half is simply more epochs.
        # Rebalanced 2026-08-15, and this is the measurement that forced it.
        #
        # The old scales were inherited from the insertion task, where the module
        # is inside its rails and lateral motion *must* be tiny. Extraction is
        # the opposite problem: the module ends completely unconstrained and has
        # to be steered. The inherited numbers gave the tool 0.24 m/s along the
        # pull axis and 0.03 m/s across it, an 8:1 asymmetry, and 0.24 rad/s of
        # rotation.
        #
        # Measured on extract v7 over 9,002 held-out episodes, the *module*
        # rotates at 0.296 rad/s at p95 and 0.767 rad/s at maximum. The wrist
        # could not follow it even in principle, so the grip attitude ran away to
        # the 0.350 rad limit in 99% of failures and no reward weighting could
        # have fixed it: the policy was being asked to track something faster
        # than its action space could move.
        #
        # Lateral goes to 0.12 m/s, still half the axial rate, and rotation to
        # 0.60 rad/s, comfortably above the module's observed maximum.
        scale=(0.008, 0.004, 0.004, 0.020, 0.020, 0.020),
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
    # Raised fivefold with the free band widened past the seating feed. The
    # first certified grasp lost 188 and 211 episodes a stage to capture_failed
    # against 1 and 6 timeouts, so the policy was reaching the pin and shoving
    # it, not failing to arrive: the binding cost is disturbance, not search.
    disturbance = RewTerm(func=mdp.blade_disturbance_penalty, weight=-1.0)
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

    Episode length is generous relative to the work: measured captures complete
    in a couple of seconds once the approach is solved, and the failure mode is
    never converging rather than running out of time, so a longer episode buys
    nothing and a shorter one would hide the difference.
    """

    observations: GraspObservationsCfg = GraspObservationsCfg()
    actions: GraspActionsCfg = GraspActionsCfg()
    events: GraspEventsCfg = GraspEventsCfg()
    rewards: GraspRewardsCfg = GraspRewardsCfg()
    terminations: GraspTerminationsCfg = GraspTerminationsCfg()
    curriculum: GraspCurriculumCfg = GraspCurriculumCfg()
    # 6 s was tuned when a capture only had to reach 20 mm of grip error.
    # Aligning the skill with what the chain actually waits for -- 10 mm --
    # doubles the precision demanded of the same motion, and the chain says the
    # old budget was already the binding constraint: 77 of its 113 failures were
    # captures overrunning 6 s while the skill scored 96.10% alone. Both move
    # together because they are one defect measured two ways.
    episode_length_s: float = 10.0

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        self.events = GraspEventsCfg()
        # The reset has to put the tool *outside* the 20 mm capture tolerance,
        # or the skill is not a grasp.
        #
        # The first schedule here was (0.010, 0.025, 0.050) rad, and it measured
        # 99.3% at stage 1 for the wrong reason: successful captures completed
        # at a median of 0.30 s and a 95th percentile of 0.77 s, which is the
        # first few control steps. The reset was landing the tool 13.8 mm from
        # the grip point, already inside the tolerance, so the policy only had
        # to decide to close. Stage 2 then measured 35% because it was the only
        # stage that required an approach at all, and nothing in the earlier
        # stages had taught one.
        #
        # A skill certified on the first schedule would have been an expensive
        # way of reporting that a reset works.
        self.events.reset_arm.params["noise_by_stage"] = (0.030, 0.055, 0.085)
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
        # This skill rebuilt the event set above, so re-apply the latch to it.
        self._configure_latch()


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
    # Declared last so it runs after reset_arm and reset_blade and overwrites
    # both. It is a separate term rather than a replacement for reset_arm
    # because the parent's configure_robustness writes
    # reset_arm.params["noise_by_stage"] and would KeyError on a term that does
    # not carry it. Left None except where a task opts in.
    reset_handoff: EventTerm | None = None
    remember_blade_pose = EventTerm(func=mdp.record_blade_reset_pose, mode="reset")
    reset_grapple = EventTerm(func=mdp.reset_grapple_progress, mode="reset")


@configclass
class ExtractRewardsCfg:
    progress = RewTerm(func=mdp.extraction_progress_reward, weight=12.0)
    success = RewTerm(func=mdp.extraction_success_reward, weight=30.0)
    # Attitude is charged far harder here than the shared default, and the
    # default is left alone because insert v6 was certified under it.
    #
    # Extract v6 certifies at 10.09% and 7,971 of its 8,093 failures end at the
    # 0.350 rad grip-attitude limit with grip *position* still holding at
    # 12.5 mm. Under the defaults, attitude at the 0.20 rad success limit costs
    # about 0.16 per step against a progress term weighted 12, so the policy was
    # being paid to trade attitude for travel. It is not a control failure that
    # more training fixes; it is the objective.
    #
    # Free below 0.04 rad, normalised over 0.06, weighted 1.0: about 3.6 per
    # step at the success limit and 9.4 at 0.30, which is the same order as
    # progress rather than two orders below it.
    # The clamp is raised because the default was switching this term off exactly
    # where it was needed. Measured on extract v11settle: grip attitude sits at
    # 0.3538 rad at the median, and with these parameters the penalty saturates
    # at its 25.0 ceiling from about 0.325 rad. Above that there is **no
    # gradient** -- 0.35 rad and 0.50 rad cost the same -- so the policy was free
    # to give attitude away once past the knee, and it parked just beyond it.
    #
    # 60 keeps the term growing across the whole range an episode can reach: the
    # extraction-failure limit is 0.35 rad, where the raw cost is 29.0, so
    # nothing an episode can visit is saturated any more. It is not larger than
    # that, because extract v7 showed an over-weighted attitude term makes
    # standing still cheaper than pulling and cost the removal chain 11 points.
    #
    # Insertion keeps the 25.0 default along with the rest of its defaults,
    # because insert v6's certification was produced under them.
    retention = RewTerm(
        func=mdp.grip_retention_penalty,
        weight=-0.50,
        params={
            "free_rad": 0.04,
            "orientation_scale": 0.06,
            "orientation_weight": 1.0,
            "max_penalty": 60.0,
        },
    )
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.10)
    # Nothing paid this policy to arrive settled. The success predicate asks for
    # a module that is clear *and* under the derived velocity limits, and every
    # dense term here was about travel. Measured: extract v10 trained to a higher
    # reward than v8 -- 158.7 against 148.4 -- and certified at 0.00%, losing the
    # grip in 8,988 of 9,010 episodes by pulling through the line at speed.
    #
    # Sized against that measurement. At v8's terminal 0.0710 m/s, five times the
    # 0.0143 limit, this charges 18 per step inside the last 60 mm. Progress is
    # potential-based, so covering that 60 mm pays the same total however fast it
    # is crossed and the only thing speed buys is the 0.10 per step of
    # elapsed_time_penalty saved. Decisive against that, and far below the
    # one-off success term.
    settling = RewTerm(func=mdp.extraction_settling_penalty, weight=-2.0)
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
    # 15 s was never enough and the evidence for that is unusually clean.
    # Certified on 15 s, extract v4's median cycle time is 15.000 s -- *every*
    # episode ran out the clock -- while the module reached 458 mm of the
    # required 495. This is the insert v5 situation exactly, where 12 s made a
    # skill look unreliable that was merely slow, and lengthening it to 20 s
    # with a fine-tune took it from 6.96% to 95.57%.
    #
    # The clock alone is not the fix, and that was measured before changing it:
    # replaying v4 unchanged at 25 s and 40 s converts 449 timeouts into 512
    # lost grips at a hard 478 mm ceiling, because a policy asked to work past
    # its trained horizon degrades rather than continues. The clock has to move
    # *and* the policy has to be fine-tuned against it, which is what insert v6
    # did.
    #
    # The chained workflow reads this field for its extract-phase budget, so the
    # skill and the chain cannot disagree about how long a pull is allowed.
    episode_length_s: float = 25.0

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        self.events = ExtractEventsCfg()
        # Wide enough to cover the states a capture hands over, for the same
        # reason as the insert task; see its configure_robustness. Measured on
        # the chained run: with the old 0.0005 rad reset the hand-off grip error
        # matched this task's own to within a millimetre and the policy still
        # reversed into the rack on all eight seeds tried, because the arm's
        # joint configuration was outside anything it had trained on.
        #
        # Doubled again on 2026-08-15, because (0.010, 0.015, 0.020) was still
        # too narrow and the chain says so precisely. Extract v7 certifies at
        # 28.48% alone with a 15.2 s median, and chained it overruns its own
        # 25 s budget in 138 of 192 removals while the *module's* orientation
        # error reaches 0.85 rad against 0.12 rad in isolation. A skill that
        # tumbles the payload only when its predecessor hands it over is a skill
        # trained on the wrong initial states, not a skill that cannot pull.
        # Halved on 2026-08-16, because the wider half was not measuring the
        # skill. The reset writes a noisy arm pose and then places the module at
        # a FIXED pose, and the scripted two-stage capture closes wherever the
        # arm happens to be -- so past a point the pads close on nothing and the
        # episode is dead on control step 1, before the policy acts at all.
        #
        # Measured on extract v13unsat at stage 2, sweeping the noise scale:
        #
        #   0.040 rad   57.3% dead at step 1   37.28% success
        #   0.020 rad    0.2% dead at step 1   99.02% success
        #   0.010 rad    0.0% dead at step 1   99.41% success
        #
        # This is a strict *subset* of the distribution the policy trained on, so
        # nothing is retrained and no policy sees a state it has not seen. What
        # changes is that the certification stops counting unwinnable resets as
        # skill failures.
        #
        # It does not weaken the robustness claim, because the robustness claim
        # does not rest here: the chained removal certifies at 98.78% while the
        # capture hands extraction an arm pose 0.157 rad from nominal on its
        # worst axis, which is four times this reset's widest draw. The chain is
        # the evidence for tolerating a real hand-off; this reset only has to
        # produce a grip for the skill to be measured from.
        self.events.reset_arm.params["noise_by_stage"] = (0.010, 0.015, 0.020)
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
        # This skill rebuilt the event set above, so re-apply the latch to it.
        self._configure_latch()


@configclass
class InsertRewardsCfg:
    progress = RewTerm(func=mdp.insertion_progress_reward, weight=12.0)
    success = RewTerm(func=mdp.grapple_insertion_success_reward, weight=30.0)
    # **Extraction's attitude parameters, adopted here on a measurement.**
    #
    # These were the soft defaults -- free_rad 0.08, scale 0.15, weight 0.25 --
    # and the reason given for keeping them was explicit: they "must not change
    # for insertion, whose certification was produced under them". That
    # certification is insert v6 on the old workcell, which this branch replaces,
    # so the reason has expired and the defaults can be judged on their merits.
    #
    # They do not survive that. `grapple_insertion_conditions` refuses success
    # above **0.20 rad** of grip attitude, and the certified two-bay policy on the
    # old cell was passing that condition at p95 = 0.19447 -- 0.00554 rad of
    # margin, 2.8% of the limit. Under the soft defaults, 0.20 rad costs the
    # policy about 0.08 per step, which is nothing: the objective was almost
    # indifferent to the one quantity its success predicate turns on, and the
    # 0.188 rad it happened to reach was luck rather than optimisation.
    #
    # On the moved cell the same policy sits at p50 0.19994 -- about 6% higher,
    # and over the line. Every other condition passes in bay 2 at the median:
    # 5.0 mm axial of a 12 mm limit, 0.9 mm lateral of 2.5, 0.0029 rad of 0.0524.
    # Grip attitude is the whole difference between 98.60% and 10.50%.
    #
    # These are extraction's values, unchanged, and they were tuned against this
    # exact quantity for this exact reason: at 0.20 rad they cost about 3.6 per
    # step instead of 0.08, so the term finally has a gradient where the criterion
    # has an edge. The risk they carry is recorded too -- extract v7 showed an
    # over-weighted attitude term can make standing still cheaper than acting --
    # which is why the certification that follows is the judge, not the reward.
    retention = RewTerm(
        func=mdp.grip_retention_penalty,
        weight=-0.50,
        params={
            "free_rad": 0.04,
            "orientation_scale": 0.06,
            "orientation_weight": 1.0,
            "max_penalty": 60.0,
        },
    )
    time = RewTerm(func=mdp.elapsed_time_penalty, weight=-0.10)
    misalignment = RewTerm(func=mdp.insertion_misalignment_penalty, weight=-0.03)
    settling = RewTerm(func=mdp.insertion_settling_penalty, weight=-0.04)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    failure = RewTerm(func=mdp.extraction_failure_reward, weight=-15.0)


@configclass
class InsertTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Not the shared insertion predicate: its grasp check is written against the
    # top-down convention and reads a head-on capture as 2.1 rad of grasp error,
    # so it can never fire here. The first insert policy drove the blade to
    # 0.1 mm of the goal and still timed out in 1024 of 1024 episodes.
    insertion_success = DoneTerm(func=mdp.grapple_insertion_success_mask)
    # Reuses extraction's predicate on purpose: it is the one that asks whether
    # a *physical* grip is still there, rather than measuring a fixed joint
    # against the frame that fixed joint defines.
    extraction_failed = DoneTerm(func=mdp.extraction_failure)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class SingleStageCurriculumCfg:
    """One reset distance, so the stage mixture validator has nothing to ramp."""

    pose_noise = CurrTerm(
        func=mdp.InsertionSuccessRateCurriculum,
        params={
            "success_term": "insertion_success",
            "threshold": 0.80,
            "window_size": 2_000,
            "max_stage": 0,
            "minimum_level_steps": 1_600,
            "stage_mixtures": ((1.0,),),
        },
    )


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
    curriculum: SingleStageCurriculumCfg = SingleStageCurriculumCfg()
    # 167 mm at the 45 mm/s axial scale is 3.7 s of pure travel, and 12 s looked
    # like room to search without room to dawdle. Certification said otherwise:
    # successful insertions completed at a median of 11.70 s and a maximum of
    # 12.00 s, so every success was finishing on the buzzer, and 52% of failures
    # were still outside the 12 mm axial tolerance when the clock stopped. 20 s
    # was about 1.7x the measured time a success actually takes.
    #
    # **30 s, for the third time and the same reason, and this one was measured
    # on the chain rather than on the skill.** Handed the state a real capture
    # produces, insert v6 takes 12.3 s at the median but runs to about 18.5 s at
    # p95 -- into the buzzer again -- and 6 of every 7 chain-phase failures end
    # still short on axial depth rather than having gone wrong. Extending only
    # the clock, with the policy and every success tolerance untouched, moves it
    # from 95.31% to 97.92% on the reproduced hand-off, and two of the four
    # remaining failures are capture overruns rather than insertions at all.
    # The slowest insertion that succeeded under the longer clock took 24.3 s, so
    # 30 s covers the measured distribution with margin; a later session with
    # more evidence can tighten it toward that 24.3 s.
    #
    # This is a budget, not a success threshold. The predicate --- 12 mm axial,
    # 2.5 mm lateral, 0.0524 rad, both velocity limits and the grip --- is
    # untouched, which is the distinction rule 8 draws. What it costs is honest
    # and belongs next to the number: a servicing insertion is now allowed half a
    # minute, and the cycle time to quote is the measured one, not this bound.
    #
    # Insert v6's 95.57% describes the 20 s task and had to be re-run; see
    # docs/status.md and `grapple_insert_v6clock30_certification.json`.
    #
    # The chained workflow reads this field for its own insert-phase budget, so
    # the skill and the chain cannot disagree about how long an insertion is
    # allowed to take.
    episode_length_s: float = 30.0

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        self.events = ExtractEventsCfg()
        # Wide enough that a policy handed over from a capture is still inside
        # the distribution it trained on.
        #
        # This was (0.0005, 0.001, 0.002) rad, which is three hundredths of a
        # degree: the skill only ever saw one arm configuration. Certified alone
        # that is fine, and chained it is fatal, because a capture finishes
        # wherever the grasp policy's servoing puts it, never on the nominal
        # pose. Measured on the chained run, the hand-off grip error matches the
        # reset's to within a millimetre and the policy still reverses, which is
        # the joint configuration being out of distribution rather than the grip.
        # A skill that is going to be chained has to be trained across the states
        # its predecessor actually produces.
        self.events.reset_arm.params["noise_by_stage"] = (0.010, 0.015, 0.020)
        # Full distance only. Measured on the v4 checkpoint: the stage curriculum
        # never promoted past its first level, so the policy solved the 31 mm
        # near reset perfectly (500/500) and the 167 mm full reset barely at all
        # (7%, and 469 of 512 failures were timeouts rather than lost grips, so
        # the approach was never learned rather than the grip failing). A chained
        # workflow only ever asks for the full distance, so that is what it
        # trains on.
        self.events.reset_arm.params["poses_by_stage"] = (GRAPPLE_HEAD_ON_ARM_JOINT_POS[2],)
        self.events.reset_blade.params["poses_by_stage"] = (CONTACT_INSERTION_STAGE_BLADE_POSE[2],)
        self.events.reset_arm.params["noise_by_stage"] = (0.020,)
        # NOT enabled, and the measurement is why. Three resets were built to
        # reproduce the state the chain hands this skill, and a gate was run on
        # each *before* training: insert v6 unchanged must score near the ~80% it
        # already achieves in the chain's insert phase.
        #
        #   per-joint noise box                       0.00%
        #   measured arm poses, module left nominal  26.32%
        #   measured arm AND module poses, paired    47.17%
        #
        # Pairing the module with the arm recovered half the gap and did not
        # close it, so the hand-off is still not reproduced and no amount of PPO
        # would fix that. Training on the arm-only bank was tried anyway before
        # this gate existed and cost the chain 8 points.
        #
        # What is left is not an initial condition at all: the chained driver
        # latches the holding closure at hand-over (TwoStageRobotiqAction
        # .hold_latch) so the grip cannot relax, and no training task sets it, so
        # the module is also carried under a different gripper controller. That
        # is why the reset below is the *original* one -- insert v6's 95.57%
        # describes the task in this file, and it must keep doing so.
        #
        # This line of work is **closed**, and the thing that closed it is
        # `Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0`, which runs the real
        # capture inside the environment and reproduces the hand-off at 93.06%
        # against the best bank's 47.17%. Do not build a fifth reset.
        #
        # The whole path is deleted -- the 84 kB pose bank, its generator, and
        # `mdp.reset_from_handoff_bank` -- because nothing imported any of it and
        # keeping refuted machinery in the tree is how a future session spends a
        # night rediscovering that it does not work. The measurements are in
        # docs/status.md, which is where a result actually lives; the code was
        # only ever the way they were taken.
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
        # This skill rebuilt the event set above, so re-apply the latch to it.
        self._configure_latch()


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
