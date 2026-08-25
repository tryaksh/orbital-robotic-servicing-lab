"""A rack with two bays, and the insert skill that can seat a module in either.

A rack with one slot can only demonstrate remove-and-replace. Moving a module
from one bay to the neighbouring one is the ORU changeout a servicer actually
performs, and it is what the relocation demonstration needs.

Two things make this a separate registration rather than an edit, and both are
the discipline that has kept every promoted result in this project alive:

* the single-slot tasks are untouched, so the certifications that describe them
  keep describing them;
* the second bay is the certified one *displaced*, part for part, rather than
  re-authored, so the two cannot drift apart and any difference between them is
  attributable to position alone.

The insert skill trains on both bays at once rather than on the new one alone.
The gate is "insert >= 95% on both slots", and a policy that learns the second
bay by forgetting the first has not done the job -- so the reset draws evenly
from the two and the certification reports the worse of them rather than a pool.

Which bay an episode is about is the curriculum stage, and the arm reset pose,
the module reset pose and the insertion goal all read that same index. That is
deliberate: three separate ways of selecting a slot is three chances for them to
disagree about which one the episode is scored against.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp
from .assets import (
    CONTACT_INSERTION_STAGE_BLADE_POSE,
    FIRST_SLOT_INSERTED_POS,
    GRAPPLE_MOUNT_ANCHOR_CFG,
    GRAPPLE_ROBOT_ROOT_POS,
    SECOND_SLOT_CENTER_Y,
    SECOND_SLOT_INSERTED_POS,
    make_grapple_pin_robot_cfg,
)
from .grapple_pin_env_cfg import SingleStageCurriculumCfg, ZeroGBladeGrapplePinInsertEnvCfg
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .scene_cfg import ZeroGTwoSlotGrapplePinSceneCfg
from .workflow_demo_env_cfg import ZeroGBladeGrapplePinWorkflowEnvCfg

#: The arm pose that stages a head-on insertion into the **second** bay.
#:
#: Solved with the converged calibrator rather than assumed reachable --- rule 7,
#: because a pose in this workcell was once called unreachable on a 400-step IK
#: residual and converges to micrometres at 3,000. Produced by::
#:
#:     scripts/run_relocation.sh calibrate
#:
#: which is `calibrate_grasp_pose.py` on the **Capture** task at 3,000 steps with
#: `--target_offset 0 -0.22 0`. The report it writes is kept at
#: `artifacts/relocation/slot_two_pose.json`.
#:
#: Measured 2026-08-16: every stage converged, the full-distance one below to
#: **0.0060 mm and 0.000011 rad**, holding the head-on capture attitude. The
#: second bay is reachable with room to spare, which is what item 2's gate asked.
#:
#: One detail worth keeping, because it is the difference between "unreachable"
#: and "seeded badly": the middle stage did not converge from the zero wrist seed
#: and converged to 0.0070 mm from the -90 degree one. The calibrator sweeps four
#: wrist seeds for exactly this reason, and a single-seed run would have reported
#: a reachability failure that is not there.
SECOND_SLOT_STAGING_ARM_JOINT_POS: tuple[float, ...] = (
    -0.588512,
    -1.271002,
    1.856667,
    2.555911,
    -0.982291,
    -1.570782,
)

#: Where the module is presented for an insertion into the second bay: the
#: certified staging pose, displaced with the slot so the approach distance is
#: identical and only the bay differs.
SECOND_SLOT_STAGE_BLADE_POSE = (
    CONTACT_INSERTION_STAGE_BLADE_POSE[2][0],
    CONTACT_INSERTION_STAGE_BLADE_POSE[2][1] + SECOND_SLOT_CENTER_Y,
    *CONTACT_INSERTION_STAGE_BLADE_POSE[2][2:],
)


@configclass
class TwoSlotCommandsCfg:
    """One seated pose per bay, selected by the same index everything else uses."""

    insertion_goal = mdp.InsertionGoalCommandCfg(
        goal_pos=FIRST_SLOT_INSERTED_POS,
        goal_pos_by_stage=(FIRST_SLOT_INSERTED_POS, SECOND_SLOT_INSERTED_POS),
    )


@configclass
class DestinationBayCommandsCfg:
    """One seated pose: the bay the chain actually seats into.

    The two-bay goal it replaces indexed on a curriculum stage that no longer
    exists. See ``ZeroGBladeGrapplePinInsertTwoSlotEnvCfg``.
    """

    insertion_goal = mdp.InsertionGoalCommandCfg(goal_pos=SECOND_SLOT_INSERTED_POS)


@configclass
class TwoSlotCurriculumCfg:
    """Both bays from the first step, in equal measure.

    Barely a ramp. The policy being fine-tuned already solves the first bay, so
    level 0 is a formality it clears almost immediately; level 1 then draws the
    two bays evenly and stays there. Training the second bay *alone* is what the
    two-bay gate exists to catch — a policy that learns the new bay by forgetting
    the old one has not done the job.

    Level 0 draws only stage 0 rather than the even mixture this first carried:
    ``InsertionCurriculumMixtures`` refuses a level that samples a stage it has
    not unlocked, and it is right to. A level that can draw a locked stage is not
    a curriculum, and the validator caught it in the smoke before any GPU time
    went into it.
    """

    pose_noise = CurrTerm(
        func=mdp.InsertionSuccessRateCurriculum,
        params={
            "success_term": "insertion_success",
            "threshold": 0.80,
            "window_size": 2_000,
            "max_stage": 1,
            "minimum_level_steps": 1_600,
            "stage_mixtures": ((1.0, 0.0), (0.5, 0.5)),
        },
    )


@configclass
class ZeroGBladeGrapplePinInsertTwoSlotEnvCfg(ZeroGBladeGrapplePinInsertEnvCfg):
    """Seat a physically held module into the destination bay the chain uses.

    **One bay, with the robot parked opposite it, and that is a correction.**

    This used to draw the two bays evenly on the reasoning that a policy which
    learns the new bay by forgetting the old one has not done the job. That
    argument belongs to a robot bolted in one place. This one rides a lateral
    rail, and the rail is what makes a certification at one bay a certification
    at every bay it reaches: solved both ways, the arm's configuration parked
    opposite the second bay differs from its configuration at the first by
    **0.0000 mrad**.

    Reaching the second bay from the *first* bay's base is a different pose
    entirely -- 505 mrad on the worst joint -- and it is one the chain never
    presents, because the chain moves the base. So half of what this task used
    to train was a stretch nothing asks for, and the other half was the first
    bay, whose channel has no vertical lead-in because nothing was ever supposed
    to enter it from outside. Since the reset began at the mouth, something is.

    Both halves are gone. The robot parks where the chain parks it, the module
    seats where the chain seats it, and the rack it enters is the one with the
    lead-in the seating stroke depends on.
    """

    #: **Replication off, and it costs environments.** PhysX copies only the
    #: first environment's procedurally authored joint, so with replication on
    #: envs 1..N get the latch prim and no usable joint and the run dies with
    #: ``Fixed release latch is missing``. ``configure_base_rail`` records the
    #: same defect. This is the structural reason the chain's load path was not
    #: reachable from a skill task, and it is why this trains at 512 rather than
    #: 1024.
    scene: ZeroGTwoSlotGrapplePinSceneCfg = ZeroGTwoSlotGrapplePinSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    commands: DestinationBayCommandsCfg = DestinationBayCommandsCfg()
    curriculum: SingleStageCurriculumCfg = SingleStageCurriculumCfg()

    # **The chain's load path, carried by the task rather than by a flag.**
    #
    # The chain seats with the form lock softened to a bounded remote-centre
    # spring-damper, because a lead-in aligns a part by pushing it and a part
    # welded to a wrist cannot be pushed. This task trained without it and was
    # certified without it, and the difference was the insert skill's blocker:
    # trained on the mating compliance the median shortfall fell 202.2 -> 98.6 mm
    # and 35.5% of episodes reached seated depth against essentially none.
    # See docs/NEXT_WORK.md T9 and evidence/insert_attitude_diagnosis.json.
    #
    # It lives here rather than on ``train.py --latch_mating_compliance``
    # because a load path only a flag can reach is one the certification runs
    # without. ``scripts/verify_insert_skill.sh`` passes no such flag, so a
    # skill certified through it would otherwise be certified on pad contact
    # alone while the checkpoint it certifies was trained on the lock.
    latch_enabled: bool = True
    #: ``fixed``, not ``compliant``. With ``compliant`` the load path is the
    #: explicit wrench and the mating joint is never installed, so softening
    #: re-anchors a transform engagement set one line earlier -- measured
    #: byte-identical to not softening at all.
    latch_joint_mode: str = "fixed"
    latch_softens_on_engage: bool = True
    #: Five control steps. Engaging on the first qualifying step killed 100% of
    #: episodes inside ten steps, because this task's reset writes the module
    #: anywhere along a 436 mm stroke while the joint is authored at the spawn
    #: poses, and PhysX resolves the disagreement by snapping the two together.
    #: Deferring removes that entirely: 0% dead inside ten steps.
    latch_engage_after_steps: int = 5

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        if not SECOND_SLOT_STAGING_ARM_JOINT_POS:
            raise ValueError(
                "SECOND_SLOT_STAGING_ARM_JOINT_POS is empty: the second bay's staging pose has not "
                "been solved yet. Run `scripts/run_relocation.sh calibrate` and paste the converged "
                "joint angles into two_slot_env_cfg.py. Guessing it is what rule 7 forbids."
            )
        # **Park the robot opposite the bay it is seating into.**
        #
        # The chain does this with the lateral rail and the whole rail argument
        # rests on it: parked opposite a bay the arm's configuration is the one
        # it has at bay 1, so no bay needs a skill bay 1 does not already have.
        # A skill task that reaches the second bay by stretching from the first
        # bay's base is training the one configuration the rail exists to avoid.
        if level < 2:
            self.events.blade_mass = None
        if level < 3:
            self.events.slot_material = None
            self.events.left_guide_material = None
            self.events.right_guide_material = None
            self.events.randomize_stiction = None
            self.events.rail_stiction_force = None
        self.scene.robot = make_grapple_pin_robot_cfg(floating=level >= 4)
        # **Park the robot opposite the bay it is seating into.**
        #
        # After ``make_grapple_pin_robot_cfg``, which rebuilds the robot. The
        # chain does this with the lateral rail and the whole rail argument
        # rests on it: parked opposite a bay the arm's configuration is the one
        # it has at bay 1, solved both ways to 0.0000 mrad. A skill task that
        # reaches the second bay by stretching from the first bay's base trains
        # the one configuration -- 505 mrad away -- the rail exists to avoid.
        self.scene.robot.init_state.pos = (
            GRAPPLE_ROBOT_ROOT_POS[0],
            GRAPPLE_ROBOT_ROOT_POS[1] + SECOND_SLOT_CENTER_Y,
            GRAPPLE_ROBOT_ROOT_POS[2],
        )
        # Absolute against the authored anchor, not an increment: this method
        # runs once from ``__post_init__`` and again from every caller that
        # re-selects the level, and an increment moved the anchor to -0.44 on
        # the second call -- twice the bay it is meant to sit opposite. Same
        # defect, same fix, as the channel relief in
        # ``configure_service_destination``.
        mount = getattr(self.scene, "mount_anchor", None)
        if mount is not None:
            anchor = tuple(mount.init_state.pos)
            authored = GRAPPLE_MOUNT_ANCHOR_CFG.init_state.pos
            mount.init_state.pos = (anchor[0], authored[1] + SECOND_SLOT_CENTER_Y, anchor[2])
        if level < 4:
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None
        # **Fit the destination bay the way the chain fits it.**
        #
        # ``configure_service_destination`` installs the vertical entry ramps,
        # applies the channel relief and puts the destination bay's running
        # surfaces on the low-friction pairing, and until now only
        # ``run_workflow_demo.py`` ever called it. So every insert policy this
        # project trained was trained in a bay with no vertical lead-in, no
        # relief and production friction -- and that bay's own docstring records
        # the measurement: a module delivered from outside "cocked to 36 mrad,
        # exactly the 2c/L the channel admits, and then did not move for six
        # thousand control steps of pushing, under every mating variant tried".
        #
        # The skill was being asked for something the geometry forbids. That is
        # the whole of why it certifies at 0.00% while holding the grip
        # perfectly, and it is the last of the ways this task and the chain
        # described different problems.
        self.configure_service_destination()
        self._configure_latch()


@configclass
class ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class RelocationCommandsCfg:
    """The goal is the second bay, always.

    Not a per-stage goal like the two-slot insert task: a relocation *starts* in
    bay one and *finishes* in bay two, so the reset stage and the goal are about
    different bays by definition and keying both off one index would be wrong.
    """

    insertion_goal = mdp.InsertionGoalCommandCfg(goal_pos=SECOND_SLOT_INSERTED_POS)


@configclass
class ZeroGBladeGrapplePinTwoSlotWorkflowEnvCfg(ZeroGBladeGrapplePinWorkflowEnvCfg):
    """Capture, extract, cross to the next bay, and seat, in one episode.

    The relocation: the deliverable this whole roadmap is for. Physics, robot,
    pin and contacts are the workflow profile's; what changes is that the rack
    has a second bay and the seated goal is in it.
    """

    scene: ZeroGTwoSlotGrapplePinSceneCfg = ZeroGTwoSlotGrapplePinSceneCfg(
        num_envs=1,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    commands: RelocationCommandsCfg = RelocationCommandsCfg()
    # Capture, the pull, a three-leg transit and the insertion, each on its own
    # certified clock, plus the seating pause. Generous rather than tight: a
    # demonstration that runs out of time mid-insertion is worse than one that
    # idles at the end, and the per-phase budgets are what actually bound it.
    episode_length_s: float = 90.0


@configclass
class ZeroGBladeGrapplePinTwoSlotWorkflowPlayEnvCfg(ZeroGBladeGrapplePinTwoSlotWorkflowEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "RelocationCommandsCfg",
    "SECOND_SLOT_STAGE_BLADE_POSE",
    "SECOND_SLOT_STAGING_ARM_JOINT_POS",
    "TwoSlotCommandsCfg",
    "TwoSlotCurriculumCfg",
    "ZeroGBladeGrapplePinInsertTwoSlotEnvCfg",
    "ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg",
    "ZeroGBladeGrapplePinTwoSlotWorkflowEnvCfg",
    "ZeroGBladeGrapplePinTwoSlotWorkflowPlayEnvCfg",
]
