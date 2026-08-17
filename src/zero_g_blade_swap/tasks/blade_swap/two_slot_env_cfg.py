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
    BLADE_INSERTED_POS,
    CONTACT_INSERTION_STAGE_BLADE_POSE,
    GRAPPLE_HEAD_ON_ARM_JOINT_POS,
    SECOND_SLOT_CENTER_Y,
    SECOND_SLOT_INSERTED_POS,
    make_grapple_pin_robot_cfg,
)
from .grapple_pin_env_cfg import ExtractEventsCfg, ZeroGBladeGrapplePinInsertEnvCfg
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
    -0.820182,
    -1.560585,
    2.204724,
    2.497455,
    -0.750621,
    -1.570789,
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
        goal_pos=BLADE_INSERTED_POS,
        goal_pos_by_stage=(BLADE_INSERTED_POS, SECOND_SLOT_INSERTED_POS),
    )


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
    """Insert a physically held module into either bay of a two-bay rack."""

    scene: ZeroGTwoSlotGrapplePinSceneCfg = ZeroGTwoSlotGrapplePinSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    commands: TwoSlotCommandsCfg = TwoSlotCommandsCfg()
    curriculum: TwoSlotCurriculumCfg = TwoSlotCurriculumCfg()

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        if not SECOND_SLOT_STAGING_ARM_JOINT_POS:
            raise ValueError(
                "SECOND_SLOT_STAGING_ARM_JOINT_POS is empty: the second bay's staging pose has not "
                "been solved yet. Run `scripts/run_relocation.sh calibrate` and paste the converged "
                "joint angles into two_slot_env_cfg.py. Guessing it is what rule 7 forbids."
            )
        # The parent collapses these to the single certified bay; restore two
        # entries, in the order the goal and the curriculum use.
        self.events = ExtractEventsCfg()
        self.events.reset_arm.params["poses_by_stage"] = (
            GRAPPLE_HEAD_ON_ARM_JOINT_POS[2],
            tuple(SECOND_SLOT_STAGING_ARM_JOINT_POS),
        )
        self.events.reset_arm.params["noise_by_stage"] = (0.020, 0.020)
        self.events.reset_blade.params["poses_by_stage"] = (
            CONTACT_INSERTION_STAGE_BLADE_POSE[2],
            SECOND_SLOT_STAGE_BLADE_POSE,
        )
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
