"""The relocation seen through a camera, on a rack that has somewhere to look.

The single-bay vision profile asks the camera one question: *where is the
module*. That is a real question and it is certified — but a servicer standing in
front of a rack asks a different one first, and it is the one a single bay cannot
pose: **which bay holds the part**.

This profile is the two-bay version of `vision_grapple_env_cfg`, and it changes
exactly two things about it:

* the scene is `ZeroGTwoSlotGrapplePinSceneCfg` rather than the single-bay one,
  so there are two channels for the camera to distinguish and the relocation has
  somewhere to put the module;
* the collector records `mdp.slot_occupancy_label` beside the pose label, which
  is the supervision for the pose head's occupancy branch.

Everything else is inherited rather than restated: the same camera, the same
orbital lighting and albedo randomization, the same module jitter, the same
`PerceivedGraspError` term standing where the ground-truth grip vector stands in
the state profile. That is deliberate — the claim being tested is what the camera
adds and costs, so any second difference between the arms would confound it.

Separate registrations, like every other geometry change here, so
`evidence/vision_workflow_{camera,oracle,blind}_certification.json` keep
describing the single-bay workflow they were run on.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .assets import BLADE_INSERTED_POS, CONTACT_INSERTION_STAGE_BLADE_POSE, SECOND_SLOT_CENTER_Y
from .scene_cfg import ZeroGTwoSlotGrapplePinSceneCfg, make_tiled_camera_cfg
from .two_slot_env_cfg import RelocationCommandsCfg
from .vision_grapple_env_cfg import (
    PerceivedWorkflowObsCfg,
    VisionGrappleCollectObsCfg,
    VisionGrappleEventsCfg,
    ZeroGBladeGrappleVisionCollectEnvCfg,
)
from .workflow_demo_env_cfg import (
    WorkflowCurriculumCfg,
    WorkflowRewardsCfg,
    WorkflowTerminationsCfg,
)


#: Where the module is presented for collection, one pose per curriculum stage.
#:
#: The occupancy branch is a classifier and it needs to see all three states the
#: rack can be in. The workflow profile's own stage poses are three *depths* in
#: the first bay, which would label every frame as "bay one, or on its way there"
#: and teach the head the prior rather than the rack. So:
#:
#:   stage 0   seated in bay one          -> (1, 0)
#:   stage 1   seated in bay two          -> (0, 1)
#:   stage 2   presented at the mouth     -> (0, 0)
#:
#: Bay two's pose is bay one's displaced by ``SECOND_SLOT_CENTER_Y``, part for
#: part, exactly as the bay itself is -- so a difference between the two in the
#: camera's answer is attributable to position alone. The per-episode module
#: jitter that makes this a perception problem at all is +-15 mm laterally, well
#: inside the 72.5 mm channel half-width the label is defined against, so it
#: cannot flip a label.
VISION_TWO_SLOT_STAGE_BLADE_POSE = (
    CONTACT_INSERTION_STAGE_BLADE_POSE[0],
    (
        CONTACT_INSERTION_STAGE_BLADE_POSE[0][0],
        CONTACT_INSERTION_STAGE_BLADE_POSE[0][1] + SECOND_SLOT_CENTER_Y,
        *CONTACT_INSERTION_STAGE_BLADE_POSE[0][2:],
    ),
    CONTACT_INSERTION_STAGE_BLADE_POSE[2],
)


@configclass
class VisionTwoSlotGrappleSceneCfg(ZeroGTwoSlotGrapplePinSceneCfg):
    """Two bays, with the same servicing camera watching the interface.

    The camera is `make_tiled_camera_cfg()` unchanged, mount and 180 mm focal
    length included. It is not re-aimed for the second bay, and that is the
    honest configuration to measure: a fixed servicing camera is what a real
    manipulator carries, and whether this one frames both bays well enough to
    tell them apart is a question for the occupancy accuracy to answer rather
    than one to design away.
    """

    camera: TiledCameraCfg = make_tiled_camera_cfg()


@configclass
class SlotOccupancyLabelObsCfg(ObsGroup):
    """The classification label: which bays hold the module, one value each.

    A separate observation group from the pose label rather than more channels
    on it. They are different kinds of quantity with different losses — metres
    and radians against a logit — and concatenating them is how a bay ends up
    being reported in millimetres by a normalised regression loss.
    """

    occupancy = ObsTerm(func=mdp.slot_occupancy_label)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionTwoSlotCollectObsCfg(VisionGrappleCollectObsCfg):
    occupancy_label: SlotOccupancyLabelObsCfg = SlotOccupancyLabelObsCfg()


@configclass
class ZeroGBladeGrappleVisionTwoSlotCollectEnvCfg(ZeroGBladeGrappleVisionCollectEnvCfg):
    """Record what the camera sees of a two-bay rack, and which bay is full.

    Inherits the collect profile's render interval, reset re-render and event
    reconstruction, so the only declared differences are the scene, the goal and
    the extra label.
    """

    scene: VisionTwoSlotGrappleSceneCfg = VisionTwoSlotGrappleSceneCfg(
        num_envs=64,
        env_spacing=2.6,
        # Both randomizers address per-environment prims, which cloning would
        # collapse into one shared material and one shared light. Restated from
        # the parent because declaring a scene replaces it wholesale.
        replicate_physics=False,
        clone_in_fabric=False,
    )
    observations: VisionTwoSlotCollectObsCfg = VisionTwoSlotCollectObsCfg()
    events: VisionGrappleEventsCfg = VisionGrappleEventsCfg()
    # A relocation finishes in the second bay, so the goal is that bay and not a
    # per-stage lookup: the reset stage says where the module *starts*, which is
    # a different bay by definition. Read from two_slot_env_cfg rather than
    # rebuilt, so the state and camera relocations cannot be scored against
    # different goals.
    commands: RelocationCommandsCfg = RelocationCommandsCfg()
    # Capture, the pull, the three-leg transit and the insertion, each on its own
    # certified clock. The driver derives the episode from the phases the
    # workflow actually runs and overwrites this, so it is a ceiling rather than
    # the operative number.
    episode_length_s: float = 90.0

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # Present the module in whichever bay this episode's stage names, so the
        # occupancy branch sees all three states of the rack rather than three
        # depths in one bay. Only the module's pose changes: the collector holds
        # the arm still and resets often, which is what a pose regressor needs.
        self.events.reset_blade.params["poses_by_stage"] = VISION_TWO_SLOT_STAGE_BLADE_POSE


@configclass
class ZeroGBladeGrappleVisionTwoSlotWorkflowEnvCfg(ZeroGBladeGrappleVisionTwoSlotCollectEnvCfg):
    """The relocation driven by a camera estimate rather than by the truth."""

    observations: PerceivedWorkflowObsCfg = PerceivedWorkflowObsCfg()
    rewards: WorkflowRewardsCfg = WorkflowRewardsCfg()
    terminations: WorkflowTerminationsCfg = WorkflowTerminationsCfg()
    curriculum: WorkflowCurriculumCfg = WorkflowCurriculumCfg()


@configclass
class TwoSlotInstallCommandsCfg:
    """The seated pose in the FIRST bay, on a rack that has two of them."""

    insertion_goal = mdp.InsertionGoalCommandCfg(goal_pos=BLADE_INSERTED_POS)


@configclass
class ZeroGBladeGrappleVisionTwoSlotInstallEnvCfg(ZeroGBladeGrappleVisionTwoSlotWorkflowEnvCfg):
    """Installation into the first bay, on a two-bay rack, seen through the camera.

    Why this exists rather than only the relocation: the three vision arms measure
    what perception *costs*, and that measurement needs a manipulation task that
    completes. The relocation chain does not yet complete for reasons that have
    nothing to do with the camera --- see the transit findings in docs/status.md ---
    so running the arms on it would compare three ways of failing.

    The installation does complete, and it is certified on the single-bay rack at
    96.35%. Putting it on the two-bay rack changes exactly what item 6 is about:
    the camera now has to find the module on a rack where a second bay exists and
    could hold it, which is the harder perception problem, while the manipulation
    stays the one with a certified baseline to compare against.

    The goal is therefore the first bay, not the second. Keying it off the reset
    stage the way the two-slot insert task does would be wrong here for the same
    reason it is wrong for the relocation: this workflow starts and finishes in the
    same bay.
    """

    commands: TwoSlotInstallCommandsCfg = TwoSlotInstallCommandsCfg()


__all__ = [
    "SlotOccupancyLabelObsCfg",
    "VisionTwoSlotCollectObsCfg",
    "VisionTwoSlotGrappleSceneCfg",
    "TwoSlotInstallCommandsCfg",
    "ZeroGBladeGrappleVisionTwoSlotCollectEnvCfg",
    "ZeroGBladeGrappleVisionTwoSlotInstallEnvCfg",
    "ZeroGBladeGrappleVisionTwoSlotWorkflowEnvCfg",
]
