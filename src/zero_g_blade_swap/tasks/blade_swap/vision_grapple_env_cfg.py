"""The servicing workflow seen through a camera instead of through the simulator.

Every policy in this repository so far reads the module's pose straight out of
the simulator. That is the project's central weakness and it has been recorded
as one since 2026-08-10: with a rigid known object and perfect knowledge, this
is motion planning, and reinforcement learning is not earning its keep. The
pose-belief experiment closed half the gap by making the *slot* physically move
by an amount no observation reveals, and certified 99.87% under it. What was
never closed is where the estimate comes from.

This profile closes it. The scene is the head-on grapple-pin workcell, unchanged
in every physical respect — same pin, same contacts, same 69 N interface — with
a camera added and the rack's albedo and the orbital sun randomized per episode.
``mdp.PerceivedGraspError`` replaces the ground-truth grip vector with one
regressed from that camera, so the capture and insertion policies run on an
estimate they have to live with rather than on the truth.

Three things make the comparison honest rather than decorative:

* **The policies are unchanged.** The same certified checkpoints run in both
  arms. Nothing is retrained to flatter the camera, so the difference between
  the two numbers is the perception and nothing else.
* **The label is the quantity the policy actually consumes**, not a proxy. The
  regressor predicts the module pose; the grip vector is then computed from it
  and the arm's own forward kinematics, which is what a real system would do and
  is legitimately known.
* **The randomization is on during collection and evaluation both.** A pose head
  trained under one lighting condition and tested under the same one measures
  nothing.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .grapple_pin_env_cfg import (
    GrappleSkillObsCfg,
)
from .scene_cfg import ZeroGGrapplePinSceneCfg, make_tiled_camera_cfg
from .vision_insertion_env_cfg import VisualRandomizationCfg
from .workflow_demo_env_cfg import (
    WorkflowCurriculumCfg,
    WorkflowObservationsCfg,
    WorkflowRewardsCfg,
    WorkflowTerminationsCfg,
    ZeroGBladeGrapplePinWorkflowEnvCfg,
)


@configclass
class VisionGrappleSceneCfg(ZeroGGrapplePinSceneCfg):
    """The head-on capture scene with a servicing camera watching the interface."""

    camera: TiledCameraCfg = make_tiled_camera_cfg()


@configclass
class GraspPoseLabelObsCfg(ObsGroup):
    """The regression label: where the module actually is, in the tool's frame.

    Recorded beside each image by ``scripts/collect_grapple_vision.py``. It is
    the module's pose in its own environment's frame -- position and axis-angle
    orientation -- because that is what is genuinely *in* the image. The
    tool-relative grip vector the policy consumes is computed from this estimate
    plus the arm's own forward kinematics, which a real robot knows exactly.
    """

    module_pose = ObsTerm(func=mdp.module_pose_label)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GrappleRgbObsCfg(ObsGroup):
    """What the servicing camera sees, with sensor noise."""

    rgb = ObsTerm(
        func=mdp.camera_rgb_with_radiation_noise,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionGrappleCollectObsCfg(WorkflowObservationsCfg):
    """Everything the collector needs in one step: images, labels, and the
    three policies' own inputs so the workflow can be driven while recording."""

    rgb: GrappleRgbObsCfg = GrappleRgbObsCfg()
    pose_label: GraspPoseLabelObsCfg = GraspPoseLabelObsCfg()


@configclass
class VisionGrappleEventsCfg(VisualRandomizationCfg):
    """Randomized optics *and* an unknown module pose.

    The second is what makes this a perception problem rather than a lighting
    demo: without it the module sits at one of three fixed stage poses and a
    head could memorise them.
    """

    jitter_module = EventTerm(func=mdp.jitter_module_pose, mode="reset")


@configclass
class ZeroGBladeGrappleVisionCollectEnvCfg(ZeroGBladeGrapplePinWorkflowEnvCfg):
    """Run the certified workflow while recording what a camera would have seen."""

    scene: VisionGrappleSceneCfg = VisionGrappleSceneCfg(
        num_envs=64,
        env_spacing=2.6,
        # Both randomizers address per-environment prims, which cloning would
        # collapse into one shared material and one shared light.
        replicate_physics=False,
        clone_in_fabric=False,
    )
    observations: VisionGrappleCollectObsCfg = VisionGrappleCollectObsCfg()
    events: VisionGrappleEventsCfg = VisionGrappleEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Eight physics steps per render exactly matches the 15 Hz camera.
        self.sim.render_interval = 8
        self.num_rerenders_on_reset = 1

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # Every ancestor rebuilds the event set from its own class, which drops
        # the two Replicator terms. Copy the fully configured result onto a
        # subclass that declares them rather than replaying each ancestor's
        # decisions and drifting out of sync with them.
        visual = VisionGrappleEventsCfg()
        for name, value in self.events.__dict__.items():
            setattr(visual, name, value)
        self.events = visual


@configclass
class PerceivedGrappleSkillObsCfg(GrappleSkillObsCfg):
    """The skills' own observation, with the grip vector coming from the camera.

    One term differs from what the policies trained on, and it is deliberately
    the only one: ``grip_error`` is regressed from the image instead of read
    from the simulator. Everything else — joint state, tool pose, gripper state,
    module velocity — is proprioception a real robot genuinely has.
    """

    grip_error = ObsTerm(func=mdp.PerceivedGraspError)


@configclass
class PerceivedWorkflowObsCfg(WorkflowObservationsCfg):
    grasp: PerceivedGrappleSkillObsCfg = PerceivedGrappleSkillObsCfg()


@configclass
class ZeroGBladeGrappleVisionWorkflowEnvCfg(ZeroGBladeGrappleVisionCollectEnvCfg):
    """The servicing workflow driven by a camera estimate rather than the truth."""

    observations: PerceivedWorkflowObsCfg = PerceivedWorkflowObsCfg()
    rewards: WorkflowRewardsCfg = WorkflowRewardsCfg()
    terminations: WorkflowTerminationsCfg = WorkflowTerminationsCfg()
    curriculum: WorkflowCurriculumCfg = WorkflowCurriculumCfg()


__all__ = [
    "GraspPoseLabelObsCfg",
    "GrappleRgbObsCfg",
    "PerceivedGrappleSkillObsCfg",
    "PerceivedWorkflowObsCfg",
    "VisionGrappleCollectObsCfg",
    "VisionGrappleEventsCfg",
    "VisionGrappleSceneCfg",
    "ZeroGBladeGrappleVisionCollectEnvCfg",
    "ZeroGBladeGrappleVisionWorkflowEnvCfg",
]
