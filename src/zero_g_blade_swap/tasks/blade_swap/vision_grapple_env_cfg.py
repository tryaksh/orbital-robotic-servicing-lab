"""Camera collection and leak-free visual workflow configurations.

The deployed profile replaces every policy channel derived from the live module
state: grip error, extraction remaining, insertion-goal error, and module
linear/angular velocity. All channels share one cached pose estimate per control
step. Robot joint state, tool forward kinematics, gripper sensing, commands, and
previous action remain available because equivalent signals exist on hardware.

The collection profile deliberately retains exact labels beside images for
offline supervised training. It is not a deployment profile. The workflow
profile is audited during configuration and fails closed if one of the exact
module-state observation functions is wired back into grasp, extract, or insert.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .grapple_pin_env_cfg import GrappleSkillObsCfg
from .scene_cfg import ZeroGGrapplePinSceneCfg, make_tiled_camera_cfg
from .vision_insertion_env_cfg import VisualRandomizationCfg
from .workflow_demo_env_cfg import (
    WorkflowCurriculumCfg,
    WorkflowExtractObsCfg,
    WorkflowInsertObsCfg,
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
    """Privileged regression label: module pose in the environment frame.

    Recorded beside each image by ``scripts/collect_grapple_vision.py``. The
    environment-local position avoids encoding the cloned simulation tile. The
    axis-angle orientation and position are both visible in the image; the
    tool-relative policy values are derived later with forward kinematics.

    This group exists only on the collection environment and is forbidden from
    all three deployed policy groups.
    """

    module_pose = ObsTerm(func=mdp.module_pose_label)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GrappleRgbObsCfg(ObsGroup):
    """Normalized camera RGB with the configured sensor-noise model."""

    rgb = ObsTerm(
        func=mdp.camera_rgb_with_radiation_noise,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionGrappleCollectObsCfg(WorkflowObservationsCfg):
    """Images, privileged labels, and state-policy inputs for data collection.

    This is intentionally not a deployable observation configuration. Exact
    state drives the existing policies while paired image/label examples are
    written for offline training.
    """

    rgb: GrappleRgbObsCfg = GrappleRgbObsCfg()
    pose_label: GraspPoseLabelObsCfg = GraspPoseLabelObsCfg()


@configclass
class VisionGrappleEventsCfg(VisualRandomizationCfg):
    """Randomized optics and an episode-random module displacement.

    Moving the module prevents a pose head from memorizing the finite set of
    nominal workflow stage poses.
    """

    jitter_module = EventTerm(func=mdp.jitter_module_pose, mode="reset")


@configclass
class ZeroGBladeGrappleVisionCollectEnvCfg(ZeroGBladeGrapplePinWorkflowEnvCfg):
    """Run the state-driven workflow while recording camera supervision."""

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
        # The 640 px tiled RGB-D render product has two queued buffers behind
        # the reset render.  With one rerender, some clean process starts expose
        # the annotator's zero-filled allocation for the whole first episode.
        # Drain those buffers while the workcell is still at reset; this changes
        # neither physics nor the 15 Hz measurement cadence.
        self.num_rerenders_on_reset = 3

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # Every ancestor rebuilds the event set from its own class, which drops
        # the Replicator terms. Copy its configured result onto the visual event
        # subclass rather than duplicating every ancestor's decisions here.
        visual = VisionGrappleEventsCfg()
        for name, value in self.events.__dict__.items():
            setattr(visual, name, value)
        self.events = visual


@configclass
class PerceivedGrappleSkillObsCfg(GrappleSkillObsCfg):
    """Grasp-policy layout with every module-state channel camera-derived.

    Widths remain unchanged: the grip error is six values and filtered module
    velocity is six values. Joint, tool, and gripper terms remain legitimate
    proprioceptive signals.
    """

    grip_error = ObsTerm(func=mdp.PerceivedGraspError)
    blade_velocity = ObsTerm(func=mdp.PerceivedModuleVelocity, scale=0.10)


@configclass
class PerceivedWorkflowExtractObsCfg(WorkflowExtractObsCfg):
    """Extract-policy layout with pose, travel, and velocity from the camera."""

    grip_error = ObsTerm(func=mdp.PerceivedGraspError)
    blade_velocity = ObsTerm(func=mdp.PerceivedModuleVelocity, scale=0.10)
    remaining_travel = ObsTerm(func=mdp.PerceivedExtractionRemaining)


@configclass
class PerceivedWorkflowInsertObsCfg(WorkflowInsertObsCfg):
    """Insert-policy layout with pose, goal error, and velocity from the camera."""

    grip_error = ObsTerm(func=mdp.PerceivedGraspError)
    blade_velocity = ObsTerm(func=mdp.PerceivedModuleVelocity, scale=0.10)
    blade_goal_error = ObsTerm(func=mdp.PerceivedInsertionGoalError)


@configclass
class PerceivedWorkflowObsCfg(WorkflowObservationsCfg):
    """All three policy groups bound to one shared cached module estimator."""

    grasp: PerceivedGrappleSkillObsCfg = PerceivedGrappleSkillObsCfg()
    extract: PerceivedWorkflowExtractObsCfg = PerceivedWorkflowExtractObsCfg()
    insert: PerceivedWorkflowInsertObsCfg = PerceivedWorkflowInsertObsCfg()

    def __post_init__(self) -> None:
        # Keep the guard with the deployable group itself so profiles such as
        # the two-bay workflow inherit the contract without needing to remember
        # an environment-level hook.
        mdp.audit_vision_deployment_observations(self)


@configclass
class ZeroGBladeGrappleVisionWorkflowEnvCfg(ZeroGBladeGrappleVisionCollectEnvCfg):
    """The full workflow driven by one camera estimate per control step."""

    observations: PerceivedWorkflowObsCfg = PerceivedWorkflowObsCfg()
    rewards: WorkflowRewardsCfg = WorkflowRewardsCfg()
    terminations: WorkflowTerminationsCfg = WorkflowTerminationsCfg()
    curriculum: WorkflowCurriculumCfg = WorkflowCurriculumCfg()

    pose_head_checkpoint: str | None = None
    perception_mode: str = mdp.PERCEPTION_DEPLOYMENT
    # ``fiducial_pnp`` is the production path: calibrated geometry with a
    # fail-closed reprojection gate. ``pose_head`` remains available as the
    # learned-perception research baseline.
    perception_backend: str = mdp.PERCEPTION_BACKEND_POSE_HEAD
    perception_velocity_filter_time_constant_s: float = 0.10
    # Drawing-level prior for the blind control.  It is fixed by the requested
    # relocation scenario and is not populated from simulator state at reset.
    perception_blind_occupancy: tuple[float, float] = (1.0, 0.0)
    # Compatibility with the existing evaluation driver. Only the endpoints are
    # accepted; intermediate oracle blends fail closed in the estimator.
    pose_head_oracle_blend: float = 0.0
    pose_head_blind: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        mdp.audit_vision_deployment_observations(self.observations)


__all__ = [
    "GraspPoseLabelObsCfg",
    "GrappleRgbObsCfg",
    "PerceivedGrappleSkillObsCfg",
    "PerceivedWorkflowExtractObsCfg",
    "PerceivedWorkflowInsertObsCfg",
    "PerceivedWorkflowObsCfg",
    "VisionGrappleCollectObsCfg",
    "VisionGrappleEventsCfg",
    "VisionGrappleSceneCfg",
    "ZeroGBladeGrappleVisionCollectEnvCfg",
    "ZeroGBladeGrappleVisionWorkflowEnvCfg",
]
