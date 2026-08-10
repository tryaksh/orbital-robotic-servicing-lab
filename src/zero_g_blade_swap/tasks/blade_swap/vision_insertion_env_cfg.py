"""Force-feedback insertion seen through a camera instead of through state.

This task exists so the visual-randomization machinery stays *reachable*. Until
2026-08-10 the tiled camera, ``OrbitalLightingRandomizer``,
``RackMaterialRandomizer``, ``camera_rgb_with_radiation_noise``, and the
``VisionActor`` were reachable only from the eight-phase swap task, which was
deleted. Rather than delete weeks of infrastructure with it, they are repointed
here, at the insertion scene the certified results actually live on.

**Nothing here is certified, and no policy has been trained on it.** It is the
scaffold for P3, where the injected pose belief of the P1 task is replaced by a
real estimate regressed from these images. Two things are deliberately left for
that work rather than guessed at now:

- The camera pose is the one authored for the swap scene, unchanged. Whether it
  frames the slot well enough to regress a millimetre-scale pose error is a
  measurement, not an assumption.
- Both randomizers require ``replicate_physics=False``, which costs throughput.
  Benchmark before committing to a long run.

The actor sees proprioception and RGB only. Ground-truth blade pose stays in the
critic and in a separate diagnostic group, which is also what
``scripts/collect_teacher.py`` records as the pose-regression label.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .contact_insertion_env_cfg import ContactInsertionEventsCfg
from .force_limited_insertion_env_cfg import (
    FORCE_OBSERVATION_FILTER_S,
    FORCE_OBSERVATION_SCALE_N,
    ForceFeedbackPolicyObsCfg,
    ForceLimitedInsertionSceneCfg,
    ZeroGBladeForceFeedbackInsertionEnvCfg,
)
from .insertion_env_cfg import ARM_CFG, WRIST_CFG
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .scene_cfg import make_tiled_camera_cfg


@configclass
class VisionInsertionSceneCfg(ForceLimitedInsertionSceneCfg):
    """The force-feedback insertion scene with a 64x64 tiled camera."""

    camera: TiledCameraCfg = make_tiled_camera_cfg()


@configclass
class InsertionProprioObsCfg(ObsGroup):
    """What a deployable actor could actually measure on hardware.

    No blade pose. That is the entire point: on this task the blade's position
    has to come from the image, not from the simulator. Contact force stays,
    because a wrist force/torque sensor is a real instrument and because this
    repository has already measured that removing it costs 59% more contact
    impulse.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ARM_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ARM_CFG})
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    contact_wrench = ObsTerm(
        func=mdp.BladeContactWrenchObservation,
        params={
            "force_scale_n": FORCE_OBSERVATION_SCALE_N,
            "filter_time_constant_s": FORCE_OBSERVATION_FILTER_S,
        },
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class InsertionRgbObsCfg(ObsGroup):
    image = ObsTerm(
        func=mdp.camera_rgb_with_radiation_noise,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "rgb",
            "noise_std_range": (0.025, 0.025),
        },
        clip=(0.0, 1.0),
    )

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class InsertionCriticObsCfg(ObsGroup):
    """Privileged state for asymmetric actor-critic training.

    Everything the actor is denied: the true blade-to-goal error, the blade's
    velocity, the mount's true deflection, and the randomized dynamics the actor
    can only infer.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ARM_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ARM_CFG})
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    blade_goal_error = ObsTerm(func=mdp.insertion_goal_error)
    blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)
    mount_state = ObsTerm(func=mdp.robot_mount_state)
    robot_root = ObsTerm(func=mdp.robot_root_pose_local)
    randomized_physics = ObsTerm(func=mdp.randomized_physics_parameters)
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class BladePoseDiagnosticObsCfg(ObsGroup):
    """The pose-regression label, kept as its own group so it can be recorded.

    ``scripts/collect_teacher.py`` writes this group beside each image, which is
    exactly the supervision a P3 pose head needs.
    """

    goal_error = ObsTerm(func=mdp.insertion_goal_error)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionInsertionObservationsCfg:
    proprio: InsertionProprioObsCfg = InsertionProprioObsCfg()
    rgb: InsertionRgbObsCfg = InsertionRgbObsCfg()
    critic: InsertionCriticObsCfg = InsertionCriticObsCfg()
    blade_pose: BladePoseDiagnosticObsCfg = BladePoseDiagnosticObsCfg()


@configclass
class VisionInsertionPlayObservationsCfg(VisionInsertionObservationsCfg):
    """Adds the state teacher's own observation group alongside the images.

    ``scripts/collect_teacher.py`` drives this environment with a force-feedback
    checkpoint while recording what a camera would have seen, so the teacher's
    exact 57-value input has to be computed in the same environment. Training
    profiles do not pay for it.
    """

    policy: ForceFeedbackPolicyObsCfg = ForceFeedbackPolicyObsCfg()


@configclass
class VisualRandomizationCfg(ContactInsertionEventsCfg):
    """Replicator rack albedo and EventManager orbital-sun randomization.

    Both terms refuse to construct unless ``replicate_physics=False``, because
    Replicator has to address one material and one light per environment.
    """

    rack_albedo = EventTerm(
        func=mdp.RackMaterialRandomizer,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("rack"),
            "mesh_name": "geometry/mesh",
            "steel_color_range": ((0.08, 0.09, 0.10), (0.72, 0.78, 0.84)),
            "gold_color_range": ((0.45, 0.22, 0.02), (1.0, 0.78, 0.24)),
            "metallic_range": (0.65, 1.0),
            "roughness_range": (0.08, 0.55),
            "gold_probability": 0.5,
        },
    )
    orbital_sun = EventTerm(
        func=mdp.OrbitalLightingRandomizer,
        mode="reset",
        params={
            "prim_path": "/World/OrbitalSun",
            "intensity_range": (2_500.0, 8_000.0),
            "angle_range": (0.10, 0.45),
            "pitch_range_deg": (20.0, 80.0),
            "yaw_range_deg": (0.0, 360.0),
            "color_temperature_range": (5_500.0, 7_500.0),
        },
    )


@configclass
class ZeroGBladeVisionInsertionEnvCfg(ZeroGBladeForceFeedbackInsertionEnvCfg):
    """Insertion from 64x64 RGB plus proprioception, under orbital lighting.

    Identical physics, actions, rewards, and terminations to the force-feedback
    task. What changes is the actor's input: images instead of the simulator's
    blade pose.
    """

    scene: VisionInsertionSceneCfg = VisionInsertionSceneCfg(
        num_envs=128,
        env_spacing=2.6,
        # Both randomizers address per-environment prims, which cloning would
        # collapse; the contact sensor needs real USD clones for the same reason.
        replicate_physics=False,
        clone_in_fabric=False,
    )
    observations: VisionInsertionObservationsCfg = VisionInsertionObservationsCfg()
    events: VisualRandomizationCfg = VisualRandomizationCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Eight physics steps per render exactly matches the 15 Hz camera.
        self.sim.render_interval = 8
        self.num_rerenders_on_reset = 1

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # Every ancestor's ``configure_robustness`` rebuilds the event set from
        # its own class, which drops the two Replicator terms. Copy the fully
        # configured result onto a subclass that declares them, rather than
        # replaying each ancestor's decisions here and drifting out of sync with
        # them. ``VisualRandomizationCfg`` is a strict superset of the fields the
        # rebuilt object carries.
        visual = VisualRandomizationCfg()
        for name, value in self.events.__dict__.items():
            setattr(visual, name, value)
        self.events = visual


@configclass
class ZeroGBladeVisionInsertionPlayEnvCfg(ZeroGBladeVisionInsertionEnvCfg):
    observations: VisionInsertionPlayObservationsCfg = VisionInsertionPlayObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        configure_insertion_play_presentation(self)


__all__ = [
    "BladePoseDiagnosticObsCfg",
    "InsertionCriticObsCfg",
    "InsertionProprioObsCfg",
    "InsertionRgbObsCfg",
    "VisionInsertionObservationsCfg",
    "VisionInsertionPlayObservationsCfg",
    "VisionInsertionSceneCfg",
    "VisualRandomizationCfg",
    "ZeroGBladeVisionInsertionEnvCfg",
    "ZeroGBladeVisionInsertionPlayEnvCfg",
]
