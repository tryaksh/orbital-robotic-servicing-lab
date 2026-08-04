"""Manager-based RL configurations for autonomous zero-g blade swapping."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .mdp.actions import ROBOTIQ_2F85_JOINT_NAMES, RobotiqBinaryActionCfg
from .scene_cfg import ZeroGBladeSwapSceneCfg, make_tiled_camera_cfg

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
OBSERVED_JOINTS = [*ARM_JOINTS, *ROBOTIQ_2F85_JOINT_NAMES]
ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=OBSERVED_JOINTS, preserve_order=True)
WRIST_CFG = SceneEntityCfg("robot", body_names=["wrist_3_link"])


@configclass
class ActionsCfg:
    """Six relative Cartesian DOFs plus one physical gripper DOF."""

    arm = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="wrist_3_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=mdp.TOOL_OFFSET_POS,
            rot=mdp.TOOL_OFFSET_ROT,
        ),
        scale=(0.035, 0.035, 0.035, 0.12, 0.12, 0.12),
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )
    gripper = RobotiqBinaryActionCfg(
        asset_name="robot",
        joint_names=ROBOTIQ_2F85_JOINT_NAMES,
        open_position=0.0,
        closed_position=0.785,
        threshold=0.0,
        close_on_positive=True,
    )


@configclass
class CommandsCfg:
    blade_goal = mdp.BladeSwapCommandCfg()


@configclass
class TeacherPolicyObsCfg(ObsGroup):
    """Compact state policy used to learn a high-quality teacher."""

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ROBOT_JOINT_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ROBOT_JOINT_CFG})
    gripper_state = ObsTerm(func=mdp.gripper_state)
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    blade_pose = ObsTerm(func=mdp.blade_pose_local, params={"asset_cfg": SceneEntityCfg("blade")})
    spare_blade_pose = ObsTerm(func=mdp.blade_pose_local, params={"asset_cfg": SceneEntityCfg("spare_blade")})
    blade_in_gripper = ObsTerm(
        func=mdp.blade_pose_in_end_effector_frame,
        params={"robot_cfg": WRIST_CFG, "blade_cfg": SceneEntityCfg("blade")},
    )
    spare_in_gripper = ObsTerm(
        func=mdp.blade_pose_in_end_effector_frame,
        params={"robot_cfg": WRIST_CFG, "blade_cfg": SceneEntityCfg("spare_blade")},
    )
    goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "blade_goal"})
    phase = ObsTerm(func=mdp.task_phase_one_hot, params={"command_name": "blade_goal"})
    mount_state = ObsTerm(func=mdp.robot_mount_state, params={"asset_cfg": SceneEntityCfg("robot")})
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ProprioObsCfg(ObsGroup):
    """Deployable non-visual state accompanying the RGB stream."""

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ROBOT_JOINT_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ROBOT_JOINT_CFG})
    gripper_state = ObsTerm(func=mdp.gripper_state)
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    phase = ObsTerm(func=mdp.task_phase_one_hot, params={"command_name": "blade_goal"})
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class RgbObsCfg(ObsGroup):
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
class CriticObsCfg(ObsGroup):
    """Privileged state for asymmetric actor-critic training."""

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ROBOT_JOINT_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ROBOT_JOINT_CFG})
    gripper_state = ObsTerm(func=mdp.gripper_state)
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    robot_root = ObsTerm(func=mdp.robot_root_pose_local, params={"asset_cfg": SceneEntityCfg("robot")})
    blade_pose = ObsTerm(func=mdp.both_blade_poses_local)
    blade_velocity = ObsTerm(func=mdp.both_blade_velocities, scale=0.1)
    blade_in_gripper = ObsTerm(
        func=mdp.blade_pose_in_end_effector_frame,
        params={"robot_cfg": WRIST_CFG, "blade_cfg": SceneEntityCfg("blade")},
    )
    spare_in_gripper = ObsTerm(
        func=mdp.blade_pose_in_end_effector_frame,
        params={"robot_cfg": WRIST_CFG, "blade_cfg": SceneEntityCfg("spare_blade")},
    )
    goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "blade_goal"})
    phase = ObsTerm(func=mdp.task_phase_one_hot, params={"command_name": "blade_goal"})
    previous_action = ObsTerm(func=mdp.last_action)
    randomized_physics = ObsTerm(func=mdp.randomized_physics_parameters)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class BladePoseDiagnosticObsCfg(ObsGroup):
    pose = ObsTerm(func=mdp.both_blade_poses_local)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TeacherObservationsCfg:
    """Exact top-level groups consumed by the teacher RL-Games config."""

    policy: TeacherPolicyObsCfg = TeacherPolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class VisionObservationsCfg:
    """Dictionary actor input plus privileged critic and diagnostics."""

    proprio: ProprioObsCfg = ProprioObsCfg()
    rgb: RgbObsCfg = RgbObsCfg()
    critic: CriticObsCfg = CriticObsCfg()
    blade_pose: BladePoseDiagnosticObsCfg = BladePoseDiagnosticObsCfg()


@configclass
class PhaseObsCfg(ObsGroup):
    value = ObsTerm(func=mdp.task_phase_one_hot, params={"command_name": "blade_goal"})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PlayObservationsCfg:
    """Teacher state and RGB together for synchronized demonstration capture."""

    policy: TeacherPolicyObsCfg = TeacherPolicyObsCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    rgb: RgbObsCfg = RgbObsCfg()
    critic: CriticObsCfg = CriticObsCfg()
    phase: PhaseObsCfg = PhaseObsCfg()


@configclass
class PhysicsRandomizationCfg:
    """Dynamics-gap randomization and reset logic."""

    blade_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("blade"),
            "mass_distribution_params": (5.0, 15.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    spare_blade_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("spare_blade"),
            "mass_distribution_params": (5.0, 15.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    slot_sliding_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("blade_slot"),
            "static_friction_range": (0.25, 2.00),
            "dynamic_friction_range": (0.20, 1.50),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    slot_left_guide_sliding_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("blade_slot_left_guide"),
            "static_friction_range": (0.25, 2.00),
            "dynamic_friction_range": (0.20, 1.50),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    slot_right_guide_sliding_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("blade_slot_right_guide"),
            "static_friction_range": (0.25, 2.00),
            "dynamic_friction_range": (0.20, 1.50),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_progress = EventTerm(func=mdp.reset_task_progress, mode="reset")
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.035, 0.035),
            "velocity_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        },
    )
    reset_blade = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.006, 0.006),
                "y": (-0.006, 0.006),
                "z": (-0.002, 0.002),
                "roll": (-0.015, 0.015),
                "pitch": (-0.015, 0.015),
                "yaw": (-0.02, 0.02),
            },
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.005, 0.005),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
            "asset_cfg": SceneEntityCfg("blade"),
        },
    )
    reset_spare_blade = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.006, 0.006), "y": (-0.004, 0.004), "z": (-0.002, 0.002), "yaw": (-0.02, 0.02)},
            "velocity_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)},
            "asset_cfg": SceneEntityCfg("spare_blade"),
        },
    )
    curriculum_initial_state = EventTerm(func=mdp.apply_curriculum_initial_state, mode="reset")
    randomize_rack_stiction = EventTerm(
        func=mdp.randomize_rail_stiction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("blade"),
            "breakaway_force_range": (10.0, 120.0),
            "viscous_coefficient_range": (2.0, 25.0),
        },
    )
    randomize_supply_stiction = EventTerm(
        func=mdp.randomize_rail_stiction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("spare_blade"),
            "breakaway_force_range": (10.0, 120.0),
            "viscous_coefficient_range": (2.0, 25.0),
        },
    )
    clear_mount_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    base_wobble_excitation = EventTerm(
        func=mdp.MountWobblePulse,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        is_global_time=False,
        params={
            "force_range": (-30.0, 30.0),
            "torque_range": (-6.0, 6.0),
            "duration_range_s": (0.2, 0.8),
            "quiet_range_s": (0.75, 2.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    rack_stiction = EventTerm(
        func=mdp.apply_rail_stiction,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        params={
            "asset_cfg": SceneEntityCfg("blade"),
            "rail_center_y": 0.0,
            "rail_center_z": 0.72,
            "engagement_x_range": (0.46, 1.0),
        },
    )
    supply_stiction = EventTerm(
        func=mdp.apply_rail_stiction,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        params={
            "asset_cfg": SceneEntityCfg("spare_blade"),
            "rail_center_y": -0.42,
            "rail_center_z": 0.50,
            "engagement_x_range": (0.13, 0.70),
        },
    )


@configclass
class VisualRandomizationCfg(PhysicsRandomizationCfg):
    """Replicator rack albedo and EventManager orbital-sun randomization."""

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
class RewardsCfg:
    reach_blade = RewTerm(
        func=mdp.distance_to_blade,
        weight=1.5,
        params={"robot_cfg": WRIST_CFG, "blade_cfg": SceneEntityCfg("blade"), "std": 0.18},
    )
    alignment = RewTerm(
        func=mdp.alignment_reward,
        weight=2.0,
        params={"robot_cfg": WRIST_CFG},
    )
    alignment_penalty = RewTerm(
        func=mdp.lateral_orientation_penalty,
        weight=-0.5,
        params={"robot_cfg": WRIST_CFG},
    )
    extraction_progress = RewTerm(func=mdp.extraction_progress, weight=2.0)
    insertion_progress = RewTerm(
        func=mdp.insertion_progress,
        weight=2.0,
        params={"command_name": "blade_goal", "position_std": 0.15, "rotation_std": 0.35},
    )
    stable_grasp = RewTerm(func=mdp.stable_grasp_reward, weight=3.0)
    phase_milestone = RewTerm(func=mdp.phase_completion_reward, weight=5.0)
    completed_insertion = RewTerm(
        func=mdp.insertion_completion_reward,
        weight=25.0,
        params={"command_name": "blade_goal"},
    )
    full_swap = RewTerm(func=mdp.full_swap_reward, weight=50.0, params={"command_name": "blade_goal"})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    blade_motion = RewTerm(func=mdp.blade_motion_penalty, weight=-0.015)
    workspace = RewTerm(func=mdp.workspace_violation_penalty, weight=-2.0)
    mount_deflection = RewTerm(func=mdp.mount_deflection_penalty, weight=-20.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    full_success = DoneTerm(
        func=mdp.blade_inserted,
        params={"command_name": "blade_goal"},
    )
    blade_lost = DoneTerm(func=mdp.blade_out_of_workspace)
    mount_unstable = DoneTerm(
        func=mdp.robot_mount_unstable,
        params={"maximum_displacement": 0.018, "maximum_rotation": 0.0436332313},
    )
    non_finite = DoneTerm(func=mdp.non_finite_state)


@configclass
class CurriculumCfg:
    task_stage = CurrTerm(
        func=mdp.SuccessRateStageCurriculum,
        params={
            "success_term": "full_success",
            "threshold": 0.70,
            "window_size": 100,
            "max_stage": 3,
        },
    )


def _simulation_cfg(render_interval: int) -> sim_utils.SimulationCfg:
    """Build one VRAM-conscious PhysX configuration for the 12 GB target."""

    return sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=render_interval,
        gravity=(0.0, 0.0, 0.0),
        use_fabric=True,
        enable_scene_query_support=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            solve_articulation_contact_last=True,
            enable_external_forces_every_iteration=True,
            enable_ccd=False,
            enable_stabilization=False,
            bounce_threshold_velocity=0.1,
            friction_offset_threshold=0.02,
            friction_correlation_distance=0.0125,
            gpu_max_rigid_contact_count=2**21,
            gpu_max_rigid_patch_count=2**18,
            gpu_found_lost_pairs_capacity=2**20,
            gpu_found_lost_aggregate_pairs_capacity=2**20,
            gpu_total_aggregate_pairs_capacity=2**20,
            gpu_collision_stack_size=2**25,
            gpu_heap_capacity=2**25,
            gpu_temp_buffer_capacity=2**23,
            gpu_max_num_partitions=8,
            gpu_max_soft_body_contacts=2**16,
            gpu_max_particle_contacts=2**16,
        ),
        render=sim_utils.RenderCfg(
            enable_translucency=False,
            enable_reflections=False,
            enable_global_illumination=False,
            antialiasing_mode="Off",
            rendering_mode="performance",
        ),
    )


@configclass
class ZeroGBladeSwapEnvCfg(ManagerBasedRLEnvCfg):
    """Common 30 Hz control task with 120 Hz zero-gravity physics."""

    scene: ZeroGBladeSwapSceneCfg = ZeroGBladeSwapSceneCfg(
        num_envs=1024,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: PhysicsRandomizationCfg = PhysicsRandomizationCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    sim: sim_utils.SimulationCfg = _simulation_cfg(render_interval=4)
    decimation: int = 4
    episode_length_s: float = 45.0
    is_finite_horizon: bool = False

    def __post_init__(self) -> None:
        # Contact reporting is automatically absent because no ContactSensor is
        # instantiated and every spawner sets activate_contact_sensors=False.
        # PhysX contact solving remains enabled for real Robotiq grasping.
        self.viewer.eye = (2.6, -3.0, 2.2)
        self.viewer.lookat = (0.45, 0.0, 0.70)


@configclass
class ZeroGBladeSwapTeacherEnvCfg(ZeroGBladeSwapEnvCfg):
    """1024 replicated, camera-free environments for fast teacher PPO."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.replicate_physics = True
        self.scene.clone_in_fabric = True
        self.scene.camera = None
        self.observations = TeacherObservationsCfg()
        self.events = PhysicsRandomizationCfg()
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 0


@configclass
class ZeroGBladeSwapVisionEnvCfg(ZeroGBladeSwapEnvCfg):
    """128 non-replicated 64x64 camera environments for vision PPO."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 128
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
        self.scene.camera = make_tiled_camera_cfg()
        self.observations = VisionObservationsCfg()
        self.events = VisualRandomizationCfg()
        # Eight physics steps per render exactly matches the 15 Hz camera.
        self.sim.render_interval = 8
        self.num_rerenders_on_reset = 1


@configclass
class ZeroGBladeSwapPlayEnvCfg(ZeroGBladeSwapVisionEnvCfg):
    """Small visual scene exposing teacher state and RGB simultaneously."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations = PlayObservationsCfg()
        self.viewer.eye = (2.2, -2.4, 1.8)
        self.viewer.lookat = (0.45, 0.0, 0.72)


__all__ = [
    "ZeroGBladeSwapEnvCfg",
    "ZeroGBladeSwapPlayEnvCfg",
    "ZeroGBladeSwapTeacherEnvCfg",
    "ZeroGBladeSwapVisionEnvCfg",
]
