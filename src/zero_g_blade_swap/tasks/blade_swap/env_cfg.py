"""Workcell constants shared by every task in this package.

This file used to hold the eight-phase blade-swap task: a reach, grasp, extract,
stow, acquire, align, insert, retreat state machine with its own commands,
rewards, terminations, and curriculum. It was deleted on 2026-08-10. Four of
those five servicing stages have no physics content — in simulation they are a
state machine — and the project's measurable result is contact mechanics, so
carrying them invited a mock-up reading of the work. See ``CLAUDE.md``.

What is left is what every surviving task genuinely shares: the arm's joint
names and the one PhysX configuration this 12 GB laptop was tuned for.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def make_simulation_cfg(render_interval: int) -> sim_utils.SimulationCfg:
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


__all__ = ["ARM_JOINTS", "make_simulation_cfg"]
