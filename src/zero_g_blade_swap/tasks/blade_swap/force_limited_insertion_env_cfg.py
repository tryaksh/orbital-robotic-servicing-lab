"""Insertion that must also respect a contact-force budget.

Level 2 reaches 100% success while its worst-case contact force varies about
sevenfold with approach length, because nothing in the task ever asked for a
gentle insertion. This profile adds the missing objective: contact load above a
free allowance is penalised, and an episode that exceeds the force limit aborts
instead of pushing through.

Observations and actions are byte-identical to the rigid-grasp task, so a
promoted Level-2 checkpoint can be fine-tuned here rather than retrained. The
policy does not observe contact force; it has to learn a gentler motion, which
is what a wrench-limited real controller would demand of it.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .rigid_grasp_insertion_env_cfg import (
    RigidGraspInsertionRewardsCfg,
    ZeroGBladeRigidGraspInsertionEnvCfg,
)
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .scene_cfg import ZeroGRigidGraspInsertionSceneCfg

# Measured Level-2 contact load, pooled over 4,513 successful episodes, was
# p95 16.6 N with a 66.4 N worst case. The free allowance sits just above the
# median so ordinary rail contact stays unpenalised, and the abort limit sits
# just under the measured worst case so the tail is what has to change.
FREE_CONTACT_FORCE_N = 5.0
CONTACT_FORCE_SCALE_N = 20.0
CONTACT_FORCE_LIMIT_N = 60.0

# The mild profile above turned out to shape only the extreme tail: at the
# measured median contact of 4.7 N its penalty is exactly zero, and at p95 it is
# 0.17 per step against a success reward of 35, so PPO had almost no reason to
# push back. The strict profile below charges 0.85 at the median and 19.5 at
# p95, which is the same order as the success term, and is the other end of the
# success-versus-force trade-off.
STRICT_FREE_CONTACT_FORCE_N = 1.5
STRICT_CONTACT_FORCE_SCALE_N = 6.0
STRICT_CONTACT_FORCE_LIMIT_N = 30.0


@configclass
class ForceLimitedInsertionSceneCfg(ZeroGRigidGraspInsertionSceneCfg):
    """Rigid-grasp scene that reports blade contacts to the MDP."""

    blade_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SpareBlade",
        update_period=0.0,
        history_length=0,
    )


@configclass
class ForceLimitedInsertionRewardsCfg(RigidGraspInsertionRewardsCfg):
    contact_force = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-0.5,
        params={"free_force_n": FREE_CONTACT_FORCE_N, "force_scale_n": CONTACT_FORCE_SCALE_N},
    )
    force_abort = RewTerm(func=mdp.excessive_contact_force_reward, weight=-15.0)


@configclass
class ForceLimitedInsertionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insertion_success = DoneTerm(func=mdp.contact_insertion_success_mask)
    insertion_failed = DoneTerm(
        func=mdp.insertion_failure,
        params={"grasp_position_limit": 0.025, "grasp_orientation_limit": 0.25},
    )
    excessive_contact_force = DoneTerm(
        func=mdp.excessive_contact_force,
        params={"force_limit_n": CONTACT_FORCE_LIMIT_N},
    )
    mount_unstable = DoneTerm(func=mdp.robot_mount_unstable)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class ZeroGBladeForceLimitedInsertionEnvCfg(ZeroGBladeRigidGraspInsertionEnvCfg):
    """Rigid-grasp insertion under a simulated wrench limit."""

    scene: ForceLimitedInsertionSceneCfg = ForceLimitedInsertionSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        # Fabric-cloned prims do not carry the contact-report API, so the
        # sensor would resolve zero bodies on every clone but env 0.
        clone_in_fabric=False,
    )
    rewards: ForceLimitedInsertionRewardsCfg = ForceLimitedInsertionRewardsCfg()
    terminations: ForceLimitedInsertionTerminationsCfg = ForceLimitedInsertionTerminationsCfg()

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # The parent rebuilds the blade config, which drops the contact-report
        # flag the sensor needs.
        self.scene.spare_blade.spawn.activate_contact_sensors = True


@configclass
class ZeroGBladeForceLimitedInsertionPlayEnvCfg(ZeroGBladeForceLimitedInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class StrictForceLimitedInsertionRewardsCfg(ForceLimitedInsertionRewardsCfg):
    contact_force = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-3.0,
        params={"free_force_n": STRICT_FREE_CONTACT_FORCE_N, "force_scale_n": STRICT_CONTACT_FORCE_SCALE_N},
    )


@configclass
class StrictForceLimitedInsertionTerminationsCfg(ForceLimitedInsertionTerminationsCfg):
    excessive_contact_force = DoneTerm(
        func=mdp.excessive_contact_force,
        params={"force_limit_n": STRICT_CONTACT_FORCE_LIMIT_N},
    )


@configclass
class ZeroGBladeStrictForceLimitedInsertionEnvCfg(ZeroGBladeForceLimitedInsertionEnvCfg):
    """The aggressive end of the success-versus-contact-force trade-off."""

    rewards: StrictForceLimitedInsertionRewardsCfg = StrictForceLimitedInsertionRewardsCfg()
    terminations: StrictForceLimitedInsertionTerminationsCfg = StrictForceLimitedInsertionTerminationsCfg()


@configclass
class ZeroGBladeStrictForceLimitedInsertionPlayEnvCfg(ZeroGBladeStrictForceLimitedInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "CONTACT_FORCE_LIMIT_N",
    "STRICT_CONTACT_FORCE_LIMIT_N",
    "ZeroGBladeStrictForceLimitedInsertionEnvCfg",
    "ZeroGBladeStrictForceLimitedInsertionPlayEnvCfg",
    "ForceLimitedInsertionRewardsCfg",
    "ForceLimitedInsertionTerminationsCfg",
    "ZeroGBladeForceLimitedInsertionEnvCfg",
    "ZeroGBladeForceLimitedInsertionPlayEnvCfg",
]
