"""Insertion when the policy's idea of where the slot is happens to be wrong.

This is the task the project pivoted to on 2026-08-10, and the reason is in one
sentence: every earlier task told the policy its exact pose error, so a scripted
controller would have solved it and RL could not demonstrate its value.

Here the slot is not where the policy thinks it is. Each episode displaces the
rack's guide rails laterally by an amount the actor is never told, and the goal
moves with them; the actor is given the error against the slot's *nominal*
position. The critic keeps ground truth, which is the asymmetric arrangement both
FORGE (arXiv 2408.04587) and arXiv 2604.19677 use. Contact against a rail is the
only channel that can reveal the displacement, and that is precisely the variable
this task exists to measure.

The displacement had to be physical rather than a number added to the reported
error, and ``mdp/uncertainty.py`` explains why at length: the blade is welded to
the tool, the tool pose is observed, so with a fixed goal the true error is an
exactly learnable function of an observation the actor already has, and an
injected bias would be recoverable and therefore fake.

Two profiles are registered and they differ in exactly one thing:

- :class:`ZeroGBladeUncertainInsertionEnvCfg` — the actor observes the contact
  wrench.
- :class:`ZeroGBladeUncertainInsertionBlindEnvCfg` — it does not.

Everything else is byte-identical: scene, physics, actions, rewards,
terminations, curriculum, and the PPO configuration. The repository has run this
experimental design once before, for force feedback against a matched control,
and it is what makes the resulting difference attributable to sensing rather
than to retraining. Both must be trained from scratch, because the observation
width differs and no checkpoint survives a change of observation width.

Three things are adopted rather than invented, and one is deliberately refused:

- IndustReal's sampling-based curriculum over the displacement magnitude, in
  ``mdp/uncertainty.py``. The hardest displacement is present from the first
  training step and the easy end is withdrawn as success improves, which is the
  direct fix for the "99.3% in 0.30 s" pathology in ``docs/status.md``.
- FORGE's force-threshold conditioning, replacing the two fixed quadratic
  penalty profiles that are already recorded here as ineffective.
- A 1 N noise floor on the observed contact force, which is what arXiv 2604.19677
  and FORGE both model. The earlier force observation was a perfect sensor.
- **Refused:** force-direction prediction (arXiv 2602.14174) and hybrid
  position/force control (arXiv 2604.19677). Both are action-space changes.
  Changing the action space and the observation in one experiment would make the
  result unattributable; they are the next experiment.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .assets import (
    BLADE_SIZE,
    INSERTION_STAGE_ARM_JOINT_POS,
    INSERTION_STAGE_BLADE_POSE,
    SLOT_ENTRY_LEFT_FLARE_CFG,
)
from .contact_insertion_env_cfg import ContactInsertionEventsCfg
from .force_limited_insertion_env_cfg import (
    CONTACT_FORCE_LIMIT_N,
    FORCE_OBSERVATION_FILTER_S,
    FORCE_OBSERVATION_SCALE_N,
    ForceLimitedInsertionTerminationsCfg,
    ZeroGBladeStrictForceLimitedInsertionEnvCfg,
)
from .guided_slot_env_cfg import _enable_channel
from .insertion_env_cfg import ARM_CFG, WRIST_CFG
from .rigid_grasp_insertion_env_cfg import RigidGraspInsertionRewardsCfg
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .scene_cfg import ZeroGGuidedSlotSceneCfg

# ---------------------------------------------------------------------------
# Where the channel has to be for the uncertainty to be physical.
#
# The certified slot runs from x = 0.45 to 1.05 and the blade is 450 mm long, so
# at every reset distance the promoted tasks use — including the furthest — the
# blade already sits inside the rails, its front face 358 mm past the mouth. The
# rails leave 0.75 mm of clearance per side. Displacing them by more than that
# would therefore start the episode with the blade *inside* a rail, which is the
# impossible interpenetrating pose ``assets.py`` warns about, and displacing them
# by less than that is smaller than the 2.5 mm lateral success tolerance and so
# measures nothing.
#
# The channel is therefore moved downstream until its lead-in begins ahead of
# where the blade starts. Everything below is derived, not chosen:
#
#   blade front face at the staging reset  0.5829 + 0.225      = 0.8079 m
#   flare opening rate, 12 degrees         tan(12 deg)          = 0.2126 m/m
#   opening needed for a 4 mm offset       80 + 4 - 80.75       = 3.25 mm
#   minimum lead-in ahead of the blade     3.25 / 0.2126        = 15.3 mm
#
# 32 mm is used instead of the minimum 15.3 mm, which leaves 3.6 mm of clearance
# between the blade edge and the flare face at the full trained displacement.
BLADE_FRONT_AT_STAGING_X = INSERTION_STAGE_BLADE_POSE[2][0] + 0.5 * BLADE_SIZE[0]
LEAD_IN_AHEAD_OF_BLADE_M = 0.032
CHANNEL_MOUTH_X = BLADE_FRONT_AT_STAGING_X + LEAD_IN_AHEAD_OF_BLADE_M
# The far end stays where the certified rails end, so the channel still runs past
# the seated blade's front face at 0.975 m.
CHANNEL_FAR_X = 1.05
CHANNEL_LENGTH_M = CHANNEL_FAR_X - CHANNEL_MOUTH_X
CHANNEL_CENTRE_X = 0.5 * (CHANNEL_MOUTH_X + CHANNEL_FAR_X)
#: Distance the whole assembly moves. The flares keep their exact 12-degree
#: relationship to the mouth by being translated rather than re-derived.
SLOT_TRANSLATION_X = CHANNEL_MOUTH_X - 0.45
FLARE_CENTRE_X = SLOT_ENTRY_LEFT_FLARE_CFG.init_state.pos[0] + SLOT_TRANSLATION_X

#: Assets whose length is trimmed to the shortened channel.
CHANNEL_ASSET_NAMES = (
    "blade_slot",
    "blade_slot_left_guide",
    "blade_slot_right_guide",
    "blade_slot_upper_left_lip",
    "blade_slot_upper_right_lip",
)
FLARE_ASSET_NAMES = ("blade_slot_entry_left_flare", "blade_slot_entry_right_flare")


def _relocate_channel(scene) -> None:
    """Trim and move the channel so its lead-in starts ahead of the blade.

    Written absolutely rather than incrementally, because the rigid-grasp parent
    rebuilds the floor and side rails on every ``configure_robustness`` call and
    this therefore runs more than once.
    """

    for name in CHANNEL_ASSET_NAMES:
        asset = getattr(scene, name)
        width, height = asset.spawn.size[1], asset.spawn.size[2]
        asset.spawn.size = (CHANNEL_LENGTH_M, width, height)
        asset.init_state.pos = (CHANNEL_CENTRE_X, asset.init_state.pos[1], asset.init_state.pos[2])
    for name in FLARE_ASSET_NAMES:
        asset = getattr(scene, name)
        asset.init_state.pos = (FLARE_CENTRE_X, asset.init_state.pos[1], asset.init_state.pos[2])


@configclass
class UncertainSlotSceneCfg(ZeroGGuidedSlotSceneCfg):
    """The guided channel, plus the contact reporting the experiment needs.

    The lead-in flares are not decoration here, they are the mechanism. A
    square-edged channel would meet an offset blade with its end face, and the
    resulting contact points straight back along the approach axis whichever side
    the blade is off. A 12-degree flare tilts that normal, so the contact force
    carries the sign of the error, which is the only thing a force-aware policy
    can use and a force-blind one cannot.
    """

    blade_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SpareBlade",
        update_period=0.0,
        history_length=0,
    )


# How wrong the pose estimate is allowed to be during training, in metres.
#
# Grounded in this task's own tolerances rather than copied: the insertion
# success predicate allows 2.5 mm of lateral error and the Level-2 rails leave
# 1.5 mm of side clearance, so a 4 mm belief error is already larger than the
# entire lateral tolerance and about 2.7 times the physical clearance. For
# comparison, FORGE trains at sigma 2.5 mm with a 5 mm ceiling and arXiv
# 2604.19677 trains at sigma 1 mm before evaluating out to 7.5 mm, both against
# geometry looser than this one.
BELIEF_BIAS_CEILING_M = 0.004
# IndustReal moves its bound 5 mm up and 3 mm down against a 20 mm range. These
# keep that 5:3 ratio against this task's 4 mm range.
BELIEF_BIAS_INCREASE_M = 0.0005
BELIEF_BIAS_DECREASE_M = 0.0003

# Estimator jitter. Small on purpose: it is not the interesting term, because a
# policy averages it away in a few control steps. It exists so the belief is not
# a suspiciously perfect constant.
BELIEF_POSITION_JITTER_M = 0.0005
BELIEF_ORIENTATION_JITTER_RAD = 0.005

# FORGE samples a maximum allowable force per episode and conditions the policy
# on it. This range is that mechanism with this workcell's measured numbers: the
# promoted Level-2 policy's peak contact force is 4.7 N median and 16.6 N p95,
# so 5 N is tighter than typical contact and 20 N is looser than p95.
FORCE_THRESHOLD_MINIMUM_N = 5.0
FORCE_THRESHOLD_MAXIMUM_N = 20.0

# A real wrist force/torque sensor is not noiseless. Both FORGE and arXiv
# 2604.19677 model roughly 1 N; the observation this repository certified
# earlier was an idealized sensor and said so.
CONTACT_FORCE_NOISE_N = 1.0


@configclass
class UncertainInsertionActorObsCfg(ObsGroup):
    """What a policy could plausibly know, and nothing more.

    ``belief_error`` replaces ``insertion_goal_error``. That single substitution
    is the pivot: everything else on this list was already available to the
    promoted policies.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ARM_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ARM_CFG})
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    belief_error = ObsTerm(
        func=mdp.BeliefPoseErrorObservation,
        params={
            "position_jitter_m": BELIEF_POSITION_JITTER_M,
            "orientation_jitter_rad": BELIEF_ORIENTATION_JITTER_RAD,
        },
    )
    blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)
    mount_state = ObsTerm(func=mdp.robot_mount_state)
    force_threshold = ObsTerm(
        func=mdp.ContactForceThresholdObservation,
        params={
            "minimum_n": FORCE_THRESHOLD_MINIMUM_N,
            "maximum_n": FORCE_THRESHOLD_MAXIMUM_N,
            "force_scale_n": FORCE_OBSERVATION_SCALE_N,
        },
    )
    contact_wrench = ObsTerm(
        func=mdp.BladeContactWrenchObservation,
        params={
            "force_scale_n": FORCE_OBSERVATION_SCALE_N,
            "filter_time_constant_s": FORCE_OBSERVATION_FILTER_S,
            "noise_std_n": CONTACT_FORCE_NOISE_N,
        },
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UncertainInsertionBlindActorObsCfg(UncertainInsertionActorObsCfg):
    """The ablation. One term removed; nothing else may differ."""

    contact_wrench = None


@configclass
class UncertainInsertionCriticObsCfg(ObsGroup):
    """Privileged state, including the pose error the actor is denied.

    Asymmetric actor-critic is what makes this trainable: the value function can
    tell a good action taken under a bad belief from a bad action, which the
    actor's own observation cannot.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": ARM_CFG})
    joint_vel = ObsTerm(func=mdp.normalized_joint_velocity, params={"asset_cfg": ARM_CFG})
    end_effector = ObsTerm(func=mdp.end_effector_pose_local, params={"asset_cfg": WRIST_CFG})
    true_goal_error = ObsTerm(func=mdp.insertion_goal_error)
    blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)
    mount_state = ObsTerm(func=mdp.robot_mount_state)
    robot_root = ObsTerm(func=mdp.robot_root_pose_local)
    randomized_physics = ObsTerm(func=mdp.randomized_physics_parameters)
    force_threshold = ObsTerm(
        func=mdp.ContactForceThresholdObservation,
        params={
            "minimum_n": FORCE_THRESHOLD_MINIMUM_N,
            "maximum_n": FORCE_THRESHOLD_MAXIMUM_N,
            "force_scale_n": FORCE_OBSERVATION_SCALE_N,
        },
    )
    contact_wrench = ObsTerm(
        func=mdp.BladeContactWrenchObservation,
        params={
            "force_scale_n": FORCE_OBSERVATION_SCALE_N,
            "filter_time_constant_s": FORCE_OBSERVATION_FILTER_S,
            # The critic reads the sensor without its noise floor, matching
            # arXiv 2604.19677's noiseless critic variants.
            "noise_std_n": 0.0,
        },
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UncertainInsertionObservationsCfg:
    policy: UncertainInsertionActorObsCfg = UncertainInsertionActorObsCfg()
    critic: UncertainInsertionCriticObsCfg = UncertainInsertionCriticObsCfg()


@configclass
class UncertainInsertionBlindObservationsCfg:
    policy: UncertainInsertionBlindActorObsCfg = UncertainInsertionBlindActorObsCfg()
    critic: UncertainInsertionCriticObsCfg = UncertainInsertionCriticObsCfg()


# One reset distance, not three. With the channel moved downstream, the two
# nearer curriculum stages would start the blade inside the rails again, which is
# the interpenetration this whole construction exists to avoid. The staging pose
# is the only one that begins in the lead-in, so it is the only one used, and the
# stage curriculum is collapsed to a single level rather than left reporting
# three identical ones.
SINGLE_STAGE_ARM_JOINT_POS = (INSERTION_STAGE_ARM_JOINT_POS[2],)
SINGLE_STAGE_BLADE_POSE = (INSERTION_STAGE_BLADE_POSE[2],)
SINGLE_STAGE_RESET_NOISE = (0.004,)
SINGLE_STAGE_MIXTURE = ((1.0,),)


@configclass
class UncertainInsertionEventsCfg(ContactInsertionEventsCfg):
    """The rigid-grasp reset, plus the one that makes the task uncertain.

    Ordering matters and is why this runs at ``mode="reset"`` alongside the
    other reset terms: the slot has to be in its new place before the first
    observation of the episode is computed. The goal pose follows it inside
    ``InsertionGoalCommand``, which recomputes every step rather than depending
    on whether events or commands reset first.
    """

    displace_slot = EventTerm(func=mdp.randomize_slot_offset, mode="reset")


@configclass
class UncertainInsertionRewardsCfg(RigidGraspInsertionRewardsCfg):
    """The certified reward set, with FORGE's hinge replacing the fixed penalty."""

    # The quadratic profile at two strengths is recorded in docs/status.md as
    # having moved mean contact by 2.6% and impulse not at all. Reaching for a
    # steeper version of the same curve is the one thing already known to fail.
    contact_force = None
    force_threshold = RewTerm(
        func=mdp.force_threshold_penalty,
        weight=-3.0,
        params={"force_scale_n": FORCE_OBSERVATION_SCALE_N},
    )


@configclass
class UncertainInsertionCurriculumCfg:
    """One real axis, and one collapsed to a single level.

    Reset distance is fixed at the staging pose because the moved channel leaves
    no room for the nearer stages, so its curriculum term survives only to keep
    the evaluation tooling and the recorded stage column working. The
    displacement gets IndustReal's sampling-based curriculum, because it is the
    axis that carries the difficulty and the one where the
    overfitting-to-an-easy-reset pathology would otherwise bite.
    """

    pose_noise = CurrTerm(
        func=mdp.InsertionSuccessRateCurriculum,
        params={
            "success_term": "insertion_success",
            "threshold": 0.80,
            "window_size": 2_000,
            "max_stage": 0,
            "minimum_level_steps": 1_600,
            "stage_mixtures": SINGLE_STAGE_MIXTURE,
        },
    )
    belief_bias = CurrTerm(
        func=mdp.BeliefSamplingCurriculum,
        params={
            "success_term": "insertion_success",
            "bias_ceiling_m": BELIEF_BIAS_CEILING_M,
            "increase_m": BELIEF_BIAS_INCREASE_M,
            "decrease_m": BELIEF_BIAS_DECREASE_M,
            "window_size": 2_000,
            "minimum_level_steps": 1_600,
        },
    )


@configclass
class ZeroGBladeUncertainInsertionEnvCfg(ZeroGBladeStrictForceLimitedInsertionEnvCfg):
    """Force-aware insertion under a wrong pose belief.

    Inherits the strict force-limited task's scene, physics, actions, and
    terminations, so the contact sensor, the non-Fabric cloning it requires, and
    the 30 N training abort all come along unchanged.
    """

    scene: UncertainSlotSceneCfg = UncertainSlotSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        # Fabric-cloned prims do not carry the contact-report API, and the whole
        # experiment rests on that sensor.
        clone_in_fabric=False,
    )
    observations: UncertainInsertionObservationsCfg = UncertainInsertionObservationsCfg()
    events: UncertainInsertionEventsCfg = UncertainInsertionEventsCfg()
    rewards: UncertainInsertionRewardsCfg = UncertainInsertionRewardsCfg()
    curriculum: UncertainInsertionCurriculumCfg = UncertainInsertionCurriculumCfg()

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # Every ancestor rebuilds the event set from its own class, dropping the
        # displacement term. Copy the configured result onto a subclass that
        # declares it rather than replaying each ancestor's decisions here.
        events = UncertainInsertionEventsCfg()
        for name, value in self.events.__dict__.items():
            setattr(events, name, value)
        self.events = events
        # One reset distance, and it must survive the parent's rebuild.
        self.events.reset_arm.params["poses_by_stage"] = SINGLE_STAGE_ARM_JOINT_POS
        self.events.reset_arm.params["noise_by_stage"] = SINGLE_STAGE_RESET_NOISE
        self.events.reset_blade.params["poses_by_stage"] = SINGLE_STAGE_BLADE_POSE
        # The parent rebuilds the floor and rails for the level and knows nothing
        # about the channel or where it now sits.
        _enable_channel(self.scene)
        _relocate_channel(self.scene)


@configclass
class ZeroGBladeUncertainInsertionBlindEnvCfg(ZeroGBladeUncertainInsertionEnvCfg):
    """The matched control: identical in every way except it cannot feel."""

    observations: UncertainInsertionBlindObservationsCfg = UncertainInsertionBlindObservationsCfg()


@configclass
class ZeroGBladeUncertainInsertionPlayEnvCfg(ZeroGBladeUncertainInsertionEnvCfg):
    """Evaluation profile, judged under the shared 60 N abort limit.

    Every force policy this project has published was measured at 60 N. Aborting
    at the 30 N training limit here would truncate the force distribution being
    compared and manufacture an improvement out of a termination rule.
    """

    terminations: ForceLimitedInsertionTerminationsCfg = ForceLimitedInsertionTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeUncertainInsertionBlindPlayEnvCfg(ZeroGBladeUncertainInsertionBlindEnvCfg):
    terminations: ForceLimitedInsertionTerminationsCfg = ForceLimitedInsertionTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "BELIEF_BIAS_CEILING_M",
    "BELIEF_BIAS_DECREASE_M",
    "BELIEF_BIAS_INCREASE_M",
    "BELIEF_ORIENTATION_JITTER_RAD",
    "BELIEF_POSITION_JITTER_M",
    "CONTACT_FORCE_LIMIT_N",
    "CONTACT_FORCE_NOISE_N",
    "FORCE_THRESHOLD_MAXIMUM_N",
    "FORCE_THRESHOLD_MINIMUM_N",
    "UncertainInsertionActorObsCfg",
    "UncertainInsertionBlindActorObsCfg",
    "UncertainInsertionCriticObsCfg",
    "UncertainInsertionCurriculumCfg",
    "CHANNEL_CENTRE_X",
    "CHANNEL_LENGTH_M",
    "CHANNEL_MOUTH_X",
    "FLARE_CENTRE_X",
    "SLOT_TRANSLATION_X",
    "UncertainInsertionEventsCfg",
    "UncertainSlotSceneCfg",
    "UncertainInsertionRewardsCfg",
    "ZeroGBladeUncertainInsertionBlindEnvCfg",
    "ZeroGBladeUncertainInsertionBlindPlayEnvCfg",
    "ZeroGBladeUncertainInsertionEnvCfg",
    "ZeroGBladeUncertainInsertionPlayEnvCfg",
]
