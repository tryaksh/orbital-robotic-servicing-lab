"""The grapple skills, trained on the estimator's error instead of on exact state.

**One measurement motivates every line of this file.** The chain scores 20/24 on
the vision task when the module pose comes from the simulator and 4/24 when the
same code path reads it from the cameras, with the task, the seeds, the
environment count, the checkpoints, the guard and every observation term held
fixed. Thirteen of the twenty-four camera-driven episodes time out in extraction
and never engage the form lock; the module is frequently lost outright. That is
not the estimator being inaccurate -- its error on healthy episodes is about
2 mm -- it is three policies meeting an error distribution that was not in their
training data.

These registrations put it there. They are separate tasks, not an option on the
existing ones, for the reason this repository keeps every losing arm: the
published skill certificates were measured on exact state, and a policy trained
against a noised observation is a different policy answering a different
question. Both stay runnable, and the difference between them is the result.

What changes, and only this:

* the four module-derived observation terms read
  :class:`~.mdp.estimator_surrogate.SurrogateModuleStateEstimator` instead of
  live simulator state, at the deployed camera period, with the certified
  residual and the certified miss rate, and with the velocity the deployed
  estimator's own differencing manufactures;
* nothing else. Rewards, terminations, tolerances, the curriculum, the load
  path, the action space, the reset distribution and the phase budgets are the
  ones the published certificates used.

Rewards and terminations deliberately keep exact state. Noising the *reward*
would change what the task is; noising the *observation* changes only what the
policy is allowed to see, which is the question being asked. The critic sees
what the actor sees, so no privileged value function is being trained here
either -- that would be a distillation result and would have to be labelled one.

The robot's own joint positions, joint velocities and end-effector pose stay
exact in every one of these tasks. They are encoder and forward-kinematics
information; a real servicer has them.
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from zero_g_blade_swap.servicing_camera import CAMERA_UPDATE_PERIOD_S

from . import mdp
from .grapple_pin_env_cfg import (
    ExtractPolicyObsCfg,
    GrappleSkillObsCfg,
    ZeroGBladeGrapplePinExtractEnvCfg,
    ZeroGBladeGrapplePinGraspEnvCfg,
    ZeroGBladeGrapplePinInsertEnvCfg,
)
from .mdp.estimator_surrogate import DEFAULT_ESTIMATOR_CERTIFICATION
from .robust_insertion_env_cfg import configure_insertion_play_presentation

# The deployed velocity filter. Repeated here rather than imported from a vision
# config so a state task never has to pull in a camera scene to be constructed;
# the two are asserted equal by ``tests/test_estimator_surrogate_contract.py``.
DEPLOYED_VELOCITY_FILTER_TIME_CONSTANT_S = 0.10


@configclass
class NoisedGrappleSkillObsCfg(GrappleSkillObsCfg):
    """What all three skills see once the module state comes from an estimator."""

    grip_error = ObsTerm(func=mdp.NoisedGraspError)
    blade_velocity = ObsTerm(func=mdp.NoisedModuleVelocity, scale=0.10)


@configclass
class NoisedGraspObservationsCfg:
    policy: NoisedGrappleSkillObsCfg = NoisedGrappleSkillObsCfg()


@configclass
class NoisedExtractPolicyObsCfg(NoisedGrappleSkillObsCfg):
    remaining_travel = ObsTerm(func=mdp.NoisedExtractionRemaining)


@configclass
class NoisedExtractObservationsCfg:
    policy: NoisedExtractPolicyObsCfg = NoisedExtractPolicyObsCfg()


@configclass
class NoisedInsertPolicyObsCfg(NoisedGrappleSkillObsCfg):
    blade_goal_error = ObsTerm(func=mdp.NoisedInsertionGoalError)


@configclass
class NoisedInsertObservationsCfg:
    policy: NoisedInsertPolicyObsCfg = NoisedInsertPolicyObsCfg()


class _EstimatorNoiseSettings:
    """The three fields the surrogate reads, with the deployed values.

    Held in one place because a task that noises two of the three is not a
    partial answer, it is an uninterpretable one.
    """

    estimator_noise_certification: str = DEFAULT_ESTIMATOR_CERTIFICATION
    estimator_noise_camera_period_s: float = CAMERA_UPDATE_PERIOD_S
    perception_velocity_filter_time_constant_s: float = DEPLOYED_VELOCITY_FILTER_TIME_CONSTANT_S


@configclass
class PoseOnlyNoisedExtractObsCfg(NoisedExtractPolicyObsCfg):
    """The pose channels carry the estimator's error; the velocity is exact.

    **This is one half of the experiment that says which channel does the
    damage**, and it needs no training and no cameras. Thirteen of twenty-four
    camera-driven episodes time out in extraction, and the two candidate causes
    are the pose residual itself and the velocity the estimator manufactures by
    differencing a staircase. Held apart, each is measurable on an unchanged
    checkpoint.
    """

    blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)


@configclass
class PoseOnlyNoisedExtractObservationsCfg:
    policy: PoseOnlyNoisedExtractObsCfg = PoseOnlyNoisedExtractObsCfg()


@configclass
class VelocityOnlyNoisedExtractObsCfg(ExtractPolicyObsCfg):
    """The velocity is the estimator's; every pose channel is exact.

    The other half. The surrogate still runs, so the velocity is differenced and
    filtered over the same sample-and-hold staircase the deployed estimator
    produces -- which is the property that makes this channel's noise floor
    twenty times a seated module's own speed.
    """

    blade_velocity = ObsTerm(func=mdp.NoisedModuleVelocity, scale=0.10)


@configclass
class VelocityOnlyNoisedExtractObservationsCfg:
    policy: VelocityOnlyNoisedExtractObsCfg = VelocityOnlyNoisedExtractObsCfg()


@configclass
class ZeroGBladeGrapplePinGraspNoisedEnvCfg(ZeroGBladeGrapplePinGraspEnvCfg, _EstimatorNoiseSettings):
    observations: NoisedGraspObservationsCfg = NoisedGraspObservationsCfg()


@configclass
class ZeroGBladeGrapplePinExtractNoisedEnvCfg(ZeroGBladeGrapplePinExtractEnvCfg, _EstimatorNoiseSettings):
    observations: NoisedExtractObservationsCfg = NoisedExtractObservationsCfg()


@configclass
class ZeroGBladeGrapplePinInsertNoisedEnvCfg(ZeroGBladeGrapplePinInsertEnvCfg, _EstimatorNoiseSettings):
    observations: NoisedInsertObservationsCfg = NoisedInsertObservationsCfg()


@configclass
class ZeroGBladeGrapplePinExtractPoseNoisedEnvCfg(ZeroGBladeGrapplePinExtractEnvCfg, _EstimatorNoiseSettings):
    observations: PoseOnlyNoisedExtractObservationsCfg = PoseOnlyNoisedExtractObservationsCfg()


@configclass
class ZeroGBladeGrapplePinExtractVelocityNoisedEnvCfg(ZeroGBladeGrapplePinExtractEnvCfg, _EstimatorNoiseSettings):
    observations: VelocityOnlyNoisedExtractObservationsCfg = VelocityOnlyNoisedExtractObservationsCfg()


@configclass
class ZeroGBladeGrapplePinExtractPoseNoisedPlayEnvCfg(ZeroGBladeGrapplePinExtractPoseNoisedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeGrapplePinExtractVelocityNoisedPlayEnvCfg(ZeroGBladeGrapplePinExtractVelocityNoisedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


class _InLoopTailSettings(_EstimatorNoiseSettings):
    """The estimator's in-loop tail, as its own arm.

    The certified residual is measured on held-out still frames, where position
    error tops out at 3.3 mm. In the loop the same estimator records a
    per-episode maximum around 30 mm, and it does so on the episodes that
    *succeed* as much as on the ones that fail -- 1.89, 2.00 and 2.11 mm of mean
    error on the winners against 2.29, 5.99 and 1.98 on the losers, across the
    three vision seeds. So the tail is a property of the poses a closed loop
    visits, not a symptom of an episode already going wrong.

    The two constants are calibrated against those two recorded numbers and
    nothing else. With the bulk sigma at 0.683 mm and roughly 950 camera frames
    an episode, a 3% outlier rate at 15x the bulk gives a mean around 1.5 mm and
    a per-episode maximum around 32 mm, against the 2.0 mm and 30 mm the loop
    records. That is calibration against deployment rather than derivation from a
    certificate, and it is the reason this is a separate task: the first arm's
    constants are inverted from published evidence and this one's are fitted to
    it, and a reader is entitled to know which is which.
    """

    estimator_noise_outlier_rate: float = 0.03
    estimator_noise_outlier_scale: float = 15.0


@configclass
class ZeroGBladeGrapplePinExtractNoisedTailEnvCfg(ZeroGBladeGrapplePinExtractEnvCfg, _InLoopTailSettings):
    observations: NoisedExtractObservationsCfg = NoisedExtractObservationsCfg()


@configclass
class ZeroGBladeGrapplePinExtractNoisedTailPlayEnvCfg(ZeroGBladeGrapplePinExtractNoisedTailEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeGrapplePinGraspNoisedPlayEnvCfg(ZeroGBladeGrapplePinGraspNoisedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeGrapplePinExtractNoisedPlayEnvCfg(ZeroGBladeGrapplePinExtractNoisedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeGrapplePinInsertNoisedPlayEnvCfg(ZeroGBladeGrapplePinInsertNoisedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "DEPLOYED_VELOCITY_FILTER_TIME_CONSTANT_S",
    "PoseOnlyNoisedExtractObsCfg",
    "PoseOnlyNoisedExtractObservationsCfg",
    "VelocityOnlyNoisedExtractObsCfg",
    "VelocityOnlyNoisedExtractObservationsCfg",
    "ZeroGBladeGrapplePinExtractPoseNoisedEnvCfg",
    "ZeroGBladeGrapplePinExtractPoseNoisedPlayEnvCfg",
    "ZeroGBladeGrapplePinExtractVelocityNoisedEnvCfg",
    "ZeroGBladeGrapplePinExtractVelocityNoisedPlayEnvCfg",
    "NoisedExtractObservationsCfg",
    "NoisedExtractPolicyObsCfg",
    "NoisedGraspObservationsCfg",
    "NoisedGrappleSkillObsCfg",
    "NoisedInsertObservationsCfg",
    "NoisedInsertPolicyObsCfg",
    "ZeroGBladeGrapplePinExtractNoisedEnvCfg",
    "ZeroGBladeGrapplePinExtractNoisedTailEnvCfg",
    "ZeroGBladeGrapplePinExtractNoisedTailPlayEnvCfg",
    "ZeroGBladeGrapplePinExtractNoisedPlayEnvCfg",
    "ZeroGBladeGrapplePinGraspNoisedEnvCfg",
    "ZeroGBladeGrapplePinGraspNoisedPlayEnvCfg",
    "ZeroGBladeGrapplePinInsertNoisedEnvCfg",
    "ZeroGBladeGrapplePinInsertNoisedPlayEnvCfg",
]
