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
    InsertPolicyObsCfg,
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
class ZeroGBladeGrapplePinGraspNoisedEnvCfg(ZeroGBladeGrapplePinGraspEnvCfg, _EstimatorNoiseSettings):
    observations: NoisedGraspObservationsCfg = NoisedGraspObservationsCfg()


@configclass
class ZeroGBladeGrapplePinExtractNoisedEnvCfg(ZeroGBladeGrapplePinExtractEnvCfg, _EstimatorNoiseSettings):
    observations: NoisedExtractObservationsCfg = NoisedExtractObservationsCfg()


@configclass
class ZeroGBladeGrapplePinInsertNoisedEnvCfg(ZeroGBladeGrapplePinInsertEnvCfg, _EstimatorNoiseSettings):
    observations: NoisedInsertObservationsCfg = NoisedInsertObservationsCfg()


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
    "NoisedExtractObservationsCfg",
    "NoisedExtractPolicyObsCfg",
    "NoisedGraspObservationsCfg",
    "NoisedGrappleSkillObsCfg",
    "NoisedInsertObservationsCfg",
    "NoisedInsertPolicyObsCfg",
    "ZeroGBladeGrapplePinExtractNoisedEnvCfg",
    "ZeroGBladeGrapplePinExtractNoisedPlayEnvCfg",
    "ZeroGBladeGrapplePinGraspNoisedEnvCfg",
    "ZeroGBladeGrapplePinGraspNoisedPlayEnvCfg",
    "ZeroGBladeGrapplePinInsertNoisedEnvCfg",
    "ZeroGBladeGrapplePinInsertNoisedPlayEnvCfg",
]
