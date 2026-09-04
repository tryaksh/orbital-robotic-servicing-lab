"""Train a skill against the estimator it is deployed on, without rendering a frame.

**This exists because the 67-point perception gap was measured as a substitution
rather than inferred.** The same vision task, the same three held-out seeds, the
same eight environments, the same checkpoints, the same guard and the same
observation terms score 20/24 when the module pose comes from the simulator and
4/24 when it comes from the cameras. One term differs. So the loss is the
estimator -- and it is not the estimator being bad, because its own error on
healthy episodes is about 2 mm and it is certified on held-out frames against
unchanged gates. It is a policy meeting an error distribution it never trained
on. The literature calls the same number: a state-privileged teacher at 98%
distils to a vision student at 73% in *Residual RL for Precise Assembly*
(arXiv 2407.16677), and that is with the transfer done properly.

Doing it properly here means putting the estimator's error into the skills'
training distribution. Doing it with the real cameras does not fit: the skills
need thousands of epochs and the vision task renders. So this module samples the
estimator's *statistics* against simulator state, and every constant it uses is
read out of the estimator's own certification rather than chosen --
:class:`~zero_g_blade_swap.estimator_noise.EstimatorNoiseModel` does that
inversion and is CPU-tested.

Four properties of the deployed estimator are reproduced, and they are not all
the same kind of error:

1. **Sample and hold.** The camera updates at ``CAMERA_UPDATE_PERIOD_S`` and the
   control loop runs at 30 Hz, so the pose is constant across pairs of control
   steps and steps between them.
2. **Isotropic residual.** Position and rotation-vector error are drawn per
   camera frame at the sigma the certified p95 of the error *norm* implies.
3. **Missed detections.** A frame is missed at the certified rate, and a missed
   frame holds the previous pose -- the deployed estimator fails closed, which
   makes a miss a *stale* observation rather than a large one.
4. **The velocity the estimator manufactures.** Velocity is not sampled. It is
   the same finite difference and first-order filter the deployed estimator
   runs, at the same control rate, over this staircase. That is deliberate: the
   camera period is twice the control period, so a differenced estimate is zero
   on one step and a full jump on the next, and a 1.9 mm jump over a 1/15 s
   camera period is 28.6 mm/s of pure noise on a channel whose seated signal is
   about 0.69 mm/s. Modelling that as a Gaussian on the velocity would have
   hidden the structure that makes it hard.

What is **not** modelled: occlusion geometry and the hold-until-capture
interlock. Those are properties of the scene rather than of the sensor, and a
surrogate that invented them would be tuning. So this is a *lower bound* on
deployment error and the reports that use it say so.

**The in-loop tail is separate, and it is switchable.** The certified residual is
measured on held-out still frames, where the position error tops out at 3.3 mm.
In the loop the same estimator records a per-episode maximum around 30 mm --
*including on the episodes that succeed*, whose mean error is the same 1.9 to
2.1 mm as the ones that fail. So the tail is a property of the poses a closed
loop actually visits, not a symptom of an episode already going wrong, and a
surrogate without it is missing a cause rather than a consequence. It is off by
default, because switching it on is a second change and would make the first
arm's result unattributable; `estimator_noise_outlier_rate` and
`estimator_noise_outlier_scale` turn it on as its own arm.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import axis_angle_from_quat, quat_from_angle_axis, quat_inv, quat_mul

from zero_g_blade_swap.estimator_noise import EstimatorNoiseModel
from zero_g_blade_swap.pose_head import MODULE_POSE_DIM
from zero_g_blade_swap.servicing_camera import CAMERA_UPDATE_PERIOD_S

from .perception import (
    _orientation_from_module_pose,
    extraction_remaining_from_module_pose,
    grip_error_from_module_pose,
    insertion_goal_error_from_module_pose,
    module_pose_label,
)

# The certificate the skills are noised from. It is the flush-datum estimator
# qualification on held-out workflow-envelope frames; naming it here rather than
# copying its numbers means a training run cannot be noised from figures that no
# longer match any evidence file.
DEFAULT_ESTIMATOR_CERTIFICATION = "evidence/fiducial_rgbd_flush_v4_seed285_gripper_clear.json"


class SurrogateModuleStateEstimator:
    """Deliver the deployed estimator's statistics from simulator state.

    The interface is deliberately the same as ``ModuleStateEstimator.estimate``
    -- ``(pose, filtered_velocity)`` cached per control step -- so the four
    observation terms below are the same shape and ordering as both the state
    terms they replace and the perceived terms they stand in for.
    """

    def __init__(self, env) -> None:
        self._env = env
        cfg = env.cfg
        certification = str(getattr(cfg, "estimator_noise_certification", DEFAULT_ESTIMATOR_CERTIFICATION))
        self._model = EstimatorNoiseModel.from_certification(certification)

        camera_period_s = float(getattr(cfg, "estimator_noise_camera_period_s", CAMERA_UPDATE_PERIOD_S))
        if camera_period_s < float(env.step_dt):
            raise ValueError(
                "estimator_noise_camera_period_s is shorter than one control step; the surrogate would "
                "then be sampling faster than the camera it stands in for"
            )
        # Whole control steps, because the pose can only change on a step
        # boundary. The deployed camera is 15 Hz against a 30 Hz loop, so this
        # is 2, and the resulting staircase is the point of the model.
        self._camera_period_steps = max(1, int(round(camera_period_s / float(env.step_dt))))

        self._filter_time_constant_s = float(getattr(cfg, "perception_velocity_filter_time_constant_s", 0.10))
        if self._filter_time_constant_s < 0.0:
            raise ValueError("perception_velocity_filter_time_constant_s must be non-negative")

        # The in-loop tail, off by default. See the module docstring: the
        # certified still-frame residual tops out at 3.3 mm while the same
        # estimator in the loop records about 30 mm per episode, on winning
        # episodes as much as on losing ones.
        self._outlier_rate = float(getattr(cfg, "estimator_noise_outlier_rate", 0.0))
        self._outlier_scale = float(getattr(cfg, "estimator_noise_outlier_scale", 1.0))
        if not 0.0 <= self._outlier_rate <= 1.0:
            raise ValueError("estimator_noise_outlier_rate must be a probability")
        if self._outlier_scale < 1.0:
            raise ValueError("estimator_noise_outlier_scale below 1 would make the tail lighter than the bulk")

        device = env.device
        self._pose = torch.zeros((env.num_envs, MODULE_POSE_DIM), device=device)
        self._velocity = torch.zeros((env.num_envs, 6), device=device)
        self._previous_position = torch.zeros((env.num_envs, 3), device=device)
        self._previous_orientation = torch.zeros((env.num_envs, 4), device=device)
        self._previous_orientation[:, 0] = 1.0
        self._history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self._pose_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self._missed_frames = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self._camera_frames = 0
        self._last_camera_step: int | None = None
        self._last_update_step: int | None = None
        self._cached_step: int | None = None

    @property
    def model(self) -> EstimatorNoiseModel:
        return self._model

    @property
    def camera_period_steps(self) -> int:
        return self._camera_period_steps

    def describe(self) -> dict[str, float | int | str]:
        """Report what a run was noised with, for the training run's own record."""

        described: dict[str, float | int | str] = dict(self._model.describe())
        described["camera_period_control_steps"] = self._camera_period_steps
        described["velocity_filter_time_constant_s"] = self._filter_time_constant_s
        described["camera_frames"] = self._camera_frames
        described["missed_frames"] = int(self._missed_frames.sum().item())
        described["outlier_rate"] = self._outlier_rate
        described["outlier_scale"] = self._outlier_scale
        return described

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Invalidate reset environments so no estimate crosses an episode boundary."""

        ids = slice(None) if env_ids is None else env_ids
        self._history_valid[ids] = False
        self._pose_valid[ids] = False
        self._velocity[ids] = 0.0
        # A partial reset can land after this step's observations were cached.
        # Invalidate the whole batch rather than reason about which group ran.
        self._cached_step = None

    def estimate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(pose, filtered_velocity)`` for the current control step."""

        step = int(self._env.common_step_counter)
        if self._cached_step == step:
            return self._pose, self._velocity

        if self._last_camera_step is None or step - self._last_camera_step >= self._camera_period_steps:
            self._sample_camera_frame()
            self._last_camera_step = step

        self._update_velocity(step)
        self._cached_step = step
        return self._pose, self._velocity

    def _sample_camera_frame(self) -> None:
        """Draw one noisy pose per environment, and hold it where a frame is missed."""

        truth = module_pose_label(self._env)
        device = truth.device
        count = truth.shape[0]

        # One scale per environment per frame: the bulk sigma, or the tail's
        # multiple of it. Drawing the scale rather than a separate outlier keeps
        # the residual isotropic and keeps the bulk exactly as certified.
        scale = torch.ones((count, 1), device=device)
        if self._outlier_rate > 0.0:
            scale = torch.where(
                torch.rand((count, 1), device=device) < self._outlier_rate,
                torch.full_like(scale, self._outlier_scale),
                scale,
            )
        position = truth[:, :3] + torch.randn((count, 3), device=device) * self._model.position_sigma_m * scale
        perturbation = torch.randn((count, 3), device=device) * self._model.orientation_sigma_rad * scale
        angle = torch.linalg.vector_norm(perturbation, dim=-1)
        axis = torch.zeros_like(perturbation)
        axis[:, 0] = 1.0
        axis = torch.where((angle > 1.0e-9).unsqueeze(-1), perturbation / angle.clamp_min(1.0e-9).unsqueeze(-1), axis)
        orientation = quat_mul(quat_from_angle_axis(angle, axis), _orientation_from_module_pose(truth))
        sampled = torch.cat((position, axis_angle_from_quat(orientation)), dim=-1)

        detected = torch.rand(count, device=device) < self._model.detection_rate
        # An environment with no pose yet cannot hold one: the first frame of an
        # episode is always taken, exactly as an episode that never detected
        # anything would never have started.
        detected = detected | ~self._pose_valid
        self._missed_frames += (~detected).long()
        self._camera_frames += 1
        self._pose = torch.where(detected.unsqueeze(-1), sampled, self._pose)
        self._pose_valid |= detected

    def _update_velocity(self, step: int) -> None:
        """Difference and filter at the control rate, as the deployed estimator does."""

        orientation = _orientation_from_module_pose(self._pose)
        elapsed_steps = 1 if self._last_update_step is None else max(1, step - self._last_update_step)
        dt = float(self._env.step_dt) * elapsed_steps

        linear = (self._pose[:, :3] - self._previous_position) / dt
        rotation_delta = quat_mul(orientation, quat_inv(self._previous_orientation))
        angular = axis_angle_from_quat(rotation_delta) / dt
        raw_velocity = torch.cat((linear, angular), dim=-1)
        raw_velocity = torch.where(self._history_valid.unsqueeze(-1), raw_velocity, torch.zeros_like(raw_velocity))

        tau = self._filter_time_constant_s
        alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
        self._velocity.mul_(1.0 - alpha).add_(raw_velocity, alpha=alpha)
        self._velocity.masked_fill_(~self._history_valid.unsqueeze(-1), 0.0)
        self._previous_position.copy_(self._pose[:, :3])
        self._previous_orientation.copy_(orientation)
        self._history_valid.fill_(True)
        self._last_update_step = step


def shared_surrogate_estimator(env) -> SurrogateModuleStateEstimator:
    """One surrogate per vectorized environment, shared by every noised term."""

    estimator = getattr(env, "_surrogate_module_state_estimator", None)
    if estimator is None:
        estimator = SurrogateModuleStateEstimator(env)
        env._surrogate_module_state_estimator = estimator
    return estimator


class _NoisedModuleObservation(ManagerTermBase):
    """Bind a policy view to the shared surrogate, and reset with the episode."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._estimator = shared_surrogate_estimator(env)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        self._estimator.reset(env_ids)


class NoisedGraspError(_NoisedModuleObservation):
    """Six-value tool-to-grip error, from a pose carrying the estimator's error."""

    def __call__(self, env) -> torch.Tensor:
        pose, _ = self._estimator.estimate()
        return grip_error_from_module_pose(env, pose)


class NoisedExtractionRemaining(_NoisedModuleObservation):
    """One-value extraction travel, from a pose carrying the estimator's error."""

    def __call__(self, env, target_x: float | None = None) -> torch.Tensor:
        del env
        pose, _ = self._estimator.estimate()
        if target_x is None:
            return extraction_remaining_from_module_pose(pose)
        return extraction_remaining_from_module_pose(pose, target_x)


class NoisedInsertionGoalError(_NoisedModuleObservation):
    """Six-value insertion-goal error, from a pose carrying the estimator's error."""

    def __call__(self, env, command_name: str = "insertion_goal") -> torch.Tensor:
        pose, _ = self._estimator.estimate()
        return insertion_goal_error_from_module_pose(env, pose, command_name)


class NoisedModuleVelocity(_NoisedModuleObservation):
    """Six-value velocity the estimator manufactures, not a sampled velocity."""

    def __call__(self, env) -> torch.Tensor:
        del env
        _, velocity = self._estimator.estimate()
        return velocity


__all__ = [
    "DEFAULT_ESTIMATOR_CERTIFICATION",
    "NoisedExtractionRemaining",
    "NoisedGraspError",
    "NoisedInsertionGoalError",
    "NoisedModuleVelocity",
    "SurrogateModuleStateEstimator",
    "shared_surrogate_estimator",
]
