"""Pose uncertainty that a policy cannot see its way around, plus FORGE's
force-threshold conditioning and IndustReal's sampling-based curriculum.

Every RL task in this repository before 2026-08-10 trained against a problem
containing no uncertainty: the policy observed ``insertion_goal_error``, derived
from simulator ground truth. With a rigid known object on a constrained axis and
full observability that is motion planning and force control, and a scripted
controller solves it.

**Where the uncertainty has to live, and why.** The obvious construction is to
add a bias to the reported pose error. On this task that is a *fake*
uncertainty, and it is worth stating plainly because it was built and discarded
here first. The blade is welded to the tool by a PhysX fixed joint, so its pose
is a fixed offset from the tool frame; the tool frame is observed directly as
``end_effector_pose_local``; and if the goal is a constant, then the true error
is an exactly learnable function of an observation the actor already has. A
network can recover the truth and ignore the injected bias entirely, and both
arms of the ablation would score the same for reasons that have nothing to do
with force.

So the slot genuinely moves. Each episode displaces the rack's guide rails
laterally by an offset the policy is never told, and the goal moves with them,
which is what FORGE (arXiv 2408.04587) and arXiv 2604.19677 both do: the
uncertainty is in where the *fixed part* is, not in the robot's knowledge of
itself. Nothing in the observation determines that offset. Contact against the
rail is the only thing that reveals it, which is precisely the variable this
task exists to measure.

The displacement is lateral only, for two measured reasons.

*Axial would measure the wrong thing.* A depth offset is resolvable without a
force sensor at all: the blade bottoms out, its velocity goes to zero, and
``blade_velocity`` is in both actors' observations. A force-blind policy would
find the stop as easily as a force-aware one, and the ablation would show
nothing. Lateral is different: both policies can tell they have stalled, but only
a force-aware one can tell *which side* it is jammed against, because that is
carried by the direction of the contact force and by nothing else.

*Vertical would be unsolvable.* The Level-2 profile disables the slot floor
collider, so nothing constrains the blade vertically. A vertical offset would be
unobservable and still counted against the 2.5 mm lateral success tolerance,
sinking both arms equally.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from isaaclab.utils.math import axis_angle_from_quat, subtract_frame_transforms

from zero_g_blade_swap.math_utils import update_sampling_bound

from .insertion import attached_blade_pose_world, blade_contact_force

#: Scene assets that make up the slot and its lead-in. All are kinematic rigid
#: bodies, so a written pose is a moved wall rather than something contact can
#: push back. The flares must move with the rails or the lead-in would aim at
#: the wrong place, which is the one thing that makes an offset entry possible.
SLOT_ASSET_NAMES = (
    "blade_slot",
    "blade_slot_left_guide",
    "blade_slot_right_guide",
    "blade_slot_upper_left_lip",
    "blade_slot_upper_right_lip",
    "blade_slot_entry_left_flare",
    "blade_slot_entry_right_flare",
)


def slot_offset(env) -> torch.Tensor:
    """Per-environment lateral displacement of the slot, in metres."""

    offset = getattr(env, "_slot_offset_m", None)
    if offset is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return offset


def belief_bias_magnitude(env) -> torch.Tensor:
    """How wrong the policy's idea of the slot is, per environment.

    For evaluation reporting only. This is never an observation: telling the
    policy how wrong its estimate is would remove the reason it has to touch
    anything.
    """

    return torch.linalg.vector_norm(slot_offset(env), dim=-1)


def randomize_slot_offset(
    env,
    env_ids: torch.Tensor | None,
    asset_names: Sequence[str] = SLOT_ASSET_NAMES,
) -> None:
    """Move the slot sideways by an amount the policy is never told.

    The magnitude is drawn from ``[floor, ceiling]``, where
    :class:`BeliefSamplingCurriculum` raises the floor as success improves and
    the ceiling never moves. The sign is random, so the policy cannot learn to
    correct one way.
    """

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    offset = getattr(env, "_slot_offset_m", None)
    if offset is None or offset.shape[0] != env.num_envs:
        offset = torch.zeros((env.num_envs, 3), device=env.device)
        env._slot_offset_m = offset

    floor = float(getattr(env, "_belief_bias_floor_m", torch.zeros(())))
    ceiling = float(getattr(env, "_belief_bias_ceiling_m", torch.zeros(())))
    magnitude = torch.empty(len(ids), device=env.device).uniform_(min(floor, ceiling), max(floor, ceiling))
    sign = torch.where(torch.rand(len(ids), device=env.device) < 0.5, -1.0, 1.0)
    offset[ids] = 0.0
    offset[ids, 1] = magnitude * sign

    for name in asset_names:
        # A profile may not carry every part of the channel; the plain
        # rigid-grasp scene has no lips or flares, for instance.
        if name not in env.scene.rigid_objects:
            continue
        asset = env.scene[name]
        pose = asset.data.default_root_state[ids, :7].clone()
        pose[:, :3] += env.scene.env_origins[ids] + offset[ids]
        asset.write_root_pose_to_sim(pose, env_ids=ids)


def nominal_goal_error(env, command_name: str = "insertion_goal") -> torch.Tensor:
    """The blade-to-goal error the policy *believes*, against the slot's
    nominal position rather than where the slot actually is.

    Identical in form to ``insertion_goal_error`` and computed the same way, so
    the actor's input has the same units, scale, and meaning it always had. The
    only difference is that it is wrong, by exactly the slot displacement, for
    the whole episode.
    """

    blade_position, blade_orientation = attached_blade_pose_world(env)
    goal = env.command_manager.get_command(command_name)
    believed_position = goal[:, :3] - slot_offset(env)
    local_position = blade_position - env.scene.env_origins
    relative_position, relative_quat = subtract_frame_transforms(
        local_position,
        blade_orientation,
        believed_position,
        goal[:, 3:7],
    )
    return torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)


class BeliefPoseErrorObservation(ManagerTermBase):
    """The believed pose error, plus estimator jitter.

    Jitter is deliberately the uninteresting term: a policy averages it away in
    a few control steps. The slot displacement it cannot average away at all,
    which is why that is the quantity being swept.
    """

    def __call__(
        self,
        env,
        position_jitter_m: float = 0.0005,
        orientation_jitter_rad: float = 0.005,
        command_name: str = "insertion_goal",
    ) -> torch.Tensor:
        belief = nominal_goal_error(env, command_name)
        if position_jitter_m > 0.0:
            belief[:, :3] += torch.randn_like(belief[:, :3]) * position_jitter_m
        if orientation_jitter_rad > 0.0:
            belief[:, 3:] += torch.randn_like(belief[:, 3:]) * orientation_jitter_rad
        return belief


class ContactForceThresholdObservation(ManagerTermBase):
    """FORGE's force-threshold conditioning: one value the policy is judged on.

    A maximum allowable contact force is sampled once per episode and given to
    the policy, and the reward charges a linear hinge above it. One policy
    therefore covers a family of force budgets and can be asked at deployment
    for a gentler or a more forceful insertion, instead of the two fixed penalty
    profiles this project already measured as ineffective.

    The range is FORGE's mechanism with this workcell's numbers: the promoted
    Level-2 policy's measured peak contact force is 4.7 N at the median and
    16.6 N at p95, so 5 N is tighter than typical contact and 20 N is looser
    than p95.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        env._contact_force_threshold_n = torch.zeros(env.num_envs, device=env.device)
        self._range = (
            float(cfg.params.get("minimum_n", 5.0)),
            float(cfg.params.get("maximum_n", 20.0)),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env = self._env
        thresholds = env._contact_force_threshold_n
        ids = slice(None) if env_ids is None else env_ids
        count = thresholds[ids].shape[0]
        if count == 0:
            return
        forced = getattr(env, "_forced_contact_force_threshold_n", None)
        if forced is not None:
            thresholds[ids] = float(forced)
            return
        thresholds[ids] = torch.empty(count, device=env.device).uniform_(*self._range)

    def __call__(
        self,
        env,
        minimum_n: float = 5.0,
        maximum_n: float = 20.0,
        force_scale_n: float = 20.0,
    ) -> torch.Tensor:
        del minimum_n, maximum_n
        return (env._contact_force_threshold_n / max(force_scale_n, 1.0e-6)).unsqueeze(-1)


def contact_force_threshold_n(env) -> torch.Tensor:
    threshold = getattr(env, "_contact_force_threshold_n", None)
    if threshold is None:
        return torch.zeros(env.num_envs, device=env.device)
    return threshold


def force_threshold_penalty(
    env,
    force_scale_n: float = 20.0,
    sensor_name: str = "blade_contact",
) -> torch.Tensor:
    """FORGE's excess-force penalty: ``max(0, ||F|| - F_th)``, linear.

    Linear rather than quadratic on purpose. ``docs/status.md`` records a
    quadratic penalty at two strengths, the stronger charging the same order as
    the success reward, changing mean contact by 2.6% and impulse not at all, so
    a steeper curve is the one thing already known not to work here. What
    changes is that the threshold varies per episode and the policy can see it.
    """

    excess = blade_contact_force(env, sensor_name) - contact_force_threshold_n(env)
    return excess.clamp_min(0.0) / max(force_scale_n, 1.0e-6)


class BeliefSamplingCurriculum(ManagerTermBase):
    """IndustReal's sampling-based curriculum, applied to the slot displacement.

    The displacement is sampled uniformly in ``[floor, ceiling]``. The ceiling is
    fixed at the trained maximum from the first step and only the floor rises,
    by ``increase_m`` whenever rolling success clears 80% and back by
    ``decrease_m`` if it falls under 10%.

    That ordering is the point. A stage curriculum ramps *into* difficulty and
    lets an agent converge on a policy that exploits easy initial states; this
    project's first grasp policy measured 99.3% while completing in 0.30 s
    because the reset already sat inside the success tolerance. Here the hardest
    displacement is in the distribution throughout and the easy end is withdrawn
    as the policy earns it.
    """

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._history: deque[float] = deque(maxlen=int(cfg.params["window_size"]))
        self._floor = 0.0
        self._last_change_step = int(env.common_step_counter)
        self._forced: float | None = None
        env._belief_bias_floor_m = torch.zeros((), device=env.device)
        env._belief_bias_ceiling_m = torch.full((), float(cfg.params["bias_ceiling_m"]), device=env.device)

    def force_bias(self, bias_m: float) -> None:
        """Pin both bounds, so every episode runs at one displacement."""

        if bias_m < 0.0:
            raise ValueError("forced belief bias must be non-negative")
        self._forced = float(bias_m)
        self._floor = float(bias_m)
        self._history.clear()
        self._env._belief_bias_floor_m.fill_(float(bias_m))
        self._env._belief_bias_ceiling_m.fill_(float(bias_m))

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        success_term: str,
        bias_ceiling_m: float,
        increase_m: float,
        decrease_m: float,
        window_size: int,
        minimum_level_steps: int,
    ) -> dict[str, float]:
        ids = (
            torch.arange(env.num_envs, device=env.device)[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        )
        if len(ids) > 0 and self._forced is None:
            completed = env.episode_length_buf[ids] > 0
            successes = env.termination_manager.get_term(success_term)[ids][completed]
            self._history.extend(float(value) for value in successes.detach().cpu())

        if self._forced is not None:
            return {
                "belief_bias_floor_mm": 1_000.0 * self._floor,
                "belief_bias_ceiling_mm": 1_000.0 * self._floor,
                "belief_rolling_success": 0.0,
            }

        current_step = int(env.common_step_counter)
        floor, rolling, moved = update_sampling_bound(
            self._floor,
            tuple(self._history),
            increase=increase_m,
            decrease=decrease_m,
            maximum=bias_ceiling_m,
            window_size=window_size,
            steps_elapsed=current_step - self._last_change_step,
            minimum_steps=minimum_level_steps,
        )
        if moved:
            self._floor = floor
            self._last_change_step = current_step
            self._history.clear()
            env._belief_bias_floor_m.fill_(floor)
        return {
            "belief_bias_floor_mm": 1_000.0 * self._floor,
            "belief_bias_ceiling_mm": 1_000.0 * bias_ceiling_m,
            "belief_rolling_success": float(rolling),
        }


__all__ = [
    "SLOT_ASSET_NAMES",
    "BeliefPoseErrorObservation",
    "BeliefSamplingCurriculum",
    "ContactForceThresholdObservation",
    "belief_bias_magnitude",
    "contact_force_threshold_n",
    "force_threshold_penalty",
    "nominal_goal_error",
    "randomize_slot_offset",
    "slot_offset",
]
