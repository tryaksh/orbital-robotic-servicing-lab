"""Terminal insertion metrics captured before Isaac Lab's automatic reset.

The collector runs from :class:`~zero_g_blade_swap.evaluation.TerminalMetricsMixin`
inside ``_reset_idx``, while the scene still holds the terminal physics state of
every environment that just finished.  It reads existing task functions only and
has no effect on rewards, observations, actions, or termination decisions.
"""

from __future__ import annotations

import torch

from zero_g_blade_swap.evaluation import (
    BELIEF_BIAS_FIELD,
    BLADE_MASS_FIELD,
    CONTACT_IMPULSE_FIELD,
    GRIP_ATTITUDE_APPROACH_AXIS_FIELD,
    GRIP_ATTITUDE_THIRD_AXIS_FIELD,
    GRIP_YAW_CLOSING_AXIS_FIELD,
    PEAK_CONTACT_FORCE_FIELD,
    SUCCESS_TERMINATIONS,
    TERMINAL_METRIC_FIELDS,
    TERMINATION_PRIORITY,
    TERMINATION_REASONS,
    UNCATEGORIZED_TERMINATION,
    TerminalEpisodeRecorder,
)

from .grapple import grapple_grip_attitude_axes, grapple_grip_error_metrics
from .insertion import (
    attached_blade_velocity,
    insertion_error_metrics,
    secured_blade_error_metrics,
)
from .uncertainty import belief_bias_magnitude

# Each skill ends on its own success term, so "was this episode a success"
# cannot be a single index. Grasp ends on capture_success and extract on
# extraction_success; scoring either against insertion_success would record
# every episode as a failure.
SUCCESS_REASON_IDS = tuple(TERMINATION_REASONS.index(name) for name in SUCCESS_TERMINATIONS)
UNCATEGORIZED_REASON_ID = TERMINATION_REASONS.index(UNCATEGORIZED_TERMINATION)
BLADE_CONTACT_SENSOR = "blade_contact"


class InsertionTerminalMetrics:
    """Record one reset-safe metric row per completed insertion episode."""

    def __init__(self, env, recorder: TerminalEpisodeRecorder | None = None) -> None:
        # Blade mass is randomized only from robustness level 2 upward.  Record
        # it as an extra column there so pooled reports can gate success per
        # mass band; runs without it stay loadable because reports align by name.
        self._records_mass = getattr(getattr(env.cfg, "events", None), "blade_mass", None) is not None
        self._records_contact = BLADE_CONTACT_SENSOR in env.scene.sensors
        # Only the pose-belief tasks carry a bias, and it is the swept axis, so
        # record what each episode actually ran at rather than trusting the flag.
        self._records_belief = getattr(env, "_belief_bias_magnitude_m", None) is not None
        # A head-on capture is held a quarter turn from the top-down grasp
        # convention, so it needs the grapple grip metrics rather than the
        # secured-blade ones. One flag, read in both places that care.
        self._head_on = getattr(env.cfg, "tool_target_rot", None) is not None
        fields = TERMINAL_METRIC_FIELDS
        if self._records_mass:
            fields = (*fields, BLADE_MASS_FIELD)
        if self._records_contact:
            fields = (*fields, PEAK_CONTACT_FORCE_FIELD, CONTACT_IMPULSE_FIELD)
        if self._records_belief:
            fields = (*fields, BELIEF_BIAS_FIELD)
        # Opt-in, and only where a head-on capture makes the decomposition mean
        # anything. Default off so every certification already produced keeps
        # its exact column set and stays poolable with new runs.
        self._records_grip_axes = bool(getattr(env.cfg, "record_grip_axes", False)) and self._head_on
        if self._records_grip_axes:
            fields = (
                *fields,
                GRIP_YAW_CLOSING_AXIS_FIELD,
                GRIP_ATTITUDE_THIRD_AXIS_FIELD,
                GRIP_ATTITUDE_APPROACH_AXIS_FIELD,
            )
        self.recorder = TerminalEpisodeRecorder(fields) if recorder is None else recorder
        self._blade_mass: torch.Tensor | None = None
        self._peak_force = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._impulse = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        active = set(env.termination_manager.active_terms)
        missing = active.difference(TERMINATION_REASONS)
        if missing:
            raise ValueError(f"termination terms without an evaluation category: {sorted(missing)}")
        # Lowest priority first so that a higher-priority write overwrites it.
        self._reason_terms = tuple(
            (name, TERMINATION_REASONS.index(name)) for name in reversed(TERMINATION_PRIORITY) if name in active
        )

    def termination_reason_ids(self, env) -> torch.Tensor:
        """Categorize each environment's termination by documented priority."""

        reason = torch.full(
            (env.num_envs,),
            float(UNCATEGORIZED_REASON_ID),
            dtype=torch.float32,
            device=env.device,
        )
        for name, reason_id in self._reason_terms:
            reason[env.termination_manager.get_term(name)] = float(reason_id)
        return reason

    def __call__(self, env, env_ids) -> int:
        axial, lateral, orientation = insertion_error_metrics(env)
        velocity = attached_blade_velocity(env)
        # A head-on capture is held a quarter turn from the top-down grasp
        # convention secured_blade_error_metrics is written against, so on
        # those tasks that function reports a constant ~2.15 rad and the
        # recorded column says nothing about whether the grip is intact.
        # Measure the grip in the convention the task actually holds it in.
        grip_metrics = grapple_grip_error_metrics if self._head_on else secured_blade_error_metrics
        grasp_position, grasp_orientation = grip_metrics(env)
        reason = self.termination_reason_ids(env)
        stage = getattr(env, "_insertion_curriculum_stage", None)
        stage_values = (
            torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)
            if stage is None
            else stage.to(dtype=torch.float32)
        )
        control_steps = env.episode_length_buf.to(dtype=torch.float32)
        columns = [
            torch.isin(reason, reason.new_tensor(SUCCESS_REASON_IDS)).to(dtype=torch.float32),
            reason,
            stage_values,
            control_steps,
            control_steps * float(env.step_dt),
            axial.to(dtype=torch.float32),
            lateral.to(dtype=torch.float32),
            orientation.to(dtype=torch.float32),
            torch.linalg.vector_norm(velocity[:, :3], dim=-1).to(dtype=torch.float32),
            torch.linalg.vector_norm(velocity[:, 3:], dim=-1).to(dtype=torch.float32),
            grasp_position.to(dtype=torch.float32),
            grasp_orientation.to(dtype=torch.float32),
        ]
        if self._records_mass:
            columns.append(self.blade_mass(env))
        if self._records_contact:
            # Fold in the terminal control step, which the per-step accumulator
            # never sees because this hook runs before the caller resumes.
            self.accumulate(env)
            columns.append(self._peak_force)
            columns.append(self._impulse)
        if self._records_belief:
            columns.append(belief_bias_magnitude(env).to(dtype=torch.float32))
        if self._records_grip_axes:
            axes = grapple_grip_attitude_axes(env).to(dtype=torch.float32)
            columns.extend((axes[:, 0], axes[:, 1], axes[:, 2]))
        rows = torch.stack(tuple(columns), dim=-1)
        recorded = self.recorder.record(rows[env_ids].detach().cpu())
        if self._records_contact:
            self._peak_force[env_ids] = 0.0
            self._impulse[env_ids] = 0.0
        return recorded

    def contact_force_magnitude(self, env) -> torch.Tensor:
        """Largest net contact force on any blade body, in newtons."""

        forces = env.scene.sensors[BLADE_CONTACT_SENSOR].data.net_forces_w
        if forces is None:
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        return torch.linalg.vector_norm(forces, dim=-1).amax(dim=-1).to(dtype=torch.float32)

    def accumulate(self, env) -> None:
        """Fold one control step's contact into the running episode maxima.

        Call once per control step. The terminal step is folded in separately by
        ``__call__``, because Isaac Lab resets a finished environment before the
        caller regains control.
        """

        if not self._records_contact:
            return
        force = self.contact_force_magnitude(env)
        torch.maximum(self._peak_force, force, out=self._peak_force)
        self._impulse.add_(force * float(env.step_dt))

    def blade_mass(self, env) -> torch.Tensor:
        """Per-environment blade mass, read once because startup sets it."""

        if self._blade_mass is None:
            masses = env.scene["spare_blade"].root_physx_view.get_masses()[:, 0]
            self._blade_mass = masses.to(device=env.device, dtype=torch.float32).clone()
        return self._blade_mass


__all__ = ["InsertionTerminalMetrics"]
