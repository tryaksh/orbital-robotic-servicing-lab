"""Terminal insertion metrics captured before Isaac Lab's automatic reset.

The collector runs from :class:`~zero_g_blade_swap.evaluation.TerminalMetricsMixin`
inside ``_reset_idx``, while the scene still holds the terminal physics state of
every environment that just finished.  It reads existing task functions only and
has no effect on rewards, observations, actions, or termination decisions.
"""

from __future__ import annotations

import torch

from zero_g_blade_swap.evaluation import (
    TERMINAL_METRIC_FIELDS,
    TERMINATION_PRIORITY,
    TERMINATION_REASONS,
    UNCATEGORIZED_TERMINATION,
    TerminalEpisodeRecorder,
)

from .insertion import (
    attached_blade_velocity,
    insertion_error_metrics,
    secured_blade_error_metrics,
)

SUCCESS_REASON_ID = TERMINATION_REASONS.index("insertion_success")
UNCATEGORIZED_REASON_ID = TERMINATION_REASONS.index(UNCATEGORIZED_TERMINATION)


class InsertionTerminalMetrics:
    """Record one reset-safe metric row per completed insertion episode."""

    def __init__(self, env, recorder: TerminalEpisodeRecorder | None = None) -> None:
        self.recorder = TerminalEpisodeRecorder(TERMINAL_METRIC_FIELDS) if recorder is None else recorder
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
        grasp_position, grasp_orientation = secured_blade_error_metrics(env)
        reason = self.termination_reason_ids(env)
        stage = getattr(env, "_insertion_curriculum_stage", None)
        stage_values = (
            torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)
            if stage is None
            else stage.to(dtype=torch.float32)
        )
        control_steps = env.episode_length_buf.to(dtype=torch.float32)
        rows = torch.stack(
            (
                (reason == SUCCESS_REASON_ID).to(dtype=torch.float32),
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
            ),
            dim=-1,
        )
        return self.recorder.record(rows[env_ids].detach().cpu())


__all__ = ["InsertionTerminalMetrics"]
