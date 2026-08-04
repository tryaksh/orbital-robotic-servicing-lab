"""Curricula for transitioning from reaching to precision insertion."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from zero_g_blade_swap.math_utils import update_curriculum_stage


class SuccessRateStageCurriculum(ManagerTermBase):
    """Promote global task difficulty at 70% over 100 completed episodes."""

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._history: deque[float] = deque(maxlen=int(cfg.params["window_size"]))
        self._stage = 0
        env._curriculum_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        success_term: str,
        threshold: float,
        window_size: int,
        max_stage: int,
    ) -> dict[str, float]:
        if isinstance(env_ids, slice):
            ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        if len(ids) > 0:
            completed = env.episode_length_buf[ids] > 0
            successes = env.termination_manager.get_term(success_term)[ids][completed]
            self._history.extend(float(value) for value in successes.detach().cpu())
        self._stage, rolling, promoted = update_curriculum_stage(
            self._stage,
            tuple(self._history),
            threshold=threshold,
            window_size=window_size,
            max_stage=max_stage,
        )
        if promoted:
            self._history.clear()
        env._curriculum_stage.fill_(self._stage)
        return {"stage": float(self._stage), "rolling_success": float(rolling)}


class linear_reward_weight(ManagerTermBase):
    """Linearly interpolate one reward weight over global environment steps."""

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(self._term_name)

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        term_name: str,
        start_weight: float,
        end_weight: float,
        start_step: int,
        end_step: int,
    ) -> float:
        del env_ids
        if end_step <= start_step:
            raise ValueError("end_step must be greater than start_step for linear_reward_weight.")
        step = int(env.common_step_counter)
        alpha = min(1.0, max(0.0, (step - start_step) / (end_step - start_step)))
        self._term_cfg.weight = start_weight + alpha * (end_weight - start_weight)
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return float(self._term_cfg.weight)


__all__ = ["SuccessRateStageCurriculum", "linear_reward_weight"]
