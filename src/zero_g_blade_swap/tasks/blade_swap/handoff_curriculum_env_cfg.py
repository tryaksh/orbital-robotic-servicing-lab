"""Success-gated reverse curriculum for the rack-mouth insertion handoff."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from zero_g_blade_swap.math_utils import update_reverse_station_frontier

from . import mdp
from .two_slot_env_cfg import ZeroGBladeGrapplePinInsertTwoSlotEnvCfg

INSERT_STATION_COUNT = 9
START_FRONTIER_STATION = 6
DEFAULT_BLADE_ASSET_CFG = SceneEntityCfg("spare_blade")


def reset_reverse_station_curriculum(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    arm_poses_by_bay: tuple[tuple[tuple[float, ...], ...], ...],
    blade_poses_by_bay: tuple[tuple[tuple[float, ...], ...], ...],
    finger_positions: tuple[float, ...],
    hold_positions: tuple[float, ...],
    noise_rad: float,
    station_count: int,
    frontier_probability: float,
    blade_asset_cfg: SceneEntityCfg = DEFAULT_BLADE_ASSET_CFG,
) -> None:
    """Sample the active frontier heavily while retaining solved later starts."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    frontier = int(getattr(env, "_insert_station_frontier", START_FRONTIER_STATION))
    if station_count <= 0 or not 0 <= frontier < station_count:
        raise ValueError("reverse insertion frontier is outside the reset bank")
    if not 0.0 < frontier_probability <= 1.0:
        raise ValueError("frontier_probability must be in (0, 1]")

    if frontier == station_count - 1:
        stations = torch.full((len(ids),), frontier, dtype=torch.long, device=env.device)
    else:
        stations = torch.randint(frontier + 1, station_count, (len(ids),), device=env.device)
        stations[torch.rand(len(ids), device=env.device) < frontier_probability] = frontier
    for station in range(frontier, station_count):
        selected = ids[stations == station]
        if len(selected) > 0:
            mdp.reset_grapple_insert_stroke(
                env,
                selected,
                asset_cfg=asset_cfg,
                gripper_cfg=gripper_cfg,
                arm_poses_by_bay=arm_poses_by_bay,
                blade_poses_by_bay=blade_poses_by_bay,
                finger_positions=finger_positions,
                hold_positions=hold_positions,
                noise_rad=noise_rad,
                forced_station=station,
                blade_asset_cfg=blade_asset_cfg,
            )


class ReverseStationSuccessCurriculum(ManagerTermBase):
    """Move the rack-mouth frontier only after success at the current station."""

    def __init__(self, cfg: CurrTerm, env) -> None:
        super().__init__(cfg, env)
        self._frontier = int(cfg.params["start_station"])
        self._history: deque[float] = deque(maxlen=int(cfg.params["window_size"]))
        self._frontier_start_step = int(env.common_step_counter)
        env._insert_station_frontier = self._frontier

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        success_term: str,
        start_station: int,
        threshold: float,
        window_size: int,
        minimum_frontier_steps: int,
        station_count: int,
    ) -> dict[str, float]:
        del start_station
        ids = (
            torch.arange(env.num_envs, device=env.device)[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        )
        completed = env.episode_length_buf[ids] > 0
        reset_stations = getattr(env, "_insert_reset_station", None)
        if len(ids) > 0 and bool(completed.any()) and reset_stations is not None:
            successes = env.termination_manager.get_term(success_term)[ids][completed]
            stations = reset_stations[ids][completed]
            frontier_successes = successes[stations == self._frontier]
            self._history.extend(float(value) for value in frontier_successes.detach().cpu())

        current_step = int(env.common_step_counter)
        updated, rolling, promoted = update_reverse_station_frontier(
            self._frontier,
            tuple(self._history),
            threshold=threshold,
            window_size=window_size,
            steps_elapsed=current_step - self._frontier_start_step,
            minimum_steps=minimum_frontier_steps,
            maximum_station=station_count - 1,
        )
        if promoted:
            self._frontier = updated
            self._history.clear()
            self._frontier_start_step = current_step
        env._insert_station_frontier = self._frontier
        return {
            "frontier_station": float(self._frontier),
            "frontier_rolling_success": float(rolling),
            "frontier_control_steps": float(current_step - self._frontier_start_step),
        }


@configclass
class ReverseStationCurriculumCfg:
    station_frontier = CurrTerm(
        func=ReverseStationSuccessCurriculum,
        params={
            "success_term": "insertion_success",
            "start_station": START_FRONTIER_STATION,
            "threshold": 0.80,
            "window_size": 256,
            "minimum_frontier_steps": 1_600,
            "station_count": INSERT_STATION_COUNT,
        },
    )


@configclass
class ZeroGBladeGrapplePinInsertHandoffCurriculumEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg):
    """Expand from v24's solved late stations toward the chain handoff."""

    curriculum: ReverseStationCurriculumCfg = ReverseStationCurriculumCfg()

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        reset = getattr(self.events, "reset_stroke", None)
        if reset is None:
            raise ValueError("reverse insertion curriculum requires the solved stroke reset")
        reset.func = reset_reverse_station_curriculum
        reset.params.pop("forced_station", None)
        reset.params["station_count"] = INSERT_STATION_COUNT
        reset.params["frontier_probability"] = 0.50


__all__ = [
    "INSERT_STATION_COUNT",
    "ReverseStationSuccessCurriculum",
    "START_FRONTIER_STATION",
    "ZeroGBladeGrapplePinInsertHandoffCurriculumEnvCfg",
    "reset_reverse_station_curriculum",
]
