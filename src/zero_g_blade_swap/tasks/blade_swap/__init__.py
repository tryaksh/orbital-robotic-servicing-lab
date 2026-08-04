"""Gym registration for the zero-g server blade swap task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-ZeroG-BladeSwap-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ZeroGBladeSwapTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-BladeSwap-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ZeroGBladeSwapVisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vision.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-BladeSwap-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ZeroGBladeSwapPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher.yaml",
    },
)


__all__ = []
