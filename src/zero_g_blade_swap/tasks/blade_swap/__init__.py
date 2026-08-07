"""Gym registration for the zero-g server blade swap task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insertion_env_cfg:ZeroGBladeInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insertion_env_cfg:ZeroGBladeInsertionPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_insertion_env_cfg:ZeroGBladeRobustInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_robust_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Robust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_insertion_env_cfg:ZeroGBladeRobustInsertionPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_robust_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contact_insertion_env_cfg:ZeroGBladeContactInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Contact-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contact_insertion_env_cfg:ZeroGBladeContactInsertionPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-RigidGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rigid_grasp_insertion_env_cfg:ZeroGBladeRigidGraspInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-RigidGrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rigid_grasp_insertion_env_cfg:ZeroGBladeRigidGraspInsertionPlayEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

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
