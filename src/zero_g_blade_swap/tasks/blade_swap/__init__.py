"""Gym registration for the zero-g server blade swap task."""

import gymnasium as gym

from . import agents

# Insertion tasks use a subclass that can snapshot terminal metrics before
# Isaac Lab automatically resets a finished episode.  The hook stays inert
# unless an evaluator enables it, so training behaviour is unchanged.
INSERTION_ENTRY_POINT = f"{__name__}.terminal_metrics_env:TerminalMetricsManagerBasedRLEnv"

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insertion_env_cfg:ZeroGBladeInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Play-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insertion_env_cfg:ZeroGBladeInsertionPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Robust-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_insertion_env_cfg:ZeroGBladeRobustInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_robust_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Robust-Play-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_insertion_env_cfg:ZeroGBladeRobustInsertionPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_robust_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Contact-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contact_insertion_env_cfg:ZeroGBladeContactInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-Contact-Play-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contact_insertion_env_cfg:ZeroGBladeContactInsertionPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-RigidGrasp-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rigid_grasp_insertion_env_cfg:ZeroGBladeRigidGraspInsertionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-RigidGrasp-Play-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rigid_grasp_insertion_env_cfg:ZeroGBladeRigidGraspInsertionPlayEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-ForceLimited-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.force_limited_insertion_env_cfg:ZeroGBladeForceLimitedInsertionEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-ForceLimited-Play-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.force_limited_insertion_env_cfg:ZeroGBladeForceLimitedInsertionPlayEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-StrictForceLimited-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.force_limited_insertion_env_cfg:ZeroGBladeStrictForceLimitedInsertionEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-StrictForceLimited-Play-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.force_limited_insertion_env_cfg:ZeroGBladeStrictForceLimitedInsertionPlayEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-ForceFeedback-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.force_limited_insertion_env_cfg:ZeroGBladeForceFeedbackInsertionEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-Insertion-ForceFeedback-Play-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.force_limited_insertion_env_cfg:ZeroGBladeForceFeedbackInsertionPlayEnvCfg"
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
