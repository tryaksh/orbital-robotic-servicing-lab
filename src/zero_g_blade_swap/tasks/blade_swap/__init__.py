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

for _guided_id, _guided_cls in (
    ("Isaac-ZeroG-Blade-Insertion-GuidedSlot-v0", "ZeroGBladeGuidedSlotEnvCfg"),
    ("Isaac-ZeroG-Blade-Insertion-GuidedSlot-Play-v0", "ZeroGBladeGuidedSlotPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-CaptureInSlot-v0", "ZeroGBladeCaptureInSlotEnvCfg"),
    ("Isaac-ZeroG-Blade-CaptureInSlot-Play-v0", "ZeroGBladeCaptureInSlotPlayEnvCfg"),
):
    gym.register(
        id=_guided_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.guided_slot_env_cfg:{_guided_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rigid_grasp.yaml",
        },
    )

# Only the capture scene survives of the head-on grapple-pin work. The three
# skills trained on it (grasp, extract, insert) were deleted on 2026-08-10; this
# is the scene scripts/grasp_diagnostics.py measures the interface against.
for _grapple_id, _grapple_cls in (
    ("Isaac-ZeroG-Blade-GrapplePin-Capture-v0", "ZeroGBladeGrapplePinCaptureEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Capture-Play-v0", "ZeroGBladeGrapplePinCapturePlayEnvCfg"),
):
    gym.register(
        id=_grapple_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.grapple_pin_env_cfg:{_grapple_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
        },
    )

# The camera and the visual randomizers were reachable only from the deleted
# eight-phase swap task. They are repointed here, at the insertion scene. No
# policy has been trained on this task; it is P3's scaffold.
for _vision_id, _vision_cls in (
    ("Isaac-ZeroG-Blade-Insertion-Vision-v0", "ZeroGBladeVisionInsertionEnvCfg"),
    ("Isaac-ZeroG-Blade-Insertion-Vision-Play-v0", "ZeroGBladeVisionInsertionPlayEnvCfg"),
):
    gym.register(
        id=_vision_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.vision_insertion_env_cfg:{_vision_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vision.yaml",
        },
    )


__all__ = []
