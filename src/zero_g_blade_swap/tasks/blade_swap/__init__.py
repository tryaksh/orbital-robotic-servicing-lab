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

# Insertion under a wrong pose belief, and its matched force-blind ablation.
# One PPO configuration for both: the experiment is only valid if the single
# difference between them is what the actor can observe.
for _uncertain_id, _uncertain_cls in (
    ("Isaac-ZeroG-Blade-Insertion-Uncertain-v0", "ZeroGBladeUncertainInsertionEnvCfg"),
    ("Isaac-ZeroG-Blade-Insertion-Uncertain-Play-v0", "ZeroGBladeUncertainInsertionPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-Insertion-UncertainBlind-v0", "ZeroGBladeUncertainInsertionBlindEnvCfg"),
    ("Isaac-ZeroG-Blade-Insertion-UncertainBlind-Play-v0", "ZeroGBladeUncertainInsertionBlindPlayEnvCfg"),
    # Evaluation-only: the same policies with the lead-in flares removed, which
    # tests whether the ramp rather than the sensing was doing the alignment.
    (
        "Isaac-ZeroG-Blade-Insertion-UncertainNoLeadIn-Play-v0",
        "ZeroGBladeUncertainInsertionNoLeadInPlayEnvCfg",
    ),
    (
        "Isaac-ZeroG-Blade-Insertion-UncertainBlindNoLeadIn-Play-v0",
        "ZeroGBladeUncertainInsertionBlindNoLeadInPlayEnvCfg",
    ),
):
    gym.register(
        id=_uncertain_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.uncertain_insertion_env_cfg:{_uncertain_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_uncertain_insertion.yaml",
        },
    )

# The head-on grapple-pin work: the capture scene the interface specification was
# measured against, and the three skills a replacement demonstration needs.
#
# The skills were deleted on 2026-08-10 and restored on 2026-08-11. Deleting them
# was defensible on the evidence at the time — all three had failed — but it also
# removed the only path to a chained replacement demonstration, and each had
# failed for a cause that was identified and corrected in the same session and
# then never retested. They are back to be retested.
for _grapple_id, _grapple_cls in (
    ("Isaac-ZeroG-Blade-GrapplePin-Capture-v0", "ZeroGBladeGrapplePinCaptureEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Capture-Play-v0", "ZeroGBladeGrapplePinCapturePlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Grasp-v0", "ZeroGBladeGrapplePinGraspEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Grasp-Play-v0", "ZeroGBladeGrapplePinGraspPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Extract-v0", "ZeroGBladeGrapplePinExtractEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0", "ZeroGBladeGrapplePinExtractPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Insert-v0", "ZeroGBladeGrapplePinInsertEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-Insert-Play-v0", "ZeroGBladeGrapplePinInsertPlayEnvCfg"),
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
