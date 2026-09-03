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

# The same three skills, with the module state they see coming from the deployed
# estimator's measured error distribution instead of from the simulator.
#
# The chain scores 20/24 on the vision task when the module pose comes from the
# simulator and 4/24 when the same code path reads it from the cameras, so the
# 67-point step is the estimator and nothing else. It is not a perception
# defect -- the estimator's own error on healthy episodes is about 2 mm -- it is
# three policies deployed against an error distribution that was never in their
# training data. These registrations put it there.
#
# Separate ids, because the published skill certificates were measured on exact
# state and both arms have to stay runnable for the difference to mean anything.
for _noised_id, _noised_cls in (
    ("Isaac-ZeroG-Blade-GrapplePin-GraspNoised-v0", "ZeroGBladeGrapplePinGraspNoisedEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-GraspNoised-Play-v0", "ZeroGBladeGrapplePinGraspNoisedPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-v0", "ZeroGBladeGrapplePinExtractNoisedEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-Play-v0", "ZeroGBladeGrapplePinExtractNoisedPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-InsertNoised-v0", "ZeroGBladeGrapplePinInsertNoisedEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-InsertNoised-Play-v0", "ZeroGBladeGrapplePinInsertNoisedPlayEnvCfg"),
    # The two halves of the channel-isolation experiment. Extraction is where
    # the camera-driven chain loses thirteen of twenty-four episodes, and the
    # pose residual and the manufactured velocity are separable causes.
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractPoseNoised-v0", "ZeroGBladeGrapplePinExtractPoseNoisedEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractPoseNoised-Play-v0", "ZeroGBladeGrapplePinExtractPoseNoisedPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractVelocityNoised-v0", "ZeroGBladeGrapplePinExtractVelocityNoisedEnvCfg"),
    # The in-loop tail as its own arm: the first arm's constants are inverted
    # from the still-frame certificate, and this one's are calibrated against
    # what the estimator records inside a closed loop.
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractNoisedTail-v0", "ZeroGBladeGrapplePinExtractNoisedTailEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractNoisedTail-Play-v0", "ZeroGBladeGrapplePinExtractNoisedTailPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-ExtractVelocityNoised-Play-v0", "ZeroGBladeGrapplePinExtractVelocityNoisedPlayEnvCfg"),
):
    gym.register(
        id=_noised_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.estimator_noise_env_cfg:{_noised_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
        },
    )

# Insert, trained inside the chain. Its own entry point, because the capture
# phase runs inside the environment: the episode resets the capture scene, steps
# the frozen capture policy until the chain's hand-off predicate fires, latches
# the grip, and only then hands the arm to the policy being trained.
#
# A separate registration, like every other change to this scene, so the single
# slot skills and their certifications are untouched.
for _chain_id, _chain_cls in (
    ("Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0", "ZeroGBladeGrapplePinInsertChainEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-InsertChain-Play-v0", "ZeroGBladeGrapplePinInsertChainPlayEnvCfg"),
    # The same task with the attitude term reweighted, registered separately so
    # the two runs stay exactly one change apart and both stay reproducible.
    ("Isaac-ZeroG-Blade-GrapplePin-InsertChainAttitude-v0", "ZeroGBladeGrapplePinInsertChainAttitudeEnvCfg"),
    (
        "Isaac-ZeroG-Blade-GrapplePin-InsertChainAttitude-Play-v0",
        "ZeroGBladeGrapplePinInsertChainAttitudePlayEnvCfg",
    ),
):
    gym.register(
        id=_chain_id,
        entry_point=f"{__name__}.chained_insert_env_cfg:ChainedInsertEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.chained_insert_env_cfg:{_chain_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
        },
    )

# The second rack bay, and the relocation it makes possible. Separate
# registrations, so the single-slot tasks and their certifications describe
# unchanged scenes.
for _two_slot_id, _two_slot_cls in (
    ("Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0", "ZeroGBladeGrapplePinInsertTwoSlotEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-Play-v0", "ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg"),
    ("Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0", "ZeroGBladeGrapplePinTwoSlotWorkflowEnvCfg"),
    (
        "Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-Play-v0",
        "ZeroGBladeGrapplePinTwoSlotWorkflowPlayEnvCfg",
    ),
):
    gym.register(
        id=_two_slot_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.two_slot_env_cfg:{_two_slot_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
        },
    )

# The seating task with the wedge law as a terminal condition. One change from
# the task v24rack trained on, and it is not a reward: three objectives already
# left the attitude 0.4 mrad apart, so the angle is not the reward's to give.
# What the scripted guarded advance does and the skill does not is refuse to push
# a cocked module, and mdp.wedged ends the episode where 2c/theta says the module
# can go no further. The observation width is unchanged, which is what lets this
# resume the frozen v24rack weights.
for _wedge_id, _wedge_cls in (
    ("Isaac-ZeroG-Blade-GrapplePin-InsertWedgeGated-v0", "ZeroGBladeGrapplePinInsertWedgeGatedEnvCfg"),
    (
        "Isaac-ZeroG-Blade-GrapplePin-InsertWedgeGated-Play-v0",
        "ZeroGBladeGrapplePinInsertWedgeGatedPlayEnvCfg",
    ),
):
    gym.register(
        id=_wedge_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.wedge_insert_env_cfg:{_wedge_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
        },
    )

gym.register(
    id="Isaac-ZeroG-Blade-GrapplePin-InsertHandoff-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.handoff_insert_env_cfg:ZeroGBladeGrapplePinInsertHandoffEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-GrapplePin-InsertTaskSpaceHandoff-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.handoff_insert_env_cfg:"
            "ZeroGBladeGrapplePinInsertTaskSpaceHandoffEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

gym.register(
    id="Isaac-ZeroG-Blade-GrapplePin-InsertHandoffCurriculum-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.handoff_curriculum_env_cfg:"
            "ZeroGBladeGrapplePinInsertHandoffCurriculumEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

# The two-bay rack with a camera on it. A single bay can only ask the camera
# where the module is; two can ask which bay holds it, which is the question a
# servicer asks first and the one worth demonstrating. Registered separately so
# the single-bay vision certifications keep describing the scene they ran on.
for _vision_two_slot_id, _vision_two_slot_cls in (
    (
        "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Collect-v0",
        "ZeroGBladeGrappleVisionTwoSlotCollectEnvCfg",
    ),
    (
        "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0",
        "ZeroGBladeGrappleVisionTwoSlotWorkflowEnvCfg",
    ),
    # Installation into the first bay, on a two-bay rack. The vision arms measure
    # what perception costs, and that needs a manipulation task that completes;
    # the relocation does not yet, for reasons that have nothing to do with the
    # camera, so running the arms on it would compare three ways of failing.
    (
        "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Install-v0",
        "ZeroGBladeGrappleVisionTwoSlotInstallEnvCfg",
    ),
):
    gym.register(
        id=_vision_two_slot_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.vision_two_slot_env_cfg:{_vision_two_slot_cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
        },
    )

# One episode that runs capture, extraction, a scripted transit, and re-insertion
# with the three trained checkpoints. Demonstration only: no curriculum, no
# success termination, nothing to train against.
gym.register(
    id="Isaac-ZeroG-Blade-GrapplePin-Workflow-v0",
    entry_point=INSERTION_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.workflow_demo_env_cfg:ZeroGBladeGrapplePinWorkflowEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_contact_insertion.yaml",
    },
)

# The servicing workflow seen through a camera. The collect profile records what
# a camera would have seen while the certified checkpoints drive the workflow;
# the workflow profile replaces the ground-truth grip vector with one regressed
# from that camera. Same physics, same pin, same policies -- the only difference
# between the two is where the module's pose comes from.
for _grapple_vision_id, _grapple_vision_cls in (
    ("Isaac-ZeroG-Blade-GrappleVision-Collect-v0", "ZeroGBladeGrappleVisionCollectEnvCfg"),
    ("Isaac-ZeroG-Blade-GrappleVision-Workflow-v0", "ZeroGBladeGrappleVisionWorkflowEnvCfg"),
):
    gym.register(
        id=_grapple_vision_id,
        entry_point=INSERTION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.vision_grapple_env_cfg:{_grapple_vision_cls}",
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
