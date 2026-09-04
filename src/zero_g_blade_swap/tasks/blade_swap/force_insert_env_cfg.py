"""The seating task, with the policy allowed to feel the contact it is making.

**The learned seating phase has been doing contact-rich assembly blind for this
project's entire history, and that is the finding.** Ten checkpoints, three
objectives, a matched load path, a matched reset bank, corrected action scaling,
a projected observation space and a reverse station curriculum -- and in none of
them could the policy observe contact force. `BladeContactWrenchObservation` has
existed in this repository since the force-limited insertion work and was never
wired to the skill the chain actually runs.

The literature is unambiguous that this is the missing ingredient class. FORGE
(arXiv 2408.04587) takes a noisy end-effector force estimate as a policy input,
adds a force threshold and dynamics randomization, and transfers contact-rich
insertion zero-shot while tolerating **up to 5 mm of fixed-part pose error**.
This project's estimator is accurate to about 2 mm and its chain still collapses
on camera-derived state, which says the gap is what the policy is allowed to
sense, not what the sensor delivers.

**One change, and it is the observation.** Rewards, terminations, tolerances, the
curriculum, the load path, the action space, the reset distribution and the phase
budget are the ones `v24rack` trained under. The blade gains a contact sensor and
the policy gains seven values: the instantaneous contact force in tool axes, the
same force through a first-order filter standing in for a real signal chain, and
the scalar magnitude. Nothing is added to the reward, deliberately -- three
objectives already failed to move the terminal attitude, and the question this
task asks is whether the angle was ever the *reward's* to give or whether the
policy simply could not perceive the thing it was being scored on.

The observation width changes, so this cannot resume `v24rack` and trains from
scratch. That is the cost of the answer.

**What the two outcomes mean.** If the attitude comes down, the negative result
that has stood since the beginning is wrong in an informative way: an angle that
would not move under three objectives moves when the policy is allowed to feel
contact, and the paper reports a learned seating phase. If it does not, the
interface bound survives its strongest remaining challenge and
`docs/seating_controller.md` is a much stronger argument for having been tested
this way.
"""

from __future__ import annotations

import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .grapple_pin_env_cfg import InsertPolicyObsCfg
from .scene_cfg import ZeroGTwoSlotGrapplePinSceneCfg
from .two_slot_env_cfg import (
    ZeroGBladeGrapplePinInsertTwoSlotEnvCfg,
    ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg,
    ZeroGBladeGrapplePinTwoSlotWorkflowEnvCfg,
)
from .workflow_demo_env_cfg import WorkflowInsertObsCfg, WorkflowObservationsCfg

#: Divides the seven force channels. Twenty newtons is the scale the force-limited
#: insertion work chose against a measured median contact of 4.7 N, and it is
#: reused rather than re-picked so the two families report force on one scale.
FORCE_OBSERVATION_SCALE_N = 20.0

#: The sensor's noise floor. FORGE models roughly 1 N and gives the noisy signal
#: to the actor; this defaults to zero so a first result describes the idealized
#: sensor it was produced under, and the noisy arm is a separate, later change.
FORCE_OBSERVATION_NOISE_N = 0.0


@configclass
class ZeroGTwoSlotGrapplePinForceSceneCfg(ZeroGTwoSlotGrapplePinSceneCfg):
    """The chain's destination scene, with the module's contacts reported.

    ``update_period=0.0`` samples every physics step and ``history_length=0``
    keeps only the current value, which is what the observation term filters
    itself. Both match the force-limited insertion scene this borrows from.
    """

    blade_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SpareBlade",
        update_period=0.0,
        history_length=0,
    )

    def __post_init__(self) -> None:
        """Turn the module's contact reporter on, for this scene only.

        `GRAPPLE_PIN_BLADE_CFG` ships with `activate_contact_sensors=False`, and
        a contact sensor on a prim without the reporter API raises rather than
        returning zeros -- which is the right failure and is how this was found.
        The shared asset is deep-copied rather than mutated, because every
        published grapple-pin task reads the same object and none of them should
        acquire a contact reporter by side effect.

        This is a second difference from the baseline task and is disclosed as
        one. It is a *reporting* change: enabling the API does not alter contact
        dynamics, only whether they can be read.
        """

        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        blade = copy.deepcopy(self.spare_blade)
        blade.spawn.activate_contact_sensors = True
        self.spare_blade = blade


@configclass
class ForceFeedbackInsertObsCfg(InsertPolicyObsCfg):
    """The seating policy's observations, plus the contact it is making."""

    contact_wrench = ObsTerm(
        func=mdp.BladeContactWrenchObservation,
        params={
            "force_scale_n": FORCE_OBSERVATION_SCALE_N,
            "noise_std_n": FORCE_OBSERVATION_NOISE_N,
        },
    )


@configclass
class ForceFeedbackInsertObservationsCfg:
    policy: ForceFeedbackInsertObsCfg = ForceFeedbackInsertObsCfg()


@configclass
class ZeroGBladeGrapplePinInsertForceEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg):
    scene: ZeroGTwoSlotGrapplePinForceSceneCfg = ZeroGTwoSlotGrapplePinForceSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    observations: ForceFeedbackInsertObservationsCfg = ForceFeedbackInsertObservationsCfg()


@configclass
class ZeroGBladeGrapplePinInsertForcePlayEnvCfg(ZeroGBladeGrapplePinInsertForceEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg.__post_init__(self)


# ---------------------------------------------------------------------------
# The chain, with the same force channel, so the policy can actually be used.
#
# **A force-feedback seating policy cannot be dropped into the published chain**:
# that task's scene has no contact sensor and its insert observation group has no
# force channel, so the policy would be handed an observation of the wrong width.
# Training it without this would have produced a checkpoint that could only ever
# be certified in isolation -- which is the exact failure `verify_insert_skill.sh`
# exists to catch, and it would have been caught after the GPU was spent.
#
# Registered separately, so `Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0` and
# every number measured on it stay what they were.
# ---------------------------------------------------------------------------


@configclass
class ForceFeedbackWorkflowInsertObsCfg(WorkflowInsertObsCfg):
    """The chain's insert-phase input, plus the contact the module is making."""

    contact_wrench = ObsTerm(
        func=mdp.BladeContactWrenchObservation,
        params={
            "force_scale_n": FORCE_OBSERVATION_SCALE_N,
            "noise_std_n": FORCE_OBSERVATION_NOISE_N,
        },
    )


@configclass
class ForceFeedbackWorkflowObsCfg(WorkflowObservationsCfg):
    """Capture and extraction unchanged; only the seating group gains force."""

    insert: ForceFeedbackWorkflowInsertObsCfg = ForceFeedbackWorkflowInsertObsCfg()


@configclass
class ZeroGBladeGrapplePinTwoSlotWorkflowForceEnvCfg(ZeroGBladeGrapplePinTwoSlotWorkflowEnvCfg):
    scene: ZeroGTwoSlotGrapplePinForceSceneCfg = ZeroGTwoSlotGrapplePinForceSceneCfg(
        num_envs=8,
        env_spacing=2.6,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    observations: ForceFeedbackWorkflowObsCfg = ForceFeedbackWorkflowObsCfg()


__all__ = [
    "FORCE_OBSERVATION_NOISE_N",
    "ForceFeedbackWorkflowInsertObsCfg",
    "ForceFeedbackWorkflowObsCfg",
    "ZeroGBladeGrapplePinTwoSlotWorkflowForceEnvCfg",
    "FORCE_OBSERVATION_SCALE_N",
    "ForceFeedbackInsertObsCfg",
    "ForceFeedbackInsertObservationsCfg",
    "ZeroGBladeGrapplePinInsertForceEnvCfg",
    "ZeroGBladeGrapplePinInsertForcePlayEnvCfg",
    "ZeroGTwoSlotGrapplePinForceSceneCfg",
]
