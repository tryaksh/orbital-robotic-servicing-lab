"""One episode that runs the whole servicing motion, for demonstration only.

The three head-on grapple-pin skills are trained and certified separately, which
is right: "did the grasp form", "is the module clear of the rack", and "is it
seated" are three different questions and one blended reward would let a policy
trade them against each other. But a servicing demonstration has to show them in
sequence, in one continuous episode, on one blade.

This profile is that episode. It changes **no physics**: same scene, same robot,
same pin, same contacts as the grasp task. What it changes is bookkeeping —
the skill terminations are removed so the episode does not end when a phase
succeeds, and it runs long enough for all of them.

Three things about the design are worth stating, because they are what makes the
demonstration honest.

*The policies are the trained ones.* ``scripts/run_workflow_demo.py`` loads the
three checkpoints and switches between them on measured conditions, never on a
timer. No phase is scripted open-loop except the transit below.

*The transit is deliberately scripted, and says so.* Extraction ends with the
module clear of the rack at x = 0.225 and insertion was trained from the
certified staging pose at x = 0.583. The 358 mm between them is free space with
no contact in it, so there is nothing there for a policy to learn; a scripted
axial move is the honest and cheaper answer, and it is the same division of
labour the rest of this project uses — deterministic motion where it is cheap,
learned control where contact and uncertainty live.

*Each phase keeps the action scale its policy trained under.* The three skills
use different Cartesian scales, 2 mm per step for the capture against 8 mm for
the pull, because they are different motions. Running one policy at another's
scale would silently change what its actions mean, so the driver rewrites the
action term's scale at each transition.

This task must never be used for training. It has no curriculum, no success
termination, and therefore nothing to optimise.
"""

from __future__ import annotations

import dataclasses

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .grapple_pin_env_cfg import (
    GrappleSkillObsCfg,
    GraspActionsCfg,
    GraspEventsCfg,
    ZeroGBladeGrapplePinExtractEnvCfg,
    ZeroGBladeGrapplePinGraspEnvCfg,
    ZeroGBladeGrapplePinInsertEnvCfg,
)
from .robust_insertion_env_cfg import configure_insertion_play_presentation


#: Cartesian action scales, in the order the workflow runs them. Each is *read*
#: from the task the corresponding policy was trained on -- never copied -- and
#: the driver writes the matching one into the action term at every transition.
def _certified_action_scale(cfg_class: type) -> tuple[float, ...]:
    """Read a skill's action scale off its own task configuration.

    Copied constants drift, and this file proved it the expensive way. Extract's
    scales were rebalanced on the task -- lateral 0.001 to 0.004 and rotation
    0.008 to 0.020, because the module was measured rotating faster than the
    wrist could follow -- and the copy here stayed at the old values. The chain
    then drove a policy at a quarter of the lateral authority it had trained
    with, and the removal workflow went to 0.00% with every one of 598 episodes
    overrunning its budget, while the same checkpoint certified at 94.23% on
    that stage running alone.

    ``PHASE_BUDGET_S`` in ``run_workflow_demo.py`` already derives the episode
    lengths for exactly this reason. This is the same fix for the same class of
    bug: a policy must be driven by the action term it was trained against, and
    the only way to guarantee that is to read it rather than restate it.
    """

    for field in dataclasses.fields(cfg_class):
        if field.name == "actions":
            actions = field.default_factory() if field.default is dataclasses.MISSING else field.default
            return tuple(float(value) for value in actions.arm.scale)
    raise AttributeError(f"{cfg_class.__name__} declares no actions")


GRASP_ACTION_SCALE = _certified_action_scale(ZeroGBladeGrapplePinGraspEnvCfg)
EXTRACT_ACTION_SCALE = _certified_action_scale(ZeroGBladeGrapplePinExtractEnvCfg)
INSERT_ACTION_SCALE = _certified_action_scale(ZeroGBladeGrapplePinInsertEnvCfg)


def certified_episode_length_s(cfg_class: type) -> float:
    """Read a skill's episode length off its own task configuration.

    Through the dataclass field rather than the class attribute, because
    ``configclass`` rewrites these into fields and the attribute is not there to
    read. Going through the task class at all is the point: a constant copied
    elsewhere would be free to drift away from what the skill is certified on,
    which is the exact failure the phase budgets exist to prevent.
    """

    for field in dataclasses.fields(cfg_class):
        if field.name == "episode_length_s":
            if field.default is not dataclasses.MISSING:
                return float(field.default)
            return float(field.default_factory())
    raise AttributeError(f"{cfg_class.__name__} declares no episode_length_s")


#: Seconds each learned phase gets, read from the task each policy was certified
#: on so the two can never drift apart. Lives here rather than in the driver
#: because the chained-insert *training* task needs the same two numbers, and a
#: training task that gives a skill a different clock than its certification is
#: the defect these were introduced to prevent.
CAPTURE_BUDGET_S = certified_episode_length_s(ZeroGBladeGrapplePinGraspEnvCfg)
EXTRACT_BUDGET_S = certified_episode_length_s(ZeroGBladeGrapplePinExtractEnvCfg)
INSERT_BUDGET_S = certified_episode_length_s(ZeroGBladeGrapplePinInsertEnvCfg)

#: Control steps spent letting the closure drive the pin against its collar
#: before the next skill starts. One second, which is what the pull gate needed
#: to settle and what the extract task's own action term waits out.
SEAT_STEPS = 30
#: How long a qualifying capture must hold before control is handed over. The
#: same 0.30 s ``capture_success_mask`` requires, so the chain and the capture
#: skill cannot disagree about what "captured" means.
HANDOVER_HOLD_S = 0.30

#: Where the scripted transit hands over to the insert policy: the blade centre
#: the insert skill was trained to start from.
TRANSIT_TARGET_BLADE_X = 0.5829


@configclass
class WorkflowGraspObsCfg(GrappleSkillObsCfg):
    """Exactly the grasp policy's input. Its ``previous_action`` is the full
    seven-value command, because the grasp task owned the gripper too."""


@configclass
class WorkflowExtractObsCfg(GrappleSkillObsCfg):
    """Exactly the extract policy's input.

    Two things have to match the training task rather than this one. The
    previous action is the arm's six values, because the extract task did not
    command the gripper, and ``remaining_travel`` is appended last, which is
    where subclassing put it during training.
    """

    previous_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm"})
    remaining_travel = ObsTerm(func=mdp.extraction_remaining_observation)


@configclass
class WorkflowInsertObsCfg(GrappleSkillObsCfg):
    """Exactly the insert policy's input."""

    previous_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm"})
    blade_goal_error = ObsTerm(func=mdp.insertion_goal_error)


@configclass
class WorkflowObservationsCfg:
    """All three policies' inputs, computed every step.

    The driver reads whichever group belongs to the phase it is in. Computing
    all three costs a few tensor operations and removes any chance of feeding a
    policy an observation assembled in the wrong order, which is the class of
    bug that has cost this project the most time.
    """

    grasp: WorkflowGraspObsCfg = WorkflowGraspObsCfg()
    extract: WorkflowExtractObsCfg = WorkflowExtractObsCfg()
    insert: WorkflowInsertObsCfg = WorkflowInsertObsCfg()


@configclass
class WorkflowTerminationsCfg:
    """Nothing ends this episode except time or a broken simulation.

    The skill terminations are deliberately absent. A capture that ends the
    episode is correct when certifying a grasp and useless when the point is to
    carry on and pull the module out.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class WorkflowCurriculumCfg:
    """Empty. There is nothing to learn here and nothing to promote."""


@configclass
class WorkflowRewardsCfg:
    """Empty, and it has to be.

    The grasp task's reward set reads the ``capture_failed`` termination, which
    this profile removes so the episode can continue past a capture. Leaving it
    in raises a KeyError on the first step. Nothing here is being optimised, so
    there is no reason to compute a reward at all.
    """


@configclass
class ZeroGBladeGrapplePinWorkflowEnvCfg(ZeroGBladeGrapplePinGraspEnvCfg):
    """Capture, extract, transit, and re-insert, in one episode."""

    observations: WorkflowObservationsCfg = WorkflowObservationsCfg()
    actions: GraspActionsCfg = GraspActionsCfg()
    events: GraspEventsCfg = GraspEventsCfg()
    rewards: WorkflowRewardsCfg = WorkflowRewardsCfg()
    terminations: WorkflowTerminationsCfg = WorkflowTerminationsCfg()
    curriculum: WorkflowCurriculumCfg = WorkflowCurriculumCfg()
    # Generous: the capture takes about 2 s, the 495 mm pull and the 358 mm
    # transit a few more each, and the insertion is the slowest motion in the
    # project. A demonstration that runs out of time mid-insertion is worse than
    # one that idles at the end.
    episode_length_s: float = 45.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # The parent rebuilds the event set and re-applies the grasp task's
        # widened reset noise, which is what this demonstration wants: the
        # capture has to be a real approach, not a reset that already solved it.
        self.rewards = WorkflowRewardsCfg()
        self.terminations = WorkflowTerminationsCfg()
        self.curriculum = WorkflowCurriculumCfg()


__all__ = [
    "CAPTURE_BUDGET_S",
    "EXTRACT_ACTION_SCALE",
    "EXTRACT_BUDGET_S",
    "GRASP_ACTION_SCALE",
    "HANDOVER_HOLD_S",
    "INSERT_ACTION_SCALE",
    "INSERT_BUDGET_S",
    "SEAT_STEPS",
    "TRANSIT_TARGET_BLADE_X",
    "certified_episode_length_s",
    "WorkflowExtractObsCfg",
    "WorkflowGraspObsCfg",
    "WorkflowInsertObsCfg",
    "WorkflowObservationsCfg",
    "WorkflowRewardsCfg",
    "WorkflowTerminationsCfg",
    "ZeroGBladeGrapplePinWorkflowEnvCfg",
]
