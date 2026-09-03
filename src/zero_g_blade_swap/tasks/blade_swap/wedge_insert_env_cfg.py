"""The seating task with the wedge law as a terminal condition.

**One change from the task `v24rack` was trained on, and it is not a reward.**
The insert skill's depth limit is its attitude through `2c/theta`: the episodes
that stall hold 96.8 mrad and reach 261 mm, which is exactly what that attitude
admits in this bay's relieved channel. Three different objectives left the angle
0.4 mrad apart, so the attitude is not the reward's to give, and this repository
has already spent six checkpoints establishing that.

What the scripted guarded advance does and the skill does not is *refuse to push
a cocked module*. It steps only while the estimate is inside the entry envelope
and holds otherwise. The skill can drive the module past the depth its attitude
admits, and once it has, the remaining thousand-odd control steps are spent in a
state no action recovers -- still generating gradient as if it were recoverable,
and still charging the same time cost whether the module was cocked on step 40 or
on step 800.

`mdp.wedged` ends the episode there. That is an MDP change rather than a
reward-scale change: it removes unrecoverable states from the return, so the
credit for wedging lands on the steps that cocked the module. The observation
space is deliberately untouched, for two reasons -- the policy can already see
attitude and depth in ``blade_goal_error``, so a margin term would add no
information, and an unchanged observation width is what lets this resume the
frozen `v24rack` weights instead of starting over.

`mdp.wedge_margin` exists for diagnostics and is not wired here. Wiring it would
be the second arm, and it would have to be trained from scratch.

Registered separately, like every other change to this scene, so
`Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0` and every certificate taken on it
stay exactly what they were.
"""

from __future__ import annotations

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .two_slot_env_cfg import (
    ZeroGBladeGrapplePinInsertTwoSlotEnvCfg,
    ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg,
)


@configclass
class WedgeGatedInsertTerminationsCfg:
    """The two-slot seating terminations, plus the wedge.

    Restated in full rather than subclassed so that a reader can see there is
    exactly one addition, and so that a change to the base task's set cannot
    silently arrive here without a diff.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insertion_success = DoneTerm(func=mdp.grapple_insertion_success_mask)
    extraction_failed = DoneTerm(func=mdp.extraction_failure)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)
    # The addition. Not a timeout: an episode that ends here has failed, and the
    # value function should learn it as a failure rather than as a clock.
    wedged = DoneTerm(func=mdp.wedged)


@configclass
class ZeroGBladeGrapplePinInsertWedgeGatedEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg):
    terminations: WedgeGatedInsertTerminationsCfg = WedgeGatedInsertTerminationsCfg()


@configclass
class ZeroGBladeGrapplePinInsertWedgeGatedPlayEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg):
    terminations: WedgeGatedInsertTerminationsCfg = WedgeGatedInsertTerminationsCfg()


__all__ = [
    "WedgeGatedInsertTerminationsCfg",
    "ZeroGBladeGrapplePinInsertWedgeGatedEnvCfg",
    "ZeroGBladeGrapplePinInsertWedgeGatedPlayEnvCfg",
]
