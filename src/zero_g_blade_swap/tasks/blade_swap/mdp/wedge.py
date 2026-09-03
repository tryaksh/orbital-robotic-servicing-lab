"""The wedge law, as something the insert policy can see and can end on.

**The insert skill's depth limit is its attitude, and that is measured, not
argued.** `evidence/insert_depth_is_attitude.json` records the two populations:
the episodes that seat hold 46.9 mrad and arrive 0.8 mm short, and the episodes
that stall hold 96.8 mrad and stop 174.5 mm short -- and 2c/theta at 96.8 mrad
in this bay's relieved channel is 261 mm, which is the travel the stalled
episodes actually achieve. They are not stopping short of a depth they could
reach. They are as deep as their own attitude permits.

Everything tried since has left that untouched. Three objectives -- a baseline
time cost, a 4x time cost trained to convergence, and a 7x orientation penalty --
ended 0.4 mrad apart, so the attitude is not the reward's to give. The load path
was matched, the reset bank was matched, the action scaling was corrected, the
controller was projected onto module-relative state, and the hand-off station was
curriculumed backwards. The chain still keeps the scripted guarded advance.

What has never been tried is the one structural difference between the two
controllers. **The guarded advance does not push a cocked module**: it steps only
while the deployed estimate is inside the entry envelope, and holds otherwise.
The skill has no such interlock, so it can drive the module past the depth its
own attitude admits -- and once it has, every remaining step of the episode is
spent in a state no action recovers, still generating gradient as though it were
recoverable.

Two terms, both derived from the same law and from clearances measured out of
the built configuration rather than chosen:

* :func:`wedge_margin` is how much further the module may go at the attitude it
  currently holds. The policy can already see attitude and depth separately, in
  ``blade_goal_error``; this is the combination that decides its fate, which is
  a different thing from the parts.
* :func:`wedged` ends the episode when the module has passed that depth. This is
  the change that matters, and it is an MDP change rather than a reward-scale
  one: it removes unrecoverable states from the return, so the credit for
  wedging lands on the steps that cocked the module instead of being spread over
  a thousand steps of pushing a part that cannot move.

Neither term reads anything the chain does not also have. Attitude and depth are
what the deployed estimator reports, and the clearance is a property of the rack.
"""

from __future__ import annotations

import torch

from zero_g_blade_swap.grapple_geometry import EXTRACTED_BLADE_CENTRE_X

from .insertion import attached_blade_pose_world, insertion_error_metrics

#: The destination bay's clearances per side **with** its relief, measured out of
#: the built configuration in ``evidence/destination_channel_geometry.json`` and
#: repeated by ``scripts/report_insert_depth_limit.py``. The tighter of the two
#: is the one that wedges first, which is what the law needs.
RELIEVED_LATERAL_CLEARANCE_M = 0.015678
RELIEVED_VERTICAL_CLEARANCE_M = 0.012613
WEDGE_CLEARANCE_M = min(RELIEVED_LATERAL_CLEARANCE_M, RELIEVED_VERTICAL_CLEARANCE_M)

#: Below this attitude the admissible depth exceeds the whole stroke and the law
#: says nothing useful, so the margin saturates instead of dividing by zero.
MINIMUM_ATTITUDE_RAD = 1.0e-3

#: The margin is a length in metres and the stroke is about half of one. Clamping
#: keeps a near-square module from handing the policy an enormous number, which
#: is the same reason every other observation here is bounded.
MARGIN_CLAMP_M = 0.60


def _engagement_depth(env) -> torch.Tensor:
    """How far the module has entered the destination channel, from its mouth."""

    position, _ = attached_blade_pose_world(env)
    local_x = position[:, 0] - env.scene.env_origins[:, 0]
    return (local_x - EXTRACTED_BLADE_CENTRE_X).clamp_min(0.0)


def admissible_depth(env, command_name: str = "insertion_goal") -> torch.Tensor:
    """Return ``2c/theta``: how deep this attitude may go before it wedges."""

    _, _, attitude = insertion_error_metrics(env, command_name)
    return (2.0 * WEDGE_CLEARANCE_M) / attitude.clamp_min(MINIMUM_ATTITUDE_RAD)


def wedge_margin(env, command_name: str = "insertion_goal") -> torch.Tensor:
    """Return the remaining admissible travel at the attitude currently held."""

    margin = admissible_depth(env, command_name) - _engagement_depth(env)
    return margin.clamp(-MARGIN_CLAMP_M, MARGIN_CLAMP_M).unsqueeze(-1)


def wedged(env, tolerance_m: float = 0.010, command_name: str = "insertion_goal") -> torch.Tensor:
    """End the episode once the module is deeper than its attitude admits.

    ``tolerance_m`` is one seated-depth tolerance below the law's own bound, so
    an episode that is merely at the limit is not ended for being at it. It is
    the same 10 mm slack the seating predicate already carries and is not tuned
    here.
    """

    return (admissible_depth(env, command_name) + tolerance_m) < _engagement_depth(env)


__all__ = [
    "MARGIN_CLAMP_M",
    "MINIMUM_ATTITUDE_RAD",
    "RELIEVED_LATERAL_CLEARANCE_M",
    "RELIEVED_VERTICAL_CLEARANCE_M",
    "WEDGE_CLEARANCE_M",
    "admissible_depth",
    "wedge_margin",
    "wedged",
]
