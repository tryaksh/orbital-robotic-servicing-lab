"""Insertion training whose reset distribution matches the real chain caller.

The promoted two-slot task samples nine solved points uniformly along the
stroke. Conditioned evaluation showed that v24 succeeds only on the late part
of that mixture and is 0/768 at stations 0--3 and 0/96 at real handoffs. This
separate task changes exactly one variable: every reset starts at station 0,
the rack mouth where the chain transfers control. Geometry, observations,
actions, rewards, success criteria, phase budget, and fixed-to-compliant load
path are inherited unchanged.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .two_slot_env_cfg import ZeroGBladeGrapplePinInsertTwoSlotEnvCfg

HANDOFF_RESET_STATION = 0


@configclass
class ZeroGBladeGrapplePinInsertHandoffEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg):
    """Train only from the insertion state the predecessor actually supplies."""

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        reset = getattr(self.events, "reset_stroke", None)
        if reset is None:
            raise ValueError("handoff-conditioned insertion requires the solved stroke reset")
        reset.params["forced_station"] = HANDOFF_RESET_STATION


__all__ = ["HANDOFF_RESET_STATION", "ZeroGBladeGrapplePinInsertHandoffEnvCfg"]
