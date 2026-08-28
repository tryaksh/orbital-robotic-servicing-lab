"""Insertion training from a state the real chain actually handed over.

The first version of this task merely forced solved reset station zero and
called it a handoff. That changed axial depth, but it did not reproduce the
caller's arm state: the solved reset is perfectly head-on, while the physical
transit hands over a square module through the compliant lock with the wrist
about 55 mrad off and on a joint posture up to 0.44 rad away. The unchanged
v27 policy scored 64/64 from the solved reset and 0/32 from the real handoff.

This task uses one actual transit-to-insert row near the median of a clean
32-environment chain trace. The robot root is placed at the recorded carriage
endpoint and the paired arm/module state goes through the same physical reset
and deferred fixed-to-compliant load path as the normal task. Geometry,
observations, actions, rewards, success criteria and phase budget remain
inherited unchanged.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .assets import GRAPPLE_ROBOT_ROOT_POS
from .two_slot_env_cfg import ZeroGBladeGrapplePinInsertTwoSlotEnvCfg

HANDOFF_RESET_STATION = 0
# Source: artifacts/insert_v27actionscale/chain_handoff_seed4070/trace.npz
# SHA-256 CEC9E51E076486136E24484375B1C5D35E4181CDABD8AAF4258904000FAD6B31,
# clean source 97d1ab3f409ebe5b7f395e6457cb20fd613f0401, env 17. This row is
# nearest the componentwise median of the 31 successful transit-to-insert
# handoffs. The carriage endpoint is recovered from recorded tool world pose
# minus simulator-validated forward kinematics of the recorded joints.
RECORDED_HANDOFF_TRACE_SHA256 = "CEC9E51E076486136E24484375B1C5D35E4181CDABD8AAF4258904000FAD6B31"
RECORDED_HANDOFF_SOURCE_COMMIT = "97d1ab3f409ebe5b7f395e6457cb20fd613f0401"
RECORDED_HANDOFF_ROBOT_ROOT_Y_M = -0.239
RECORDED_HANDOFF_ARM_JOINTS = (
    -1.201772928,
    -2.648592234,
    2.662814856,
    3.293901920,
    -0.367445409,
    -1.734327674,
)
RECORDED_HANDOFF_BLADE_POSE = (
    0.13234924,
    -0.22563651,
    0.71610749,
    0.99999088,
    -0.00119576,
    0.00210663,
    -0.00349365,
)


@configclass
class ZeroGBladeGrapplePinInsertHandoffEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg):
    """Train only from a measured insertion state the predecessor supplied."""

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        reset = getattr(self.events, "reset_stroke", None)
        if reset is None:
            raise ValueError("handoff-conditioned insertion requires the solved stroke reset")
        reset.params["arm_poses_by_bay"] = ((RECORDED_HANDOFF_ARM_JOINTS,),)
        reset.params["blade_poses_by_bay"] = ((RECORDED_HANDOFF_BLADE_POSE,),)
        reset.params["forced_station"] = HANDOFF_RESET_STATION

        root = list(GRAPPLE_ROBOT_ROOT_POS)
        root[1] = RECORDED_HANDOFF_ROBOT_ROOT_Y_M
        self.scene.robot.init_state.pos = tuple(root)
        mount = getattr(self.scene, "mount_anchor", None)
        if mount is not None:
            anchor = list(mount.init_state.pos)
            anchor[1] = RECORDED_HANDOFF_ROBOT_ROOT_Y_M
            mount.init_state.pos = tuple(anchor)


__all__ = [
    "HANDOFF_RESET_STATION",
    "RECORDED_HANDOFF_ARM_JOINTS",
    "RECORDED_HANDOFF_BLADE_POSE",
    "RECORDED_HANDOFF_ROBOT_ROOT_Y_M",
    "RECORDED_HANDOFF_SOURCE_COMMIT",
    "RECORDED_HANDOFF_TRACE_SHA256",
    "ZeroGBladeGrapplePinInsertHandoffEnvCfg",
]
