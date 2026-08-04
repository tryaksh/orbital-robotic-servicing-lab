"""Interactive scene configurations for the zero-g blade swap task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .assets import (
    BLADE_CFG,
    MOUNT_ANCHOR_CFG,
    RACK_CFG,
    SERVICE_CADDY_CFG,
    SERVICE_CADDY_LEFT_GUIDE_CFG,
    SERVICE_CADDY_RIGHT_GUIDE_CFG,
    SLOT_CFG,
    SLOT_LEFT_GUIDE_CFG,
    SLOT_RIGHT_GUIDE_CFG,
    SPARE_BLADE_CFG,
    SUPPLY_CADDY_CFG,
    SUPPLY_CADDY_LEFT_GUIDE_CFG,
    SUPPLY_CADDY_RIGHT_GUIDE_CFG,
    CompliantD6JointCfg,
    make_robot_cfg,
)


@configclass
class ZeroGBladeSwapSceneCfg(InteractiveSceneCfg):
    """Minimal physics scene shared by teacher, vision, and play variants.

    Asset declaration order is intentional: both rigid bodies used by the D6
    mount joint must exist before the joint spawner validates and connects them.
    There is no ground plane and no dome light, so the renderer's background is
    black and only the orbital sun illuminates the task.
    """

    mount_anchor = MOUNT_ANCHOR_CFG
    robot = make_robot_cfg()
    base_compliance = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BaseCompliance",
        spawn=CompliantD6JointCfg(),
    )

    blade = BLADE_CFG
    spare_blade = SPARE_BLADE_CFG
    blade_slot = SLOT_CFG
    blade_slot_left_guide = SLOT_LEFT_GUIDE_CFG
    blade_slot_right_guide = SLOT_RIGHT_GUIDE_CFG
    supply_caddy = SUPPLY_CADDY_CFG
    supply_caddy_left_guide = SUPPLY_CADDY_LEFT_GUIDE_CFG
    supply_caddy_right_guide = SUPPLY_CADDY_RIGHT_GUIDE_CFG
    service_caddy = SERVICE_CADDY_CFG
    service_caddy_left_guide = SERVICE_CADDY_LEFT_GUIDE_CFG
    service_caddy_right_guide = SERVICE_CADDY_RIGHT_GUIDE_CFG
    rack = RACK_CFG

    # Kept absent in the teacher scene to avoid allocating render products.
    camera: TiledCameraCfg | None = None


def make_tiled_camera_cfg() -> TiledCameraCfg:
    """Create the 64x64, 15 Hz camera used by vision and data collection."""

    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1.0 / 15.0,
        height=64,
        width=64,
        data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.55, -0.65, 1.15),
            rot=(0.9610, -0.0346, 0.1438, 0.2317),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            horizontal_aperture=22.0,
            focus_distance=1.4,
            f_stop=0.0,
            clipping_range=(0.05, 4.0),
        ),
    )


__all__ = ["ZeroGBladeSwapSceneCfg", "make_tiled_camera_cfg"]
