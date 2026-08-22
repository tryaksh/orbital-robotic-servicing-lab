"""A slot shaped like a real card guide, and a grasp made inside it.

Two tasks live here, and they are the two halves of a blade replacement.

**Guided-slot insertion.** The certified slot is two vertical walls, so nothing
opposes blade pitch or roll and the policy must hold the blade level by
commanding orientation alone; terminal orientation error consumes 97.8% of its
tolerance as a result. This profile closes the channel with overhanging upper
lips and flares the mouth, which is what a 1U card guide actually looks like.
The mechanical constraint should take over the job the reward was doing.

**Capture-in-slot grasping.** Squeezing a free-floating mass in zero gravity
ejects it: closing two pads on an unconstrained 10 kg blade turns any asymmetry
in contact timing into net momentum, with no gravity or fixture to absorb it,
and the measured result was 180 mm of ejection under no load at all. A real
servicer never does this. It grasps a module that is still captured by its
guides, and only then breaks it free. This profile therefore closes the fingers
while the blade is still inside the channel, so the rails and lips absorb the
closing asymmetry.

Neither task touches the geometry the promoted Level-0/1/2 policies were
certified on; both are separate scenes and separate registrations.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .rigid_grasp_insertion_env_cfg import ZeroGBladeRigidGraspInsertionEnvCfg
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .scene_cfg import ZeroGGuidedSlotSceneCfg

# The lips and flares are the point of this profile, so they stay collidable at
# every robustness level, including level 0 where the side walls are still off.
GUIDED_SLOT_ALWAYS_COLLIDABLE = ("blade_slot_upper_left_lip", "blade_slot_upper_right_lip")
GUIDED_SLOT_ENTRY_FLARES = ("blade_slot_entry_left_flare", "blade_slot_entry_right_flare")


def _enable_channel(scene) -> None:
    """Keep the channel and its lead-in solid whatever the profile does."""

    for name in (*GUIDED_SLOT_ALWAYS_COLLIDABLE, *GUIDED_SLOT_ENTRY_FLARES):
        getattr(scene, name).spawn.collision_props.collision_enabled = True


@configclass
class ZeroGBladeGuidedSlotEnvCfg(ZeroGBladeRigidGraspInsertionEnvCfg):
    """Rigid-grasp insertion into a four-sided, funnel-mouthed channel."""

    scene: ZeroGGuidedSlotSceneCfg = ZeroGGuidedSlotSceneCfg(
        num_envs=512,
        env_spacing=2.6,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # The parent rebuilds the side walls and floor for the chosen level and
        # knows nothing about the channel, so re-assert it afterwards.
        _enable_channel(self.scene)


@configclass
class ZeroGBladeGuidedSlotPlayEnvCfg(ZeroGBladeGuidedSlotEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


__all__ = [
    "GUIDED_SLOT_ALWAYS_COLLIDABLE",
    "GUIDED_SLOT_ENTRY_FLARES",
    "ZeroGBladeGuidedSlotEnvCfg",
    "ZeroGBladeGuidedSlotPlayEnvCfg",
]
