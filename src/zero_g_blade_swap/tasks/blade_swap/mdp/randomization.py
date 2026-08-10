"""Physics, task-state, and Replicator visual randomization terms."""

from __future__ import annotations

import math

import numpy as np
import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils import get_current_stage


class MountWobblePulse(ManagerTermBase):
    """Generate bounded 0.2--0.8 s satellite-mount wrench pulses."""

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._until_next = torch.zeros_like(self._remaining)
        self._forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self._torques = torch.zeros_like(self._forces)
        self._step_dt = float(env.step_dt)
        self._quiet_range_s = tuple(cfg.params["quiet_range_s"])

    def reset(self, env_ids=None) -> None:
        ids = torch.arange(len(self._remaining), device=self._remaining.device) if env_ids is None else env_ids
        self._remaining[ids] = 0
        quiet_min = max(1, math.ceil(self._quiet_range_s[0] / self._step_dt))
        quiet_max = max(quiet_min + 1, math.ceil(self._quiet_range_s[1] / self._step_dt) + 1)
        self._until_next[ids] = torch.randint(quiet_min, quiet_max, (len(ids),), device=self._remaining.device)
        self._forces[ids] = 0.0
        self._torques[ids] = 0.0

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        duration_range_s: tuple[float, float],
        quiet_range_s: tuple[float, float],
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        self._remaining[ids] -= 1
        self._until_next[ids] -= 1
        finished = ids[self._remaining[ids] <= 0]
        self._forces[finished] = 0.0
        self._torques[finished] = 0.0
        due = ids[(self._remaining[ids] <= 0) & (self._until_next[ids] <= 0)]
        if len(due) > 0:
            minimum = max(1, math.ceil(duration_range_s[0] / float(env.step_dt)))
            maximum = max(minimum + 1, math.ceil(duration_range_s[1] / float(env.step_dt)) + 1)
            self._remaining[due] = torch.randint(minimum, maximum, (len(due),), device=env.device)
            quiet_min = max(1, math.ceil(quiet_range_s[0] / float(env.step_dt)))
            quiet_max = max(quiet_min + 1, math.ceil(quiet_range_s[1] / float(env.step_dt)) + 1)
            self._until_next[due] = self._remaining[due] + torch.randint(
                quiet_min, quiet_max, (len(due),), device=env.device
            )
            self._forces[due, 0] = torch.empty((len(due), 3), device=env.device).uniform_(*force_range)
            self._torques[due, 0] = torch.empty((len(due), 3), device=env.device).uniform_(*torque_range)

        robot = env.scene[asset_cfg.name]
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces[ids],
            torques=self._torques[ids],
            body_ids=asset_cfg.body_ids,
            env_ids=ids,
            is_global=True,
        )


class RackMaterialRandomizer(ManagerTermBase):
    """Create per-environment OmniPBR rack materials through Replicator."""

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        if env.cfg.scene.replicate_physics:
            raise RuntimeError("RackMaterialRandomizer requires replicate_physics=False.")
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep

        self._rep = rep
        self._rng = rep.rng.ReplicatorRNG()
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        mesh_name = cfg.params.get("mesh_name", "geometry/mesh").lstrip("/")
        mesh_path = f"{env.scene[asset_cfg.name].cfg.prim_path}/{mesh_name}"
        meshes = rep.functional.get.prims(stage=get_current_stage(), path_pattern=mesh_path)
        if len(meshes) != env.num_envs:
            raise RuntimeError(f"Expected {env.num_envs} rack meshes at '{mesh_path}', found {len(meshes)}.")
        for prim in meshes:
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        self._materials = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", bind_prims=meshes, count=len(meshes), project_uvw=True
        )

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mesh_name: str,
        steel_color_range: tuple[tuple[float, float, float], tuple[float, float, float]],
        gold_color_range: tuple[tuple[float, float, float], tuple[float, float, float]],
        metallic_range: tuple[float, float],
        roughness_range: tuple[float, float],
        gold_probability: float = 0.5,
    ) -> None:
        del env, env_ids, asset_cfg, mesh_name
        count = len(self._materials)
        generator = self._rng.generator
        steel = generator.uniform(steel_color_range[0], steel_color_range[1], size=(count, 3))
        gold = generator.uniform(gold_color_range[0], gold_color_range[1], size=(count, 3))
        use_gold = generator.random(count) < gold_probability
        colors = np.where(use_gold[:, None], gold, steel)
        metallic = generator.uniform(*metallic_range, size=count)
        roughness = generator.uniform(*roughness_range, size=count)
        self._rep.functional.modify.attribute(self._materials, "diffuse_color_constant", colors)
        self._rep.functional.modify.attribute(self._materials, "metallic_constant", metallic)
        self._rep.functional.modify.attribute(self._materials, "reflection_roughness_constant", roughness)


def randomize_rail_stiction(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    breakaway_force_range: tuple[float, float],
    viscous_coefficient_range: tuple[float, float],
) -> None:
    """Sample smooth-Coulomb breakaway and viscous rail resistance."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    params = getattr(env, "_rail_stiction", None)
    if params is None:
        params = {}
        env._rail_stiction = params
    if asset_cfg.name not in params:
        params[asset_cfg.name] = (
            torch.zeros(env.num_envs, device=env.device),
            torch.zeros(env.num_envs, device=env.device),
        )
    breakaway, viscous = params[asset_cfg.name]
    breakaway[ids] = torch.empty(len(ids), device=env.device).uniform_(*breakaway_force_range)
    viscous[ids] = torch.empty(len(ids), device=env.device).uniform_(*viscous_coefficient_range)


def rail_stiction_resistance(
    env,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
    rail_center_y: float,
    rail_center_z: float,
    engagement_x_range: tuple[float, float],
    velocity_smoothing: float = 0.01,
) -> torch.Tensor:
    """Return the axial rail force without writing a PhysX wrench."""

    asset = env.scene[asset_cfg.name]
    params = getattr(env, "_rail_stiction", {}).get(asset_cfg.name)
    if params is None:
        return torch.zeros(len(env_ids), device=env.device)
    breakaway, viscous = params
    local_position = asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    velocity_x = asset.data.root_lin_vel_w[env_ids, 0]
    engaged = (
        (local_position[:, 0] >= engagement_x_range[0])
        & (local_position[:, 0] <= engagement_x_range[1])
        & ((local_position[:, 1] - rail_center_y).abs() < 0.10)
        & ((local_position[:, 2] - rail_center_z).abs() < 0.07)
    )
    resistance = -breakaway[env_ids] * torch.tanh(velocity_x / velocity_smoothing) - viscous[env_ids] * velocity_x
    return torch.where(engaged, resistance.clamp(-160.0, 160.0), torch.zeros_like(resistance))


def apply_rail_stiction(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    rail_center_y: float,
    rail_center_z: float,
    engagement_x_range: tuple[float, float],
    velocity_smoothing: float = 0.01,
) -> None:
    """Apply randomized axial resistance while a blade is inside its guides."""

    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if getattr(env, "_rail_stiction", {}).get(asset_cfg.name) is None:
        raise RuntimeError(f"Rail stiction for '{asset_cfg.name}' was not sampled at reset.")
    resistance = rail_stiction_resistance(
        env,
        asset_cfg,
        ids,
        rail_center_y,
        rail_center_z,
        engagement_x_range,
        velocity_smoothing,
    )
    forces = torch.zeros((len(ids), 1, 3), device=env.device)
    torques = torch.zeros_like(forces)
    forces[:, 0, 0] = resistance
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=ids,
        is_global=True,
    )


class OrbitalLightingRandomizer(ManagerTermBase):
    """Randomize a single directional sun through Replicator functional APIs."""

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        if env.cfg.scene.replicate_physics:
            raise RuntimeError("OrbitalLightingRandomizer requires replicate_physics=False.")

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        import carb
        import omni.replicator.core as rep

        settings = carb.settings.get_settings()
        settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
        settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", False)
        settings.set_bool("/rtx/post/backgroundZeroAlpha/outputAlphaInComposite", True)
        settings.set_bool("/rtx/post/backgroundZeroAlpha/blackBackgroundInComposite", True)

        self._rep = rep
        self._rng = rep.rng.ReplicatorRNG()
        self._prims = rep.functional.get.prims(stage=get_current_stage(), path_pattern=cfg.params["prim_path"])
        if len(self._prims) == 0:
            created = rep.functional.create.distant_light(
                name="OrbitalSun",
                parent="/World",
                intensity=5_000.0,
                angle=0.25,
                color_temperature=6_500.0,
                enable_color_temperature=True,
            )
            self._prims = [created]
        if len(self._prims) != 1 or self._prims[0].GetTypeName() != "DistantLight":
            raise RuntimeError(
                f"Expected one Replicator-addressable DistantLight at '{cfg.params['prim_path']}', "
                f"found {[prim.GetPath() for prim in self._prims]}."
            )

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        prim_path: str,
        intensity_range: tuple[float, float],
        angle_range: tuple[float, float],
        pitch_range_deg: tuple[float, float],
        yaw_range_deg: tuple[float, float],
        color_temperature_range: tuple[float, float],
    ) -> None:
        del env, env_ids, prim_path
        intensity = float(self._rng.generator.uniform(*intensity_range))
        angle = float(self._rng.generator.uniform(*angle_range))
        pitch = float(self._rng.generator.uniform(*pitch_range_deg))
        yaw = float(self._rng.generator.uniform(*yaw_range_deg))
        color_temperature = float(self._rng.generator.uniform(*color_temperature_range))
        self._rep.functional.modify.attribute(self._prims, "inputs:intensity", intensity)
        self._rep.functional.modify.attribute(self._prims, "inputs:angle", angle)
        self._rep.functional.modify.attribute(self._prims, "inputs:enableColorTemperature", True)
        self._rep.functional.modify.attribute(self._prims, "inputs:colorTemperature", color_temperature)
        self._rep.functional.modify.pose(
            self._prims,
            rotation_value=(pitch, 0.0, yaw),
            rotation_order="XYZ",
        )


__all__ = [
    "MountWobblePulse",
    "OrbitalLightingRandomizer",
    "RackMaterialRandomizer",
    "apply_rail_stiction",
    "rail_stiction_resistance",
    "randomize_rail_stiction",
]
