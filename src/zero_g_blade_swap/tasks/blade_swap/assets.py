"""Lightweight procedural assets and the compliant robot mount.

The rack and blade deliberately use primitive geometry.  This keeps the physics
state small enough for the 1024-environment teacher run and makes this module
usable before production CAD/USD assets are available.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

ROBOT_ROOT_POS = (-0.45, 0.0, 0.15)
BLADE_INSERTED_POS = (0.75, 0.0, 0.72)
SPARE_BLADE_POS = (0.42, -0.42, 0.50)
SERVICE_CADDY_POS = (0.42, 0.42, 0.50)
BLADE_SIZE = (0.45, 0.16, 0.035)


def _relocate_ur10e_articulation_root(stage: Usd.Stage, environment_path: str) -> None:
    """Move the stock fixed-base articulation schema onto ``base_link``.

    The canonical UR10e USD authors ``ArticulationRootAPI`` on its global
    ``root_joint``.  Setting ``fix_root_link=False`` disables that joint, but
    Isaac Lab 2.3.2 does not relocate the API.  PhysX consequently has no valid
    floating articulation.  A floating articulation must instead be rooted on
    its first rigid body before the external compliant D6 joint is authored.
    """

    source_path = f"{environment_path}/Robot/root_joint"
    target_path = f"{environment_path}/Robot/base_link"
    source_prim = stage.GetPrimAtPath(source_path)
    target_prim = stage.GetPrimAtPath(target_path)

    if not source_prim.IsValid():
        raise RuntimeError(f"UR10e articulation-root prim is missing at '{source_path}'.")
    if not target_prim.IsValid() or not target_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"UR10e floating articulation requires a rigid body at '{target_path}'.")

    # Be idempotent for stage reloads while still rejecting an unexpected USD
    # hierarchy instead of letting PhysX fail later with mimic-joint errors.
    if target_prim.HasAPI(UsdPhysics.ArticulationRootAPI) and not source_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return
    if not source_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(f"Expected ArticulationRootAPI on the stock UR10e joint '{source_path}'.")

    source_usd_api = UsdPhysics.ArticulationRootAPI(source_prim)
    UsdPhysics.ArticulationRootAPI.Apply(target_prim)
    for attribute_name in source_usd_api.GetSchemaAttributeNames():
        value = source_prim.GetAttribute(attribute_name).Get()
        if value is not None:
            target_prim.GetAttribute(attribute_name).Set(value)

    source_physx_api = PhysxSchema.PhysxArticulationAPI(source_prim)
    if source_physx_api:
        PhysxSchema.PhysxArticulationAPI.Apply(target_prim)
        for attribute_name in source_physx_api.GetSchemaAttributeNames():
            value = source_prim.GetAttribute(attribute_name).Get()
            if value is not None:
                target_prim.GetAttribute(attribute_name).Set(value)

    source_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if source_physx_api:
        source_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)


def _define_compliant_d6(
    stage: Usd.Stage,
    container_path: str,
    cfg: CompliantD6JointCfg,
) -> Usd.Prim:
    """Author one driven six-axis USD Physics joint and return its container.

    A container Xform is used because ``InteractiveScene`` tracks generic
    ``AssetBaseCfg`` entries with an Xform view.  The actual D6 joint is the
    ``Joint`` child.
    """

    environment_path = container_path.rsplit("/", 1)[0]
    body0_path = f"{environment_path}/{cfg.body0_relative_path}"
    body1_path = f"{environment_path}/{cfg.body1_relative_path}"
    for label, body_path in (("body0", body0_path), ("body1", body1_path)):
        prim = stage.GetPrimAtPath(body_path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "Cannot create the compliant UR10e mount: expected "
                f"{label} rigid body at '{body_path}', but it was not found. "
                "The UR10e USD hierarchy may have changed; inspect the robot "
                "root-link path instead of silently falling back to pose noise."
            )

    _relocate_ur10e_articulation_root(stage, environment_path)

    container = UsdGeom.Xform.Define(stage, container_path)
    if not sim_utils.standardize_xform_ops(
        container.GetPrim(),
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    ):
        raise RuntimeError(f"Failed to author canonical Xform ops on '{container_path}'.")
    joint = UsdPhysics.Joint.Define(stage, f"{container_path}/Joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    # Keep this external spring constraint outside the robot's reduced-
    # coordinate articulation.  It still acts on the articulation root body.
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(identity)
    joint.CreateLocalRot1Attr().Set(identity)

    joint_prim = joint.GetPrim()
    axes = (
        ("transX", -cfg.translation_limit, cfg.translation_limit, cfg.translation_stiffness, cfg.translation_damping),
        ("transY", -cfg.translation_limit, cfg.translation_limit, cfg.translation_stiffness, cfg.translation_damping),
        ("transZ", -cfg.translation_limit, cfg.translation_limit, cfg.translation_stiffness, cfg.translation_damping),
        ("rotX", -cfg.rotation_limit_deg, cfg.rotation_limit_deg, cfg.rotation_stiffness, cfg.rotation_damping),
        ("rotY", -cfg.rotation_limit_deg, cfg.rotation_limit_deg, cfg.rotation_stiffness, cfg.rotation_damping),
        ("rotZ", -cfg.rotation_limit_deg, cfg.rotation_limit_deg, cfg.rotation_stiffness, cfg.rotation_damping),
    )
    for axis, lower, upper, stiffness, damping in axes:
        limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
        limit.CreateLowAttr(lower)
        limit.CreateHighAttr(upper)

        drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateTargetPositionAttr(0.0)
        if axis.startswith("rot"):
            # USD angular drive positions and velocities are expressed in
            # degrees, while the configured gains are intuitive SI/radian.
            stiffness *= math.pi / 180.0
            damping *= math.pi / 180.0
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(cfg.max_force)

    return container.GetPrim()


def spawn_compliant_d6_joint(
    prim_path: str,
    cfg: CompliantD6JointCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
) -> Usd.Prim:
    """Spawn a separate D6 joint for every matched environment.

    Isaac Lab's normal ``@clone`` helper clones a single joint prim.  Absolute
    USD relationship targets may then still point at the source environment.
    This spawner instead resolves every environment parent and authors each
    relationship explicitly, which is safer for a joint whose bodies live
    outside the cloned joint subtree.
    """

    if translation not in (None, (0.0, 0.0, 0.0)) or orientation not in (
        None,
        (1.0, 0.0, 0.0, 0.0),
    ):
        raise ValueError("CompliantD6JointCfg must be spawned at the identity pose.")

    root_path, leaf = str(prim_path).rsplit("/", 1)
    is_regex = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None
    parent_paths = find_matching_prim_paths(root_path) if is_regex else [root_path]
    if not parent_paths:
        raise RuntimeError(f"No environment parents matched compliant-joint path '{prim_path}'.")

    stage = get_current_stage()
    source_prim: Usd.Prim | None = None
    for parent_path in parent_paths:
        prim = _define_compliant_d6(stage, f"{parent_path}/{leaf}", cfg)
        source_prim = source_prim or prim
    assert source_prim is not None
    return source_prim


@configclass
class CompliantD6JointCfg(SpawnerCfg):
    """Parameters for the spring-damped satellite mounting interface."""

    func: Callable[..., Usd.Prim] = spawn_compliant_d6_joint
    body0_relative_path: str = "MountAnchor"
    body1_relative_path: str = "Robot/base_link"
    translation_limit: float = 0.015
    rotation_limit_deg: float = 2.0
    translation_stiffness: float = 12_000.0
    translation_damping: float = 220.0
    rotation_stiffness: float = 600.0
    rotation_damping: float = 50.0
    max_force: float = 20_000.0


def make_robot_cfg() -> ArticulationCfg:
    """Return the canonical UR10e/Robotiq asset with a compliant root."""

    cfg = UR10e_ROBOTIQ_2F_85_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_ROOT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": math.pi,
                "shoulder_lift_joint": -math.pi / 2.0,
                "elbow_joint": math.pi / 2.0,
                "wrist_1_joint": -math.pi / 2.0,
                "wrist_2_joint": -math.pi / 2.0,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
                ".*_inner_finger_joint": 0.0,
                ".*_inner_finger_knuckle_joint": 0.0,
                ".*_outer_.*_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )
    # The stock robot is fixed to the world.  A free root plus the D6 spring is
    # what creates real mount deflection; reset pose noise is not compliance.
    cfg.spawn.articulation_props.fix_root_link = False
    cfg.spawn.rigid_props.disable_gravity = True
    cfg.spawn.activate_contact_sensors = False
    cfg.articulation_root_prim_path = "/base_link"
    return cfg


def _rigid_props(*, kinematic: bool) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=kinematic,
        disable_gravity=True,
        linear_damping=0.08 if not kinematic else 0.0,
        angular_damping=0.08 if not kinematic else 0.0,
        max_depenetration_velocity=2.0,
        enable_gyroscopic_forces=True,
    )


def spawn_blade_with_handle(
    prim_path: str,
    cfg: BladeCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn a low-cost blade chassis plus an attached Robotiq handle collider."""

    root = sim_utils.spawn_cuboid(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = get_current_stage()
    for root_path in find_matching_prim_paths(prim_path):
        handle_path = f"{root_path}/Handle"
        if stage.GetPrimAtPath(handle_path).IsValid():
            continue
        handle = UsdGeom.Cube.Define(stage, handle_path)
        handle.CreateSizeAttr(1.0)
        sim_utils.standardize_xform_ops(
            handle.GetPrim(),
            translation=cfg.handle_offset,
            orientation=(1.0, 0.0, 0.0, 0.0),
            scale=cfg.handle_size,
        )
        sim_utils.define_collision_properties(handle_path, cfg.collision_props, stage=stage)
        material_path = f"{root_path}/geometry/{cfg.visual_material_path}"
        if stage.GetPrimAtPath(material_path).IsValid():
            sim_utils.bind_visual_material(handle_path, material_path, stage=stage)
    return root


@configclass
class BladeCuboidCfg(sim_utils.CuboidCfg):
    """Cuboid chassis with one collision handle sharing the root rigid body."""

    func: Callable[..., Usd.Prim] = spawn_blade_with_handle
    handle_offset: tuple[float, float, float] = (-0.235, 0.0, 0.0)
    handle_size: tuple[float, float, float] = (0.020, 0.075, 0.025)


def _blade_cfg(
    name: str,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    semantic_label: str,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=BladeCuboidCfg(
            size=BLADE_SIZE,
            rigid_props=_rigid_props(kinematic=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.55,
                dynamic_friction=0.45,
                restitution=0.0,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.75, roughness=0.25),
            semantic_tags=[("class", semantic_label)],
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


BLADE_CFG = _blade_cfg("Blade", BLADE_INSERTED_POS, (0.28, 0.05, 0.04), "failed_server_blade")
SPARE_BLADE_CFG = _blade_cfg("SpareBlade", SPARE_BLADE_POS, (0.04, 0.18, 0.30), "spare_server_blade")


SLOT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BladeSlot",
    spawn=sim_utils.CuboidCfg(
        size=(0.60, 0.20, 0.014),
        rigid_props=_rigid_props(kinematic=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.65,
            restitution=0.0,
            friction_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.22, 0.24), metallic=0.9, roughness=0.18),
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(BLADE_INSERTED_POS[0], 0.0, 0.6955)),
)


def _slot_guide_cfg(
    name: str,
    position: tuple[float, float, float],
    length: float = 0.60,
) -> RigidObjectCfg:
    """Create a side guide with a small interference fit for axial friction.

    In zero gravity a shelf alone produces almost no normal force and therefore
    almost no Coulomb friction.  The two guides preload the blade by roughly
    0.25 mm per side, making randomized sliding friction physically meaningful
    as a thermal-welding/stiction proxy while retaining contact-based motion.
    """

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(length, 0.018, 0.050),
            rigid_props=_rigid_props(kinematic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.65,
                restitution=0.0,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.22), metallic=0.9, roughness=0.18),
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


SLOT_LEFT_GUIDE_CFG = _slot_guide_cfg("BladeSlotLeftGuide", (BLADE_INSERTED_POS[0], 0.08875, BLADE_INSERTED_POS[2]))
SLOT_RIGHT_GUIDE_CFG = _slot_guide_cfg("BladeSlotRightGuide", (BLADE_INSERTED_POS[0], -0.08875, BLADE_INSERTED_POS[2]))


def _caddy_shelf_cfg(name: str, center: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.20, 0.014),
            rigid_props=_rigid_props(kinematic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65,
                dynamic_friction=0.50,
                restitution=0.0,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.32, 0.24, 0.08), metallic=0.85, roughness=0.28
            ),
            semantic_tags=[("class", "service_caddy")],
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(center[0], center[1], center[2] - 0.0245)),
    )


SUPPLY_CADDY_CFG = _caddy_shelf_cfg("SupplyCaddy", SPARE_BLADE_POS)
SERVICE_CADDY_CFG = _caddy_shelf_cfg("ServiceCaddy", SERVICE_CADDY_POS)
SUPPLY_CADDY_LEFT_GUIDE_CFG = _slot_guide_cfg(
    "SupplyCaddyLeftGuide", (SPARE_BLADE_POS[0], SPARE_BLADE_POS[1] + 0.08875, SPARE_BLADE_POS[2]), 0.55
)
SUPPLY_CADDY_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "SupplyCaddyRightGuide", (SPARE_BLADE_POS[0], SPARE_BLADE_POS[1] - 0.08875, SPARE_BLADE_POS[2]), 0.55
)
SERVICE_CADDY_LEFT_GUIDE_CFG = _slot_guide_cfg(
    "ServiceCaddyLeftGuide",
    (SERVICE_CADDY_POS[0], SERVICE_CADDY_POS[1] + 0.08875, SERVICE_CADDY_POS[2]),
    0.55,
)
SERVICE_CADDY_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "ServiceCaddyRightGuide",
    (SERVICE_CADDY_POS[0], SERVICE_CADDY_POS[1] - 0.08875, SERVICE_CADDY_POS[2]),
    0.55,
)


RACK_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Rack",
    spawn=sim_utils.CuboidCfg(
        size=(0.035, 0.72, 1.15),
        rigid_props=_rigid_props(kinematic=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.55,
            dynamic_friction=0.45,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.18), metallic=0.95, roughness=0.12),
        semantic_tags=[("class", "server_rack")],
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.005, 0.0, 0.76)),
)


MOUNT_ANCHOR_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/MountAnchor",
    spawn=sim_utils.CuboidCfg(
        size=(0.12, 0.12, 0.06),
        rigid_props=_rigid_props(kinematic=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.04, 0.04, 0.05), metallic=0.8, roughness=0.35),
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=ROBOT_ROOT_POS),
)


__all__ = [
    "BLADE_CFG",
    "BLADE_INSERTED_POS",
    "BLADE_SIZE",
    "CompliantD6JointCfg",
    "MOUNT_ANCHOR_CFG",
    "RACK_CFG",
    "ROBOT_ROOT_POS",
    "SERVICE_CADDY_CFG",
    "SERVICE_CADDY_LEFT_GUIDE_CFG",
    "SERVICE_CADDY_POS",
    "SERVICE_CADDY_RIGHT_GUIDE_CFG",
    "SLOT_CFG",
    "SLOT_LEFT_GUIDE_CFG",
    "SLOT_RIGHT_GUIDE_CFG",
    "SPARE_BLADE_CFG",
    "SPARE_BLADE_POS",
    "SUPPLY_CADDY_CFG",
    "SUPPLY_CADDY_LEFT_GUIDE_CFG",
    "SUPPLY_CADDY_RIGHT_GUIDE_CFG",
    "make_robot_cfg",
    "spawn_compliant_d6_joint",
]
