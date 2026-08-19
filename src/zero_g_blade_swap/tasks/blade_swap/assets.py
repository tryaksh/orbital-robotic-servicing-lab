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

from zero_g_blade_swap.grapple_geometry import (
    CLOSING_RATE_M_PER_RAD,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_PIN_COLLAR_HALF_HEIGHT,
    GRAPPLE_PIN_COLLAR_X,
    GRAPPLE_PIN_GRIP_OFFSET,
    GRAPPLE_PIN_HALF_WIDTH_Y,
    GRAPPLE_PIN_SHAFT_HALF_HEIGHT,
    GRAPPLE_PIN_SHAFT_X,
    GRAPPLE_PIN_WEDGE_HALF_HEIGHT,
    GRAPPLE_PIN_WEDGE_X,
    GRAPPLE_TOOL_OFFSET_POS,
    RATED_GRIP_FORCE_N,
    drive_torque_for_grip_force_nm,
    wedge_taper_deg,
)

ROBOT_ROOT_POS = (-0.45, 0.0, 0.15)
BLADE_INSERTED_POS = (0.75, 0.0, 0.72)
SPARE_BLADE_POS = (0.70, -0.42, 0.50)
SERVICE_CADDY_POS = (0.70, 0.42, 0.50)
BLADE_SIZE = (0.45, 0.16, 0.035)
BLADE_HANDLE_OFFSET = (-0.250, 0.0, 0.0)
TRANSFER_BLADE_X = 0.18
GUIDE_CENTER_OFFSET_Y = 0.08975  # 1.5 mm total clearance around the 160 mm blade.
TOOL_OFFSET_POS = (0.0, 0.0, 0.19)
TOOL_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)
CONTACT_TOOL_OFFSET_POS = (0.0, 0.0, 0.179)
# Measured, not assumed. Sweeping the handle along the tool axis while the
# fingers close on it saturates the 10 N-m drive limit only while the handle
# centre is roughly 60 to 150 mm below the wrist flange; at the 179 mm above the
# fingers close past the handle and PhysX reports no contact at all. 105 mm is
# the middle of that band, leaving about 45 mm of margin on each side.
# See evidence/grasp_axial_pull_gate.json and scripts/grasp_diagnostics.py.
GRASP_TOOL_OFFSET_POS = (0.0, 0.0, 0.105)
CONTACT_BLADE_HANDLE_OFFSET = (-0.250, 0.0, 0.0185)
# A 30 mm tab lying on the blade's 160 mm-wide deck cannot be gripped by this
# gripper: the pads sit 82 mm apart at their approach opening, so they foul the
# deck long before they can straddle a tab that short. The grasp task therefore
# uses a raised grapple post, the interface an orbital-servicing module would
# actually carry. Its centre sits at the pad height for the unchanged arm reset
# pose, so no joint angles have to be re-derived:
#   flange z 0.9158 - tool offset 0.105 = pads at z 0.8108
#   blade centre z 0.72, so the post centre is 0.0908 above it.
# At 150 mm tall it spans z 0.7358 to 0.8858: rooted 1.7 mm inside the deck at
# 0.7375, with the pads gripping 75 mm clear of the deck. At 75 mm wide it stays
# inside the +/-80.75 mm rail faces and clears the narrow upper lips.
GRAPPLE_POST_OFFSET = (-0.250, 0.0, 0.0908)
GRAPPLE_POST_SIZE = (0.060, 0.075, 0.150)
GRIPPER_GRASP_ROT = (0.0, 0.7071068, 0.7071068, 0.0)

# ---------------------------------------------------------------------------
# Head-on grapple pin.
#
# A parallel-jaw grip cannot hold this blade against extraction, and that is
# structural rather than a tuning failure: the gripper closes along one axis and
# the blade leaves along another, so only friction opposes the pull. Measured
# axial capacity that way is about 6 N against the 66.4 N the insertion contact
# reaction demands. The fix has to put geometry on the pull axis, which means
# approaching along it, so the interface moves onto the module. That is why the
# ISS ORU standard carries a grapple fixture rather than demanding a cleverer
# end effector.
#
# Every number below is derived from evidence/gripper_collision_envelope.json,
# which measures the 2F-85 from its collision meshes in the wrist frame:
#
#   pads close along wrist x, approach along wrist z
#   clear opening 87.08 mm at finger_joint 0, closing at 106.2 mm/rad to 0 mm
#     at the 0.8203 rad limit, so *zero is fully open* and larger values close
#   pads span 105 to 162 mm from the flange, 57 mm of usable length
#   palm face at 90 mm, and the inner knuckles fill the 90-105 mm throat to
#     within 8 mm of the tool axis at grip closures
#
# The throat measurement rules out the obvious design. A mushroom head with a
# flat shoulder has to seat between the palm and the pad trailing faces to bear
# on them, that gap is 15 mm deep, and the knuckles swing through it. At every
# closure the head would have to be wider than the pad gap to catch the pads and
# narrower than the knuckles to fit, and no closure satisfies both.
#
# What does work is a wedge clamped inside the pad aperture, which is the
# tapered-pin principle Canadarm2 captures on. The pin is thicker at its free
# end, so pulling the blade out drags thicker material into the pads and forces
# them apart against the drive, and the reaction on the pin acts along the pull
# axis. Capacity is 2 N sin(alpha) from geometry alone before any friction:
# at the 10 N-m drive limit that is 77 N against the 66.4 N gate, so the
# interface clears it without relying on a friction coefficient at all.
# Dimensions live in zero_g_blade_swap.grapple_geometry, which imports nothing
# from Isaac Lab so the test suite can defend them without a simulator. The
# reasoning behind each one is recorded there.
#
# In short: the blade's front face is at -0.225 and the rack mouth sits 75 mm in
# front of it at full insertion, which is what sets the 80 mm shaft, because the
# pads are 57 mm long and have to stay outside the mouth. The collar is taller
# than the pads can open, so it is a depth stop rather than something a
# wide-open gripper slides past, and it gives the insert direction a face to
# push on. The wedge is 70 mm across at its free end, inside the 87.08 mm
# aperture with 8.5 mm of approach clearance a side.
GRAPPLE_PIN_TAPER_DEG = wedge_taper_deg()

# Solved by scripts/calibrate_grasp_pose.py against this task's own differential
# IK, one environment per curriculum stage, fingers held open at the approach
# command and the blade pinned at its stage pose. Every stage converged to under
# 0.01 mm and 0.00003 rad. See evidence/grapple_pin_head_on_pose.json.
GRAPPLE_HEAD_ON_ARM_JOINT_POS = (
    (-0.304997, -1.421518, 2.050886, 2.512231, -1.265814, -1.570812),
    (-0.341348, -1.517620, 2.159929, 2.499296, -1.229464, -1.570783),
    (-0.403679, -1.653908, 2.293487, 2.502023, -1.167132, -1.570785),
)

# Robotiq specifies 20 to 235 N of grip force for the 2F-85. The drive limit
# this project inherited, 10 N-m, delivers about 100 N at the measured
# transmission ratio, so the simulation has been modelling the gripper at under
# half its rated strength. Virtual work against the measured closing rate
# converts one into the other: the two pads close at 106.23 mm per radian of
# `finger_joint` (evidence/gripper_collision_envelope.json), so a pad force N
# needs a drive torque of N x 0.10623 N-m.
#
#   235 N x 0.10623 m/rad = 24.96 N-m
#
# Kept as a measured constant because the experiment it enabled is a result
# worth not repeating: raising the drive to this value did *not* raise axial
# holding capacity, it lowered it. Pad force is not the binding constraint.
# See make_grapple_pin_robot_cfg and docs/status.md.
ROBOTIQ_2F85_RATED_GRIP_N = RATED_GRIP_FORCE_N[1]
ROBOTIQ_2F85_CLOSING_RATE_M_PER_RAD = CLOSING_RATE_M_PER_RAD
ROBOTIQ_2F85_RATED_DRIVE_TORQUE_NM = drive_torque_for_grip_force_nm(ROBOTIQ_2F85_RATED_GRIP_N)

# Captured from the validated differential-IK controller with the replacement
# blade centered and aligned 17 cm outside the rack.  Stage-1 PPO starts here
# instead of being asked to discover reaching, grasping, and insertion at once.
INSERTION_STAGING_ARM_JOINT_POS = (
    -0.2243491,
    -1.4427882,
    1.2887669,
    -1.4181530,
    -1.5709217,
    -3.3751323,
)
INSERTION_STAGING_BLADE_POS = (0.5829, 0.0, 0.72)

# Collision-valid poses sampled from the actual Isaac Lab differential-IK
# controller.  PPO starts near the goal, then the curriculum expands to the
# medium and full insertion distances.  The ordering is curriculum stage 0..2.
INSERTION_STAGE_ARM_JOINT_POS = (
    (-0.1899274, -1.1798824, 0.9525735, -1.3510975, -1.5791576, -3.3388975),
    (-0.2028731, -1.2927203, 1.1056069, -1.3898116, -1.5796102, -3.3510907),
    INSERTION_STAGING_ARM_JOINT_POS,
)
INSERTION_STAGE_BLADE_POSE = (
    (0.7194748, -0.0019335, 0.7216659, 0.9999804, -0.0046558, -0.0020612, 0.0036348),
    (0.6597763, -0.0020708, 0.7210070, 0.9999847, -0.0047566, -0.0005572, 0.0027788),
    (*INSERTION_STAGING_BLADE_POS, 1.0, 0.0, 0.0, 0.0),
)

# Tight-contact insertion must not begin interpenetrating a 0.75 mm-per-side
# slot.  The earlier curriculum poses contain roughly 2 mm lateral offsets
# that were tolerable only while a software fixture forced the blade.  Contact
# Phase 2.5 varies approach distance first and adds pose error in a later level.
CONTACT_INSERTION_STAGE_BLADE_POSE = (
    (INSERTION_STAGE_BLADE_POSE[0][0], 0.0, BLADE_INSERTED_POS[2], 1.0, 0.0, 0.0, 0.0),
    (INSERTION_STAGE_BLADE_POSE[1][0], 0.0, BLADE_INSERTED_POS[2], 1.0, 0.0, 0.0, 0.0),
    (*INSERTION_STAGING_BLADE_POS, 1.0, 0.0, 0.0, 0.0),
)

# The secured-grasp curriculum tolerated a small tool/handle offset. These
# contact-stage poses apply the measured IK correction so the real finger pads
# start centered on the handle instead of relying on the old spring.
CONTACT_INSERTION_STAGE_ARM_JOINT_POS = (
    (-0.1854645, -1.1956091, 0.9665422, -1.3518567, -1.5877903, -3.3365624),
    (-0.1984102, -1.3084470, 1.1195756, -1.3905708, -1.5882429, -3.3487556),
    (-0.2198862, -1.4585149, 1.3027356, -1.4189122, -1.5795544, -3.3727972),
)


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
    # The stock world-fixed joint is no longer part of the floating-base
    # articulation.  Deactivate it so PhysX does not parse its stale local
    # frames and snap the base before the external D6 mount takes control.
    source_prim.SetActive(False)


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


def spawn_fixed_grasp_joint(
    prim_path: str,
    cfg: FixedGraspJointCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
) -> Usd.Prim:
    """Author one external PhysX fixed grasp joint per environment."""

    if translation not in (None, (0.0, 0.0, 0.0)) or orientation not in (
        None,
        (1.0, 0.0, 0.0, 0.0),
    ):
        raise ValueError("FixedGraspJointCfg must be spawned at the identity pose.")
    root_path, leaf = str(prim_path).rsplit("/", 1)
    is_regex = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None
    parent_paths = find_matching_prim_paths(root_path) if is_regex else [root_path]
    if not parent_paths:
        raise RuntimeError(f"No environment parents matched fixed-grasp path '{prim_path}'.")

    stage = get_current_stage()
    source_prim: Usd.Prim | None = None
    for parent_path in parent_paths:
        container_path = f"{parent_path}/{leaf}"
        body0_path = f"{parent_path}/{cfg.body0_relative_path}"
        body1_path = f"{parent_path}/{cfg.body1_relative_path}"
        for body_path in (body0_path, body1_path):
            body = stage.GetPrimAtPath(body_path)
            if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Fixed grasp expected a rigid body at '{body_path}'.")

        container = UsdGeom.Xform.Define(stage, container_path)
        sim_utils.standardize_xform_ops(container.GetPrim())
        joint = UsdPhysics.FixedJoint.Define(stage, f"{container_path}/Joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*cfg.local_pos0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*cfg.local_pos1))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(cfg.local_rot0[0], Gf.Vec3f(*cfg.local_rot0[1:])))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(cfg.local_rot1[0], Gf.Vec3f(*cfg.local_rot1[1:])))
        source_prim = source_prim or container.GetPrim()
    assert source_prim is not None
    return source_prim


@configclass
class FixedGraspJointCfg(SpawnerCfg):
    """Rigid, gap-free abstraction for an already-secured gripper."""

    func: Callable[..., Usd.Prim] = spawn_fixed_grasp_joint
    body0_relative_path: str = "Robot/wrist_3_link"
    body1_relative_path: str = "SpareBlade"
    local_pos0: tuple[float, float, float] = CONTACT_TOOL_OFFSET_POS
    local_rot0: tuple[float, float, float, float] = TOOL_OFFSET_ROT
    local_pos1: tuple[float, float, float] = CONTACT_BLADE_HANDLE_OFFSET
    local_rot1: tuple[float, float, float, float] = GRIPPER_GRASP_ROT


def make_robot_cfg() -> ArticulationCfg:
    """Return the canonical UR10e/Robotiq asset with a compliant root."""

    cfg = UR10e_ROBOTIQ_2F_85_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_ROOT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Face the rack (+X).  The stock USD convention points the arm
                # away from the workcell when this joint is initialized at pi.
                # A small +Y staging angle keeps the open Robotiq geometry
                # clear of the pre-insertion blade while still facing +X.
                "shoulder_pan_joint": 0.35,
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


def make_insertion_robot_cfg() -> ArticulationCfg:
    """Return the fixed-base robot for nominal axial insertion training."""

    cfg = UR10e_ROBOTIQ_2F_85_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_ROOT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
            "shoulder_pan_joint": INSERTION_STAGING_ARM_JOINT_POS[0],
            "shoulder_lift_joint": INSERTION_STAGING_ARM_JOINT_POS[1],
            "elbow_joint": INSERTION_STAGING_ARM_JOINT_POS[2],
            "wrist_1_joint": INSERTION_STAGING_ARM_JOINT_POS[3],
            "wrist_2_joint": INSERTION_STAGING_ARM_JOINT_POS[4],
            "wrist_3_joint": INSERTION_STAGING_ARM_JOINT_POS[5],
            "finger_joint": 0.0,
            ".*_inner_finger_joint": 0.0,
            ".*_inner_finger_knuckle_joint": 0.0,
            ".*_outer_.*_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.spawn.rigid_props.disable_gravity = True
    cfg.spawn.activate_contact_sensors = False
    return cfg


def make_robust_insertion_robot_cfg() -> ArticulationCfg:
    """Return the insertion-staged robot with a floating compliant root."""

    cfg = make_insertion_robot_cfg()
    cfg.spawn.articulation_props.fix_root_link = False
    # The compliant-joint spawner relocates the stock articulation-root API
    # onto this rigid body before PhysX parses the floating articulation.
    cfg.articulation_root_prim_path = "/base_link"
    return cfg


def make_contact_insertion_robot_cfg(*, floating: bool = False) -> ArticulationCfg:
    """Return the insertion robot tuned for a real Robotiq contact grasp.

    The actuator values follow NVIDIA's Isaac Lab 2F-85 gear-assembly setup.
    They are stronger than the generic asset defaults but retain its 10 N-m
    simulated effort limit.  Extra solver iterations improve thin finger-pad
    contacts without adding contact sensors or changing the arm controller.
    """

    cfg = make_insertion_robot_cfg()
    # The external grasp joint is authored before the first manager reset.
    # Spawn the wrist at the same calibrated full-distance pose used by the
    # contact curriculum so PhysX does not begin by correcting a large joint
    # frame mismatch.
    for joint_name, position in zip(
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
        CONTACT_INSERTION_STAGE_ARM_JOINT_POS[2],
        strict=True,
    ):
        cfg.init_state.joint_pos[joint_name] = position
    drive = cfg.actuators["gripper_drive"]
    drive.effort_limit_sim = 10.0
    drive.velocity_limit_sim = 1.0
    drive.stiffness = 40.0
    drive.damping = 1.0
    fingers = cfg.actuators["gripper_finger"]
    fingers.effort_limit_sim = 10.0
    fingers.velocity_limit_sim = 10.0
    fingers.stiffness = 10.0
    fingers.damping = 0.05
    cfg.spawn.rigid_props.solver_position_iteration_count = 8
    cfg.spawn.rigid_props.solver_velocity_iteration_count = 2
    cfg.spawn.articulation_props.solver_position_iteration_count = 8
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 2
    if floating:
        cfg.spawn.articulation_props.fix_root_link = False
        cfg.articulation_root_prim_path = "/base_link"
    return cfg


def make_grapple_pin_robot_cfg(*, floating: bool = False) -> ArticulationCfg:
    """Return the head-on capture robot.

    One difference from the contact robot: the arm spawns at the head-on pose
    rather than the top-down one, so PhysX does not begin by correcting a
    quarter-turn joint mismatch.

    **The finger drive keeps its inherited 10 N-m limit, and that is a result,
    not an oversight.** Raising it to ``ROBOTIQ_2F85_RATED_DRIVE_TORQUE_NM``,
    the torque that produces the 2F-85's rated 235 N of grip at the measured
    transmission ratio, was tried on the hypothesis that pad force was the
    binding constraint on axial capacity. Against a matched grid it measured
    *worse*: 62 N held against 66 N for the 10 N-m drive, and complete loss of
    capture above 0.65 rad of closure. A wedge converts closing force into
    thrust along the pull axis, so in zero gravity more grip means a more
    violent capture transient on an unconstrained payload, and that transient
    spends the slip budget before the pull is even applied. See
    ``docs/status.md`` and ``evidence/grapple_pin_rated_grip_force.json``.
    """

    cfg = make_contact_insertion_robot_cfg(floating=floating)
    for joint_name, position in zip(
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
        GRAPPLE_HEAD_ON_ARM_JOINT_POS[2],
        strict=True,
    ):
        cfg.init_state.joint_pos[joint_name] = position
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
        if cfg.handle_collision_enabled:
            sim_utils.define_collision_properties(handle_path, cfg.collision_props, stage=stage)
            if cfg.handle_physics_material is not None:
                material_path = f"{root_path}/{cfg.handle_physics_material_path}"
                cfg.handle_physics_material.func(material_path, cfg.handle_physics_material)
                sim_utils.bind_physics_material(handle_path, material_path, stage=stage)
        material_path = f"{root_path}/geometry/{cfg.visual_material_path}"
        if stage.GetPrimAtPath(material_path).IsValid():
            sim_utils.bind_visual_material(handle_path, material_path, stage=stage)
    return root


def _define_box(
    stage: Usd.Stage,
    path: str,
    centre: tuple[float, float, float],
    size: tuple[float, float, float],
    cfg: GrapplePinBladeCfg,
    material_path: str,
) -> None:
    """Author one axis-aligned collision box inside the blade's rigid body."""

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    sim_utils.standardize_xform_ops(
        box.GetPrim(),
        translation=centre,
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=size,
    )
    sim_utils.define_collision_properties(path, cfg.collision_props, stage=stage)
    sim_utils.bind_physics_material(path, material_path, stage=stage)


def _define_wedge(stage: Usd.Stage, path: str, cfg: GrapplePinBladeCfg, material_path: str) -> None:
    """Author the tapered capture wedge as an explicit eight-vertex frustum.

    There is no primitive for a truncated wedge, and approximating it with a
    cone would give the flat pads point contact on a circular section instead of
    line contact across the full pin width.  The taper is in the closing axis
    only, so the pin keeps a constant width and both pads bear evenly.
    """

    distal_x, proximal_x = cfg.wedge_x
    distal_half, proximal_half = cfg.wedge_half_height
    half_width = cfg.half_width_y
    points = [
        (distal_x, -half_width, -distal_half),
        (distal_x, half_width, -distal_half),
        (distal_x, half_width, distal_half),
        (distal_x, -half_width, distal_half),
        (proximal_x, -half_width, -proximal_half),
        (proximal_x, half_width, -proximal_half),
        (proximal_x, half_width, proximal_half),
        (proximal_x, -half_width, proximal_half),
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        # Free end, blade end, then the two flanks and the two tapered faces,
        # each wound counter-clockwise seen from outside.
        [0, 3, 2, 1] + [4, 5, 6, 7] + [0, 4, 7, 3] + [1, 2, 6, 5] + [3, 7, 6, 2] + [0, 1, 5, 4]
    )
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(distal_x, -half_width, -distal_half),
            Gf.Vec3f(proximal_x, half_width, distal_half),
        ]
    )
    sim_utils.standardize_xform_ops(mesh.GetPrim())
    sim_utils.define_collision_properties(path, cfg.collision_props, stage=stage)
    # Without this PhysX defaults a mesh collider to a triangle mesh, which
    # cannot be part of a dynamic body.
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    sim_utils.bind_physics_material(path, material_path, stage=stage)


def spawn_blade_with_grapple_pin(
    prim_path: str,
    cfg: GrapplePinBladeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn the blade chassis with a head-on grapple pin on its ``-x`` face.

    Three colliders share the blade's rigid body: a shaft thin enough to pass
    through the slot, a collar the pads seat against, and the tapered wedge that
    carries the axial load.
    """

    root = sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = get_current_stage()
    for root_path in find_matching_prim_paths(prim_path):
        pin_path = f"{root_path}/GrapplePin"
        if stage.GetPrimAtPath(pin_path).IsValid():
            continue
        UsdGeom.Xform.Define(stage, pin_path)
        material_path = f"{root_path}/{cfg.pin_physics_material_path}"
        cfg.pin_physics_material.func(material_path, cfg.pin_physics_material)

        shaft_low, shaft_high = cfg.shaft_x
        collar_low, collar_high = cfg.collar_x
        _define_box(
            stage,
            f"{pin_path}/Shaft",
            (0.5 * (shaft_low + shaft_high), 0.0, 0.0),
            (shaft_high - shaft_low, 2.0 * cfg.half_width_y, 2.0 * cfg.shaft_half_height),
            cfg,
            material_path,
        )
        _define_box(
            stage,
            f"{pin_path}/Collar",
            (0.5 * (collar_low + collar_high), 0.0, 0.0),
            (collar_high - collar_low, 2.0 * cfg.half_width_y, 2.0 * cfg.collar_half_height),
            cfg,
            material_path,
        )
        _define_wedge(stage, f"{pin_path}/Wedge", cfg, material_path)
        names = ["Shaft", "Collar", "Wedge"]

        visual_path = f"{root_path}/geometry/{cfg.visual_material_path}"
        if stage.GetPrimAtPath(visual_path).IsValid():
            for name in names:
                sim_utils.bind_visual_material(f"{pin_path}/{name}", visual_path, stage=stage)
    return root


@configclass
class GrapplePinBladeCfg(sim_utils.CuboidCfg):
    """Blade chassis carrying a head-on grapple pin as extra colliders.

    ``handle_offset`` is kept as the name of the point a tool frame should be
    driven to, because the grasp metrics, the calibration servo, and the
    diagnostics all read it by that name.  Here it is the centre of the length
    of pin the pads close on.
    """

    func: Callable[..., Usd.Prim] = spawn_blade_with_grapple_pin
    handle_offset: tuple[float, float, float] = GRAPPLE_PIN_GRIP_OFFSET
    shaft_x: tuple[float, float] = GRAPPLE_PIN_SHAFT_X
    collar_x: tuple[float, float] = GRAPPLE_PIN_COLLAR_X
    wedge_x: tuple[float, float] = GRAPPLE_PIN_WEDGE_X
    half_width_y: float = GRAPPLE_PIN_HALF_WIDTH_Y
    shaft_half_height: float = GRAPPLE_PIN_SHAFT_HALF_HEIGHT
    collar_half_height: float = GRAPPLE_PIN_COLLAR_HALF_HEIGHT
    wedge_half_height: tuple[float, float] = GRAPPLE_PIN_WEDGE_HALF_HEIGHT
    # The second-generation anti-yaw walls, off by default. Turning them on
    # changes the contact every trained policy was produced against, so they are
    # measured on the physics gates first and only then trained against. That
    # order is the same one the pull gate established: never train a policy
    # against an interface that has not been characterised.
    pin_physics_material_path: str = "grapplePinPhysicsMaterial"
    # The wedge is meant to hold by geometry, so the gate must not be able to
    # pass on friction the interface would not have. This is an ordinary
    # machined-metal pairing, not the 1.2 the old rubber-on-handle grip assumed.
    pin_physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.4,
        dynamic_friction=0.3,
        restitution=0.0,
        friction_combine_mode="min",
    )


@configclass
class BladeCuboidCfg(sim_utils.CuboidCfg):
    """Cuboid chassis with one collision handle sharing the root rigid body."""

    func: Callable[..., Usd.Prim] = spawn_blade_with_handle
    handle_offset: tuple[float, float, float] = BLADE_HANDLE_OFFSET
    # A 55 mm grip width leaves 15 mm clearance inside the 2F-85 aperture;
    # 50 mm depth gives both finger pads meaningful contact area.
    handle_size: tuple[float, float, float] = (0.050, 0.055, 0.025)
    handle_collision_enabled: bool = True
    handle_physics_material_path: str = "handlePhysicsMaterial"
    handle_physics_material: sim_utils.RigidBodyMaterialCfg | None = None


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
INSERTION_BLADE_CFG = _blade_cfg(
    "SpareBlade",
    INSERTION_STAGING_BLADE_POS,
    (0.04, 0.18, 0.30),
    "rigidly_attached_replacement_blade",
)
INSERTION_BLADE_CFG.spawn.handle_collision_enabled = False
INSERTION_BLADE_CFG.spawn.physics_material.static_friction = 0.12
INSERTION_BLADE_CFG.spawn.physics_material.dynamic_friction = 0.08
INSERTION_BLADE_CFG.spawn.physics_material.friction_combine_mode = "min"


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
    """Create a side guide around the nominal 1.5 mm slot clearance.

    The contact offset gives PhysX an early contact envelope at the guide faces;
    the explicit axial stiction term supplies the zero-g breakaway/drag load.
    This avoids starting the rigid blade in an impossible interpenetrating pose.
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


SLOT_LEFT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotLeftGuide", (BLADE_INSERTED_POS[0], GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2])
)
SLOT_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotRightGuide", (BLADE_INSERTED_POS[0], -GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2])
)

# Phase 1 deliberately teaches nominal axial motion before tight-tolerance
# contact.  Five millimetres per side and a one-millimetre contact envelope
# avoid the impossible overlap in which a 0.75 mm clearance was smaller than
# the old 3 mm contact envelope.  Later phases restore production clearance.
INSERTION_GUIDE_CENTER_OFFSET_Y = 0.094
INSERTION_SLOT_CFG = SLOT_CFG.copy()
INSERTION_SLOT_CFG.spawn.collision_props.contact_offset = 0.001
INSERTION_SLOT_CFG.spawn.physics_material.static_friction = 0.12
INSERTION_SLOT_CFG.spawn.physics_material.dynamic_friction = 0.08
INSERTION_SLOT_CFG.spawn.physics_material.friction_combine_mode = "min"
INSERTION_SLOT_LEFT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotLeftGuide",
    (BLADE_INSERTED_POS[0], INSERTION_GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2]),
)
INSERTION_SLOT_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotRightGuide",
    (BLADE_INSERTED_POS[0], -INSERTION_GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2]),
)
for _insertion_guide_cfg in (INSERTION_SLOT_LEFT_GUIDE_CFG, INSERTION_SLOT_RIGHT_GUIDE_CFG):
    _insertion_guide_cfg.spawn.collision_props.contact_offset = 0.001
    _insertion_guide_cfg.spawn.physics_material.static_friction = 0.12
    _insertion_guide_cfg.spawn.physics_material.dynamic_friction = 0.08
    _insertion_guide_cfg.spawn.physics_material.friction_combine_mode = "min"

# Phase 2 restores the nominal 1.5 mm total slot clearance, but uses a contact
# envelope smaller than the 0.75 mm clearance on each side.  The old 3 mm
# envelope made the geometry overlap before the blade touched either rail.
ROBUST_INSERTION_BLADE_CFG = INSERTION_BLADE_CFG.copy()
ROBUST_INSERTION_BLADE_CFG.spawn.collision_props.contact_offset = 0.0003
ROBUST_INSERTION_BLADE_CFG.spawn.physics_material.static_friction = 0.55
ROBUST_INSERTION_BLADE_CFG.spawn.physics_material.dynamic_friction = 0.45
ROBUST_INSERTION_BLADE_CFG.spawn.physics_material.friction_combine_mode = "max"
ROBUST_INSERTION_SLOT_CFG = SLOT_CFG.copy()
ROBUST_INSERTION_SLOT_CFG.spawn.collision_props.contact_offset = 0.0004
ROBUST_INSERTION_SLOT_CFG.spawn.physics_material.static_friction = 0.12
ROBUST_INSERTION_SLOT_CFG.spawn.physics_material.dynamic_friction = 0.08
ROBUST_INSERTION_SLOT_CFG.spawn.physics_material.friction_combine_mode = "min"
ROBUST_INSERTION_SLOT_LEFT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotLeftGuide",
    (BLADE_INSERTED_POS[0], GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2]),
)
ROBUST_INSERTION_SLOT_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotRightGuide",
    (BLADE_INSERTED_POS[0], -GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2]),
)
for _robust_guide_cfg in (ROBUST_INSERTION_SLOT_LEFT_GUIDE_CFG, ROBUST_INSERTION_SLOT_RIGHT_GUIDE_CFG):
    _robust_guide_cfg.spawn.collision_props.contact_offset = 0.0004
    _robust_guide_cfg.spawn.physics_material.static_friction = 0.12
    _robust_guide_cfg.spawn.physics_material.dynamic_friction = 0.08
    _robust_guide_cfg.spawn.physics_material.friction_combine_mode = "min"

# Rigid-grasp level 0 is deliberately collision-free.  The visibly identical
# wide and tight assets below let the fixed-joint policy add real rail contact
# only after it can solve free-space insertion on held-out episodes.
RIGID_GRASP_SLOT_CFG = ROBUST_INSERTION_SLOT_CFG.copy()
RIGID_GRASP_SLOT_CFG.init_state.pos = (BLADE_INSERTED_POS[0], 0.0, 0.6940)
RIGID_GRASP_SLOT_CFG.spawn.collision_props.collision_enabled = False
RIGID_GRASP_GUIDE_CENTER_OFFSET_Y = 0.095
RIGID_GRASP_SLOT_LEFT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotLeftGuide", (BLADE_INSERTED_POS[0], RIGID_GRASP_GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2])
)
RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "BladeSlotRightGuide", (BLADE_INSERTED_POS[0], -RIGID_GRASP_GUIDE_CENTER_OFFSET_Y, BLADE_INSERTED_POS[2])
)
for _rigid_grasp_guide_cfg in (RIGID_GRASP_SLOT_LEFT_GUIDE_CFG, RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG):
    _rigid_grasp_guide_cfg.spawn.collision_props.collision_enabled = False
    _rigid_grasp_guide_cfg.spawn.collision_props.contact_offset = 0.0004
    _rigid_grasp_guide_cfg.spawn.physics_material.static_friction = 0.12
    _rigid_grasp_guide_cfg.spawn.physics_material.dynamic_friction = 0.08
    _rigid_grasp_guide_cfg.spawn.physics_material.friction_combine_mode = "min"

RIGID_GRASP_WIDE_SLOT_CFG = RIGID_GRASP_SLOT_CFG.copy()
RIGID_GRASP_WIDE_LEFT_GUIDE_CFG = RIGID_GRASP_SLOT_LEFT_GUIDE_CFG.copy()
RIGID_GRASP_WIDE_RIGHT_GUIDE_CFG = RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG.copy()
for _rigid_grasp_wide_guide_cfg in (
    RIGID_GRASP_WIDE_LEFT_GUIDE_CFG,
    RIGID_GRASP_WIDE_RIGHT_GUIDE_CFG,
):
    _rigid_grasp_wide_guide_cfg.spawn.collision_props.collision_enabled = True

RIGID_GRASP_TIGHT_SLOT_CFG = ROBUST_INSERTION_SLOT_CFG.copy()
RIGID_GRASP_TIGHT_LEFT_GUIDE_CFG = ROBUST_INSERTION_SLOT_LEFT_GUIDE_CFG.copy()
RIGID_GRASP_TIGHT_RIGHT_GUIDE_CFG = ROBUST_INSERTION_SLOT_RIGHT_GUIDE_CFG.copy()
for _rigid_grasp_tight_guide_cfg in (
    RIGID_GRASP_TIGHT_LEFT_GUIDE_CFG,
    RIGID_GRASP_TIGHT_RIGHT_GUIDE_CFG,
):
    # Do not rely on Isaac Lab's ``None means enabled`` default here: the
    # profile contract and saved env configuration must be unambiguous.
    _rigid_grasp_tight_guide_cfg.spawn.collision_props.collision_enabled = True

# Phase 2.5 removes the compliant software grasp.  Its handle is a little
# taller than the chassis so both Robotiq pads visibly surround it, and its
# collider is deliberately re-enabled after the secured-blade curriculum.
CONTACT_INSERTION_BLADE_CFG = ROBUST_INSERTION_BLADE_CFG.copy()
CONTACT_INSERTION_BLADE_CFG.spawn.handle_collision_enabled = True
# Keep the handle short enough that its raised lower surface cannot
# interpenetrate the slot floor while the blade is rail-aligned.
CONTACT_INSERTION_BLADE_CFG.spawn.handle_size = (0.060, 0.075, 0.030)
# Raise the service tab into the imported fingertip-pad workspace. Its lower
# surface remains well clear of the slot floor at the nominal blade pose.
CONTACT_INSERTION_BLADE_CFG.spawn.handle_offset = CONTACT_BLADE_HANDLE_OFFSET
# Keep the metal chassis easy to slide at contact-curriculum level 0.  The
# handle below has its own high-friction material, so stronger finger grip no
# longer also makes the chassis unrealistically sticky against the guide rails.
CONTACT_INSERTION_BLADE_CFG.spawn.physics_material.static_friction = 0.12
CONTACT_INSERTION_BLADE_CFG.spawn.physics_material.dynamic_friction = 0.08
CONTACT_INSERTION_BLADE_CFG.spawn.physics_material.friction_combine_mode = "min"
CONTACT_INSERTION_BLADE_CFG.spawn.handle_physics_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2,
    dynamic_friction=1.0,
    restitution=0.0,
    friction_combine_mode="max",
)

# The fixed joint already models a secured grasp.  Keeping a second collider
# inside the closed fingers makes PhysX solve contradictory internal contacts,
# which previously tilted/ejected the blade when rail collision was enabled.
# The handle remains visible; only its collision is disabled for this task.
RIGID_GRASP_BLADE_CFG = CONTACT_INSERTION_BLADE_CFG.copy()
RIGID_GRASP_BLADE_CFG.spawn.handle_collision_enabled = False
# Copied before the grapple post is fitted below, and pinned to the short tab
# the promoted Level-0/1/2 checkpoints were certified with. The post is a grasp
# interface; this task has no grasp to make.
RIGID_GRASP_BLADE_CFG.spawn.handle_offset = CONTACT_BLADE_HANDLE_OFFSET
RIGID_GRASP_BLADE_CFG.spawn.handle_size = (0.060, 0.075, 0.030)

# Only the physical-grasp task gets the post.
CONTACT_INSERTION_BLADE_CFG.spawn.handle_offset = GRAPPLE_POST_OFFSET
CONTACT_INSERTION_BLADE_CFG.spawn.handle_size = GRAPPLE_POST_SIZE

# The head-on capture blade. Its chassis physics match the tight-clearance
# insertion blade so rail contact behaves the same; only the grasp interface
# differs. Contact reporting stays off by default, as on every other blade here,
# and a task switches it on when it needs force metrics.
GRAPPLE_PIN_BLADE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SpareBlade",
    spawn=GrapplePinBladeCfg(
        size=BLADE_SIZE,
        rigid_props=_rigid_props(kinematic=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0003, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.55,
            dynamic_friction=0.45,
            restitution=0.0,
            friction_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.04, 0.18, 0.30), metallic=0.75, roughness=0.25
        ),
        semantic_tags=[("class", "grapple_pin_replacement_blade")],
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=INSERTION_STAGING_BLADE_POS),
)


# ---------------------------------------------------------------------------
# Guided slot: a four-sided channel with a funnelled mouth.
#
# The certified slot is two vertical side walls and nothing else, so nothing
# physically opposes blade pitch or roll and the policy has to hold the blade
# level by commanding orientation alone. That is why terminal orientation error
# eats 97.8% of its tolerance. A real 1U card guide is a channel: the card is
# captured top and bottom as well as side to side, and the mouth is flared so a
# slightly misaligned card is funnelled in rather than jammed.
#
# Geometry, all derived from the existing blade and rails:
#   blade            450 x 160 x 35 mm, centre z 0.72, so its deck is at 0.7375
#   side rails       inner faces at |y| = 80.75 mm -> 0.75 mm clearance a side
#   upper lips       bottom face at z 0.7385 -> 1.0 mm of lift before contact
#   lip span in y    62.5 to 82.5 mm, overhanging the blade edge by 17.5 mm
#   lip span in x    0.60 to 1.05, leaving the first 150 mm of slot clear so the
#                    gripper and the 75 mm-wide grapple post pass between them
#
# With about 1 mm of lift at each end of a 450 mm blade, pitch is mechanically
# limited to roughly 0.0044 rad (0.25 deg) instead of the 0.052 rad the policy
# currently has to control. The channel does the alignment, not the policy.
SLOT_UPPER_LIP_HALF_WIDTH_Y = 0.0725
SLOT_UPPER_LIP_CENTER_Z = 0.7435


def _slot_upper_lip_cfg(name: str, y_center: float) -> RigidObjectCfg:
    """Create one overhanging upper rail lip for the guided channel."""

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.020, 0.010),
            rigid_props=_rigid_props(kinematic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.65,
                restitution=0.0,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.58, 0.62), metallic=0.9, roughness=0.22),
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.825, y_center, SLOT_UPPER_LIP_CENTER_Z)),
    )


SLOT_UPPER_LEFT_LIP_CFG = _slot_upper_lip_cfg("BladeSlotUpperLeftLip", SLOT_UPPER_LIP_HALF_WIDTH_Y)
SLOT_UPPER_RIGHT_LIP_CFG = _slot_upper_lip_cfg("BladeSlotUpperRightLip", -SLOT_UPPER_LIP_HALF_WIDTH_Y)

# Lateral lead-in. Each flare is an 80 mm plate rotated 12 degrees about z and
# placed so its inner face meets the rail face exactly at the slot mouth
# (x = 0.45, |y| = 80.75 mm) and opens to |y| = 97.4 mm at x = 0.372. That turns
# a 0.75 mm-per-side keyhole into a 16.6 mm-per-side catch: the blade can arrive
# over 20x further off centre and still be walked into the channel by contact
# instead of missing the slot entirely.
SLOT_ENTRY_FLARE_DEG = 12.0
_FLARE_CENTER_X = 0.41275
_FLARE_CENTER_Y = 0.097869
# Scalar-first quaternion for a rotation of -12 degrees about z.
_FLARE_QUAT_LEFT = (0.9945219, 0.0, 0.0, -0.1045285)
_FLARE_QUAT_RIGHT = (0.9945219, 0.0, 0.0, 0.1045285)


def _slot_entry_flare_cfg(name: str, y_center: float, rotation: tuple[float, float, float, float]) -> RigidObjectCfg:
    """Create one angled lead-in plate at the slot mouth."""

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.080, 0.018, 0.050),
            rigid_props=_rigid_props(kinematic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # A lead-in that grabs defeats the point, so it is the slipperiest
                # surface in the slot.
                static_friction=0.25,
                dynamic_friction=0.20,
                restitution=0.0,
                friction_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.52, 0.16), metallic=0.85, roughness=0.30),
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(_FLARE_CENTER_X, y_center, BLADE_INSERTED_POS[2]), rot=rotation),
    )


SLOT_ENTRY_LEFT_FLARE_CFG = _slot_entry_flare_cfg("BladeSlotEntryLeftFlare", _FLARE_CENTER_Y, _FLARE_QUAT_LEFT)
SLOT_ENTRY_RIGHT_FLARE_CFG = _slot_entry_flare_cfg("BladeSlotEntryRightFlare", -_FLARE_CENTER_Y, _FLARE_QUAT_RIGHT)


# ---------------------------------------------------------------------------
# The second slot, which is what makes a relocation a relocation.
#
# A rack with one slot can only demonstrate remove-and-replace. Moving a module
# from one bay to the neighbouring one is the ORU changeout a servicer actually
# performs, and it needs the rack to have somewhere else to put it.
#
# Placed at y = -0.22 m. The module is 0.16 m wide, so two modules sitting in
# adjacent slots leave 60 mm between them -- enough that the gripper's 75 mm
# closing envelope is not trying to occupy the same space as the neighbour.
#
# Every part is *derived* from the slot that is already certified rather than
# re-authored: same rails, same clearance, same lips, same lead-in flares, same
# materials, displaced along y. Re-authoring would let the two slots drift apart
# and would make any difference between them unattributable.
SECOND_SLOT_CENTER_Y = -0.22

#: Every bay's centre line, in the order the curriculum stage, the insertion goal
#: and the perception occupancy label all index. One tuple so those three cannot
#: disagree about which bay index 1 means -- the same reasoning that made the
#: two-slot task read its arm pose, module pose and goal from a single index.
SLOT_CENTRE_Y: tuple[float, ...] = (0.0, SECOND_SLOT_CENTER_Y)


def offset_slot_asset(source: RigidObjectCfg, name: str, delta_y: float) -> RigidObjectCfg:
    """Copy one slot part to a new prim, displaced along y.

    ``configclass`` copies deeply, so the spawn, collision properties and
    physics material of the result are independent of the original and of every
    other copy -- which matters because several tasks mutate those in place.
    """

    moved = source.copy()
    moved.prim_path = f"{{ENV_REGEX_NS}}/{name}"
    position = moved.init_state.pos
    moved.init_state.pos = (position[0], position[1] + delta_y, position[2])
    return moved


SECOND_SLOT_CFG = offset_slot_asset(ROBUST_INSERTION_SLOT_CFG, "BladeSlotTwo", SECOND_SLOT_CENTER_Y)
SECOND_SLOT_LEFT_GUIDE_CFG = offset_slot_asset(
    ROBUST_INSERTION_SLOT_LEFT_GUIDE_CFG, "BladeSlotTwoLeftGuide", SECOND_SLOT_CENTER_Y
)
SECOND_SLOT_RIGHT_GUIDE_CFG = offset_slot_asset(
    ROBUST_INSERTION_SLOT_RIGHT_GUIDE_CFG, "BladeSlotTwoRightGuide", SECOND_SLOT_CENTER_Y
)
SECOND_SLOT_UPPER_LEFT_LIP_CFG = offset_slot_asset(
    SLOT_UPPER_LEFT_LIP_CFG, "BladeSlotTwoUpperLeftLip", SECOND_SLOT_CENTER_Y
)
SECOND_SLOT_UPPER_RIGHT_LIP_CFG = offset_slot_asset(
    SLOT_UPPER_RIGHT_LIP_CFG, "BladeSlotTwoUpperRightLip", SECOND_SLOT_CENTER_Y
)
SECOND_SLOT_ENTRY_LEFT_FLARE_CFG = offset_slot_asset(
    SLOT_ENTRY_LEFT_FLARE_CFG, "BladeSlotTwoEntryLeftFlare", SECOND_SLOT_CENTER_Y
)
SECOND_SLOT_ENTRY_RIGHT_FLARE_CFG = offset_slot_asset(
    SLOT_ENTRY_RIGHT_FLARE_CFG, "BladeSlotTwoEntryRightFlare", SECOND_SLOT_CENTER_Y
)

#: Where a module sits when it is seated in the second slot. Derived from the
#: first slot's seated pose, so the two can never disagree about depth.
SECOND_SLOT_INSERTED_POS = (
    BLADE_INSERTED_POS[0],
    BLADE_INSERTED_POS[1] + SECOND_SLOT_CENTER_Y,
    BLADE_INSERTED_POS[2],
)


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
    "SupplyCaddyLeftGuide",
    (SPARE_BLADE_POS[0], SPARE_BLADE_POS[1] + GUIDE_CENTER_OFFSET_Y, SPARE_BLADE_POS[2]),
    0.55,
)
SUPPLY_CADDY_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "SupplyCaddyRightGuide",
    (SPARE_BLADE_POS[0], SPARE_BLADE_POS[1] - GUIDE_CENTER_OFFSET_Y, SPARE_BLADE_POS[2]),
    0.55,
)
SERVICE_CADDY_LEFT_GUIDE_CFG = _slot_guide_cfg(
    "ServiceCaddyLeftGuide",
    (SERVICE_CADDY_POS[0], SERVICE_CADDY_POS[1] + GUIDE_CENTER_OFFSET_Y, SERVICE_CADDY_POS[2]),
    0.55,
)
SERVICE_CADDY_RIGHT_GUIDE_CFG = _slot_guide_cfg(
    "ServiceCaddyRightGuide",
    (SERVICE_CADDY_POS[0], SERVICE_CADDY_POS[1] - GUIDE_CENTER_OFFSET_Y, SERVICE_CADDY_POS[2]),
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
    "BLADE_HANDLE_OFFSET",
    "BLADE_INSERTED_POS",
    "BLADE_SIZE",
    "CompliantD6JointCfg",
    "FixedGraspJointCfg",
    "CONTACT_BLADE_HANDLE_OFFSET",
    "CONTACT_INSERTION_STAGE_ARM_JOINT_POS",
    "CONTACT_INSERTION_STAGE_BLADE_POSE",
    "CONTACT_TOOL_OFFSET_POS",
    "GRASP_TOOL_OFFSET_POS",
    "GRAPPLE_HEAD_ON_ARM_JOINT_POS",
    "GRAPPLE_HEAD_ON_TOOL_ROT",
    "GRAPPLE_PIN_BLADE_CFG",
    "GRAPPLE_PIN_COLLAR_HALF_HEIGHT",
    "GRAPPLE_PIN_COLLAR_X",
    "GRAPPLE_PIN_GRIP_OFFSET",
    "GRAPPLE_PIN_HALF_WIDTH_Y",
    "GRAPPLE_PIN_SHAFT_HALF_HEIGHT",
    "GRAPPLE_PIN_SHAFT_X",
    "GRAPPLE_PIN_TAPER_DEG",
    "GRAPPLE_PIN_WEDGE_HALF_HEIGHT",
    "GRAPPLE_PIN_WEDGE_X",
    "GRAPPLE_POST_OFFSET",
    "GRAPPLE_POST_SIZE",
    "GRAPPLE_TOOL_OFFSET_POS",
    "GrapplePinBladeCfg",
    "ROBOTIQ_2F85_CLOSING_RATE_M_PER_RAD",
    "ROBOTIQ_2F85_RATED_DRIVE_TORQUE_NM",
    "ROBOTIQ_2F85_RATED_GRIP_N",
    "SECOND_SLOT_CENTER_Y",
    "SECOND_SLOT_CFG",
    "SECOND_SLOT_ENTRY_LEFT_FLARE_CFG",
    "SECOND_SLOT_ENTRY_RIGHT_FLARE_CFG",
    "SECOND_SLOT_INSERTED_POS",
    "SECOND_SLOT_LEFT_GUIDE_CFG",
    "SECOND_SLOT_RIGHT_GUIDE_CFG",
    "SECOND_SLOT_UPPER_LEFT_LIP_CFG",
    "SECOND_SLOT_UPPER_RIGHT_LIP_CFG",
    "SLOT_ENTRY_FLARE_DEG",
    "SLOT_ENTRY_LEFT_FLARE_CFG",
    "SLOT_ENTRY_RIGHT_FLARE_CFG",
    "offset_slot_asset",
    "SLOT_UPPER_LEFT_LIP_CFG",
    "SLOT_UPPER_RIGHT_LIP_CFG",
    "GRIPPER_GRASP_ROT",
    "INSERTION_BLADE_CFG",
    "INSERTION_GUIDE_CENTER_OFFSET_Y",
    "INSERTION_SLOT_CFG",
    "INSERTION_SLOT_LEFT_GUIDE_CFG",
    "INSERTION_SLOT_RIGHT_GUIDE_CFG",
    "INSERTION_STAGE_ARM_JOINT_POS",
    "INSERTION_STAGE_BLADE_POSE",
    "INSERTION_STAGING_ARM_JOINT_POS",
    "INSERTION_STAGING_BLADE_POS",
    "MOUNT_ANCHOR_CFG",
    "RACK_CFG",
    "CONTACT_INSERTION_BLADE_CFG",
    "ROBUST_INSERTION_BLADE_CFG",
    "ROBUST_INSERTION_SLOT_CFG",
    "ROBUST_INSERTION_SLOT_LEFT_GUIDE_CFG",
    "ROBUST_INSERTION_SLOT_RIGHT_GUIDE_CFG",
    "RIGID_GRASP_SLOT_CFG",
    "RIGID_GRASP_SLOT_LEFT_GUIDE_CFG",
    "RIGID_GRASP_SLOT_RIGHT_GUIDE_CFG",
    "RIGID_GRASP_BLADE_CFG",
    "RIGID_GRASP_TIGHT_LEFT_GUIDE_CFG",
    "RIGID_GRASP_TIGHT_RIGHT_GUIDE_CFG",
    "RIGID_GRASP_TIGHT_SLOT_CFG",
    "RIGID_GRASP_WIDE_LEFT_GUIDE_CFG",
    "RIGID_GRASP_WIDE_RIGHT_GUIDE_CFG",
    "RIGID_GRASP_WIDE_SLOT_CFG",
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
    "TRANSFER_BLADE_X",
    "TOOL_OFFSET_POS",
    "TOOL_OFFSET_ROT",
    "make_insertion_robot_cfg",
    "make_contact_insertion_robot_cfg",
    "make_grapple_pin_robot_cfg",
    "make_robust_insertion_robot_cfg",
    "make_robot_cfg",
    "spawn_blade_with_grapple_pin",
    "spawn_compliant_d6_joint",
    "spawn_fixed_grasp_joint",
]
