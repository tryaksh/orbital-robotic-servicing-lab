"""Visible mounting context for an offline replay of the fixed-base workcell.

Call ``add_fixed_service_structure`` after applying the first recorded body
state. These meshes illustrate the existing fixed robot/rack mounting
assumption; they introduce no physical bodies, collisions, joints, or motion.
USD is imported only when a validated stationary recording is rendered.
"""

from __future__ import annotations

import math

import numpy as np

STATIONARY_POSITION_TOLERANCE_M = 1.0e-5
RACK_SIZE_M = (0.035, 0.72, 1.15)
ANCHOR_SIZE_M = (0.12, 0.12, 0.06)


def add_fixed_service_structure(stage, env_path: str, robot_base_positions) -> dict:
    """Add a static open frame beneath the captured robot and behind the rack.

    ``robot_base_positions`` is the complete (N, 3) world-position capture, not
    just an initial pose. A moving base is rejected before any stage mutation.
    Apply recorded state 0 before calling: the rack's current world transform
    supplies the other mounting endpoint. The shipped rack dimensions remain
    explicit because its subtree also includes moving retention pawls, whose
    aggregate bounds would be an incorrect rack envelope.
    """
    positions = np.asarray(robot_base_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 3:
        raise ValueError("Supply the complete captured robot-base positions as an (N >= 2, 3) array")
    if not np.isfinite(positions).all():
        raise ValueError("Robot-base positions must be finite")
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    max_displacement = float(displacement.max())
    if max_displacement > STATIONARY_POSITION_TOLERANCE_M:
        raise ValueError(
            f"A fixed service frame would misrepresent this moving base: {max_displacement:.9f} m travel "
            f"exceeds {STATIONARY_POSITION_TOLERANCE_M:.9f} m"
        )

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    env_path = env_path.rstrip("/")
    if not env_path.startswith("/") or not stage.GetPrimAtPath(env_path).IsValid():
        raise ValueError("env_path must identify an existing absolute USD environment path")
    rack_prim = stage.GetPrimAtPath(f"{env_path}/Rack")
    if not rack_prim.IsValid():
        raise ValueError("The existing Rack prim is required; refusing to invent its mounting location")
    rack_transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(rack_prim)
    # The mounting dimensions describe the existing upright, axis-aligned rack.
    # Reject a differently oriented/scaled asset instead of building through it.
    rack_basis = np.asarray(rack_transform, dtype=float)[:3, :3]
    if not np.allclose(rack_basis, np.eye(3), atol=1.0e-5):
        raise ValueError("This fixed service-frame layout requires the shipped axis-aligned, unscaled Rack root")
    rack = np.asarray(rack_transform.ExtractTranslation(), dtype=float)
    base = positions[0]
    deck_top = float(base[2] - 0.05)
    rack_bottom = float(rack[2] - RACK_SIZE_M[2] / 2)
    rack_top = float(rack[2] + RACK_SIZE_M[2] / 2)
    if rack_bottom <= deck_top or rack[0] <= base[0] + 0.5:
        raise ValueError("Rack/base geometry does not match the fixed service workcell; apply captured state 0 first")
    if abs(base[1] - rack[1]) > 0.08:
        raise ValueError("The fixed mounting plate would miss the frame's source-bay cross-member")
    root = f"{env_path}/ServiceStructure"
    if stage.GetPrimAtPath(root).IsValid():
        raise ValueError(f"{root} already exists; refusing to overwrite stage content")

    container = UsdGeom.Xform.Define(stage, root)
    # Recorded body poses are in world space; keep environment placement from
    # applying twice even when a renderer uses reset-xform stacks on the bodies.
    container.SetResetXformStack(True)
    container.GetPrim().SetCustomDataByKey("presentationOnly", True)
    container.GetPrim().SetCustomDataByKey("physicalLoadPath", "Existing fixed robot/rack abstraction; visuals only")
    materials = {}
    for name, colour, metallic, roughness in (
        ("FrameGraphite", (0.16, 0.20, 0.24), 0.65, 0.40),
        ("BrushedAluminium", (0.43, 0.49, 0.55), 0.80, 0.34),
        ("FastenerSteel", (0.58, 0.64, 0.69), 0.85, 0.26),
    ):
        material = UsdShade.Material.Define(stage, f"{root}/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material

    geometry = []

    def finish(prim, material_name):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[material_name])

    def cube(name, centre, size, material="FrameGraphite"):
        shape = UsdGeom.Cube.Define(stage, f"{root}/{name}")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
        shape.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        finish(shape.GetPrim(), material)
        geometry.append({"name": name, "shape": "cuboid", "centre_world_m": list(centre), "size_m": list(size)})

    def bolt(name, x, y, surface_z):
        # A washer and a small flat-sided head remain legible at the wide view.
        washer = UsdGeom.Cylinder.Define(stage, f"{root}/{name}Washer")
        washer.CreateAxisAttr("Z")
        washer.CreateRadiusAttr(0.011)
        washer.CreateHeightAttr(0.0014)
        washer.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(surface_z + 0.0007)))
        finish(washer.GetPrim(), "FastenerSteel")
        radius, height = 0.008, 0.006
        vertices = [
            Gf.Vec3f(radius * math.cos(index * math.pi / 3), radius * math.sin(index * math.pi / 3), z)
            for z in (-height / 2, height / 2) for index in range(6)
        ]
        faces = [list(range(5, -1, -1)), list(range(6, 12))]
        faces.extend([[i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6] for i in range(6)])
        head = UsdGeom.Mesh.Define(stage, f"{root}/{name}Head")
        head.CreatePointsAttr(vertices)
        head.CreateFaceVertexCountsAttr([len(face) for face in faces])
        head.CreateFaceVertexIndicesAttr([vertex for face in faces for vertex in face])
        head.CreateSubdivisionSchemeAttr("none")
        head.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(surface_z + 0.0014 + height / 2)))
        finish(head.GetPrim(), "FastenerSteel")
        geometry.append({
            "name": name, "shape": "hex_head_with_washer", "surface_world_m": [float(x), float(y), float(surface_z)],
            "head_radius_m": radius, "head_height_m": height, "washer_radius_m": 0.011, "washer_height_m": 0.0014,
        })

    left_x, right_x = float(base[0] - 0.205), float(rack[0] + 0.110)
    rear_x = float(rack[0] + 0.05)
    for side, sign in (("Left", -1), ("Right", 1)):
        y = float(rack[1] + sign * 0.30)
        cube(f"{side}LongitudinalBeam", ((left_x + right_x) / 2, y, deck_top - 0.03), (right_x - left_x, 0.06, 0.06))
        upright_x = float(rack[0] + RACK_SIZE_M[0] / 2 + 0.03)
        cube(f"{side}RackUpright", (upright_x, y, (deck_top + rack_top) / 2), (0.06, 0.06, rack_top - deck_top))
        cube(f"{side}RackFoot", (float(rack[0]), y, (deck_top + rack_bottom) / 2), (0.10, 0.12, rack_bottom - deck_top), "BrushedAluminium")
        bolt(f"{side}FrontFrameBolt", float(base[0]), y, deck_top)
        bolt(f"{side}RearFrameBolt", right_x - 0.016, y, deck_top)
    cube("RobotCrossMember", (float(base[0]), float(rack[1]), deck_top - 0.03), (0.10, 0.66, 0.06), "BrushedAluminium")
    cube("RackCrossMember", (rear_x, float(rack[1]), deck_top - 0.03), (0.12, 0.72, 0.06), "BrushedAluminium")
    anchor_bottom = float(base[2] - ANCHOR_SIZE_M[2] / 2)
    cube("RobotMountingPlate", (float(base[0]), float(base[1]), (deck_top + anchor_bottom) / 2), (0.26, 0.26, anchor_bottom - deck_top), "BrushedAluminium")
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            bolt(f"MountBolt{x_sign + 1}{y_sign + 1}", base[0] + x_sign * 0.105, base[1] + y_sign * 0.105, anchor_bottom)

    return {
        "kind": "fixed_service_structure_visualization",
        "prim_path": root,
        "scope": "Static visual context for the captured fixed robot and rack; existing physics and sensor observations are unchanged.",
        "physics_added": False,
        "collision_geometry_added": False,
        "animated_geometry_added": False,
        "is_rail_or_motion_stage": False,
        "max_captured_base_translation_m": max_displacement,
        "stationary_position_tolerance_m": STATIONARY_POSITION_TOLERANCE_M,
        "base_world_position_m": base.tolist(),
        "rack_world_position_m": rack.tolist(),
        "rack_size_m": list(RACK_SIZE_M),
        "anchor_size_m": list(ANCHOR_SIZE_M),
        "geometry": geometry,
    }
