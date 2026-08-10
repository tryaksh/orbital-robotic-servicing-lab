"""Measure where the Robotiq 2F-85 finger pads physically are.

Every 2F-85 rigid body in this asset is collapsed to within 18 mm of the wrist
flange, so ``body_pos_w`` says nothing about where a pad surface is.  This
project has already been misled once by reading those origins, and the
retraction is recorded in ``docs/status.md``.  Nothing may be designed against
guessed pad geometry again, so this script measures it.

Method, which never reads a body origin as a pad location:

1. Walk each gripper body's USD subtree and collect every prim carrying
   ``UsdPhysics.CollisionAPI``.  Only collision geometry can hold a blade;
   visual meshes are irrelevant and are excluded.
2. Express each collision prim's bound in *its own rigid body's* local frame.
   That offset is rigid, so it is valid at any joint angle.
3. Sweep ``finger_joint`` across its full 0 to 0.8203 rad range.  At each
   command, transform those local bounds out through the body pose PhysX
   reports and back into the ``wrist_3_link`` frame the tool offset is measured
   in.

The output is the envelope a grasp interface has to be designed against: which
wrist axis the pads close along, the clear opening between the pad faces at
every command, how far the pads reach past the flange, and how much pad surface
is available along the approach axis for a shaft to seat against.

The gripper is measured in free space with the blade moved out of the scene, so
this is kinematics.  It says what the gripper *can* enclose, not what it holds.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."

# The 2F-85 drive joint's own USD limits. Measured pad separation grows with the
# command, so zero is the closed end; see docs/status.md.
FINGER_JOINT_LOWER_RAD = 0.0
FINGER_JOINT_UPPER_RAD = 0.8203


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-Insertion-Contact-v0")
    parser.add_argument(
        "--commands",
        type=float,
        nargs="+",
        default=None,
        help="finger_joint targets in radians. Defaults to 12 samples across the joint's full range.",
    )
    parser.add_argument("--settle_s", type=float, default=1.0, help="Seconds held at each command before reading.")
    parser.add_argument("--point_stride", type=int, default=1, help="Keep every Nth collision-mesh vertex.")
    parser.add_argument(
        "--throat_y_half",
        type=float,
        nargs="+",
        default=[0.010, 0.015, 0.020],
        help="Half-widths on the third axis within which the throat profile looks for obstructions.",
    )
    parser.add_argument("--throat_slab_m", type=float, default=0.005, help="Depth of each throat-profile slice.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("artifacts/gripper_envelope.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.commands is None:
    step = (FINGER_JOINT_UPPER_RAD - FINGER_JOINT_LOWER_RAD) / 11.0
    args.commands = [round(FINGER_JOINT_LOWER_RAD + step * index, 6) for index in range(12)]
if any(value < FINGER_JOINT_LOWER_RAD or value > FINGER_JOINT_UPPER_RAD for value in args.commands):
    parser.error(f"--commands must lie within [{FINGER_JOINT_LOWER_RAD}, {FINGER_JOINT_UPPER_RAD}] rad")
if args.settle_s <= 0.0:
    parser.error("--settle_s must be positive")
if args.point_stride < 1:
    parser.error("--point_stride must be at least 1")
if any(value <= 0.0 for value in args.throat_y_half) or args.throat_slab_m <= 0.0:
    parser.error("--throat_y_half and --throat_slab_m must be positive")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
from isaaclab.utils.math import quat_apply, quat_inv
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.evaluation import round_floats
from zero_g_blade_swap.tasks.blade_swap.mdp.actions import (
    ROBOTIQ_2F85_COUPLING_SIGNS,
    ROBOTIQ_2F85_JOINT_NAMES,
)

# Everything the 2F-85 puts between the flange and the workpiece.
GRIPPER_BODY_PATTERNS = (
    r".*base_link_0$",
    r".*outer_knuckle$",
    r".*outer_finger$",
    r".*inner_knuckle$",
    r".*inner_finger$",
)
PAD_PATTERN = r".*inner_finger$"
AXIS_NAMES = ("x", "y", "z")


def _rigid_body_paths(robot, stage: Usd.Stage, root_path: str) -> dict[str, str]:
    """Map every articulation body name to the prim path PhysX resolved it to.

    Link prims are nested at varying depths inside the UR10e/Robotiq hierarchy
    and their names are not unique: the arm and the gripper both author a
    ``base_link``, which the articulation disambiguates as ``base_link_0``.
    PhysX's own link paths are therefore the only unambiguous mapping, and a
    name-matched stage walk is only a fallback for asset layouts it cannot
    report.
    """

    link_paths = getattr(robot.root_physx_view, "link_paths", None)
    if link_paths:
        return dict(zip(robot.body_names, (str(path) for path in link_paths[0]), strict=True))

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Robot prim is missing at '{root_path}'.")
    paths: dict[str, str] = {}
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            paths.setdefault(prim.GetName(), str(prim.GetPath()))
    return paths


def _local_collision_points(stage: Usd.Stage, body_path: str, stride: int) -> tuple[list[tuple[float, float, float]], bool]:
    """Return one rigid body's collision-surface points in its own frame.

    An axis-aligned box around a whole link is far too coarse to size a part
    that has to fit inside the gripper's throat: the inner knuckle sweeps a
    large box while being a thin bar. Mesh vertices describe the surface the
    convex collider is built from, so they answer where the metal actually is.

    Returns the points and whether any collider had to be approximated by its
    bounding box because it is an analytic shape rather than a mesh.
    """

    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"Gripper body prim is missing at '{body_path}'.")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=False,
    )
    to_body = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()

    points: list[tuple[float, float, float]] = []
    approximated = False
    for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() if prim.IsA(UsdGeom.Mesh) else None
        if mesh_points:
            to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for index in range(0, len(mesh_points), max(1, stride)):
                world = to_world.Transform(mesh_points[index])
                local = to_body.Transform(world)
                points.append((local[0], local[1], local[2]))
            continue
        # Analytic colliders (cube, cylinder, capsule) have no vertex list, so
        # fall back to their bounding box and say so in the report.
        world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if world_range.IsEmpty():
            continue
        approximated = True
        world_min, world_max = world_range.GetMin(), world_range.GetMax()
        for index in range(8):
            corner = (
                world_max[0] if index & 1 else world_min[0],
                world_max[1] if index & 2 else world_min[1],
                world_max[2] if index & 4 else world_min[2],
            )
            local = to_body.Transform(corner)
            points.append((local[0], local[1], local[2]))
    return points, approximated


def _configure(env_cfg) -> None:
    """Remove everything that would reset the scene mid-measurement."""

    env_cfg.configure_robustness(0)
    env_cfg.terminations.insertion_failed = None
    env_cfg.rewards.failure = None
    # The interval event would overwrite every environment's finger command.
    env_cfg.events.hold_gripper_closed = None
    env_cfg.episode_length_s = 1.0e6


def main() -> dict[str, object]:
    env = None
    try:
        device = args.device or "cuda:0"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"GPU run requested on {device}, but PyTorch cannot access CUDA")
        num_envs = len(args.commands)
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=num_envs)
        _configure(env_cfg)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        robot = task.scene["robot"]
        blade = task.scene["spare_blade"]

        joint_ids, joint_names = robot.find_joints(list(ROBOTIQ_2F85_JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != ROBOTIQ_2F85_JOINT_NAMES:
            raise RuntimeError(f"unexpected Robotiq joint order: {joint_names}")
        wrist_ids, _ = robot.find_bodies(["wrist_3_link"], preserve_order=True)
        body_ids, body_names = robot.find_bodies(list(GRIPPER_BODY_PATTERNS), preserve_order=False)
        pad_ids, pad_names = robot.find_bodies([PAD_PATTERN], preserve_order=True)
        if len(pad_ids) != 2:
            raise RuntimeError(f"expected two 2F-85 finger pads, resolved {pad_names}")

        root_paths = sorted(find_matching_prim_paths(robot.cfg.prim_path))
        if not root_paths:
            raise RuntimeError(f"no robot prims matched '{robot.cfg.prim_path}'")
        stage = get_current_stage()
        body_paths = _rigid_body_paths(robot, stage, root_paths[0])
        missing = [name for name in body_names if name not in body_paths]
        if missing:
            raise RuntimeError(
                f"no rigid-body prim found for {missing} under '{root_paths[0]}'; resolved {sorted(body_paths)}"
            )
        local_points: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        approximated: list[str] = []
        for name in body_names:
            points, was_approximated = _local_collision_points(stage, body_paths[name], args.point_stride)
            if not points:
                skipped.append(name)
                continue
            if was_approximated:
                approximated.append(name)
            local_points[name] = torch.tensor(points, dtype=torch.float32, device=task.device)
        if not all(name in local_points for name in pad_names):
            raise RuntimeError(f"the finger pads carry no collision geometry; resolved bodies {sorted(local_points)}")

        signs = torch.tensor(ROBOTIQ_2F85_COUPLING_SIGNS, device=task.device)
        commands = torch.tensor(args.commands, dtype=torch.float32, device=task.device)
        finger_targets = commands.unsqueeze(-1) * signs
        hold_still = torch.zeros((num_envs, task.action_manager.total_action_dim), device=task.device)
        settle_steps = max(1, int(round(args.settle_s / float(task.step_dt))))

        env.reset()
        with torch.inference_mode():
            # A blade inside the fingers would block them short of the command
            # and turn a kinematic envelope into a contact measurement.
            parked = blade.data.default_root_state.clone()
            parked[:, :3] = task.scene.env_origins + parked.new_tensor([0.0, 0.0, -5.0])
            blade.write_root_state_to_sim(parked)

            for _ in range(settle_steps):
                robot.set_joint_position_target(finger_targets, joint_ids=joint_ids)
                env.step(hold_still)

            reached = robot.data.joint_pos[:, joint_ids[0]].clone()
            drive_torque = robot.data.applied_torque[:, joint_ids[0]].abs().clone()
            wrist_position = robot.data.body_pos_w[:, wrist_ids[0]]
            wrist_orientation = robot.data.body_quat_w[:, wrist_ids[0]]
            wrist_inverse = quat_inv(wrist_orientation)

            body_id_by_name = dict(zip(body_names, body_ids, strict=True))
            envelope: dict[str, torch.Tensor] = {}
            for name, points in local_points.items():
                body_id = body_id_by_name[name]
                count = points.shape[0]
                position = robot.data.body_pos_w[:, body_id]
                orientation = robot.data.body_quat_w[:, body_id]
                world = position.unsqueeze(1) + quat_apply(
                    orientation.unsqueeze(1).expand(-1, count, -1), points.unsqueeze(0).expand(num_envs, -1, -1)
                )
                envelope[name] = quat_apply(
                    wrist_inverse.unsqueeze(1).expand(-1, count, -1),
                    world - wrist_position.unsqueeze(1),
                )

        bodies: dict[str, list[dict[str, object]]] = {}
        for name, local in envelope.items():
            bodies[name] = [
                {
                    "finger_joint_rad": round(float(commands[index]), 6),
                    "min_m": [round(float(value), 6) for value in local[index].amin(dim=0)],
                    "max_m": [round(float(value), 6) for value in local[index].amax(dim=0)],
                }
                for index in range(num_envs)
            ]

        # Which wrist axis do the pads separate along? Whichever one their
        # centres move apart on as the command opens them.
        left, right = (envelope[name] for name in pad_names)
        left_centre = 0.5 * (left.amin(dim=1) + left.amax(dim=1))
        right_centre = 0.5 * (right.amin(dim=1) + right.amax(dim=1))
        spread = (left_centre - right_centre).abs()
        travel = spread.amax(dim=0) - spread.amin(dim=0)
        closing_axis = int(torch.argmax(travel))
        approach_axis = 2  # every tool offset in this project is +z of the flange.
        third_axis = ({0, 1, 2} - {closing_axis, approach_axis}).pop()

        rows = []
        for index in range(num_envs):
            positive_is_left = bool(left_centre[index, closing_axis] > right_centre[index, closing_axis])
            near, far = (left, right) if positive_is_left else (right, left)
            # The clear opening is between the facing surfaces, not the centres.
            inner_positive = float(near[index, :, closing_axis].min())
            inner_negative = float(far[index, :, closing_axis].max())
            pad_low = torch.minimum(left[index].amin(dim=0), right[index].amin(dim=0))
            pad_high = torch.maximum(left[index].amax(dim=0), right[index].amax(dim=0))
            rows.append(
                {
                    "commanded_finger_joint_rad": round(float(commands[index]), 6),
                    "reached_finger_joint_rad": round(float(reached[index]), 6),
                    "drive_torque_nm": round(float(drive_torque[index]), 6),
                    "clear_opening_m": round(inner_positive - inner_negative, 6),
                    "pad_face_positive_m": round(inner_positive, 6),
                    "pad_face_negative_m": round(inner_negative, 6),
                    "pad_reach_along_approach_m": [round(float(pad_low[approach_axis]), 6),
                                                   round(float(pad_high[approach_axis]), 6)],
                    "pad_extent_along_approach_m": round(float(pad_high[approach_axis] - pad_low[approach_axis]), 6),
                    "pad_extent_along_third_axis_m": round(float(pad_high[third_axis] - pad_low[third_axis]), 6),
                }
            )

        # Across every body and every commanded opening, so this is the volume
        # the gripper can occupy anywhere in its travel.
        # How much room is there behind the pads? A head that provides form
        # closure has to sit between the pad trailing faces and the palm, so
        # what matters is not a body's bounding box but how close the knuckles
        # come to the tool axis at each depth. Slice the point cloud by depth
        # and report the nearest approach on the closing axis, for the two pads
        # and for everything else separately.
        others = [name for name in envelope if name not in pad_names]
        throat = []
        for index in range(num_envs):
            pad_cloud = torch.cat([envelope[name][index] for name in pad_names], dim=0)
            other_cloud = torch.cat([envelope[name][index] for name in others], dim=0)
            deepest = max(float(cloud[:, approach_axis].max()) for cloud in (pad_cloud, other_cloud))
            slices = []
            depth = 0.0
            while depth < deepest:
                upper_depth = depth + args.throat_slab_m
                entry: dict[str, object] = {"depth_from_flange_m": [round(depth, 6), round(upper_depth, 6)]}
                for label, cloud in (("pads", pad_cloud), ("other_bodies", other_cloud)):
                    at_depth = (cloud[:, approach_axis] >= depth) & (cloud[:, approach_axis] < upper_depth)
                    entry[f"nearest_{label}_m"] = {
                        f"{half:.3f}": (
                            round(float(cloud[inside, closing_axis].abs().min()), 6) if bool(inside.any()) else None
                        )
                        for half in args.throat_y_half
                        if (inside := at_depth & (cloud[:, third_axis].abs() <= half)) is not None
                    }
                slices.append(entry)
                depth = upper_depth
            throat.append({"finger_joint_rad": round(float(commands[index]), 6), "slices": slices})

        gripper_low = torch.stack([local.amin(dim=1) for local in envelope.values()]).amin(dim=0).amin(dim=0)
        gripper_high = torch.stack([local.amax(dim=1) for local in envelope.values()]).amax(dim=0).amax(dim=0)
        widest = max(rows, key=lambda row: row["clear_opening_m"])
        opening_span = widest["clear_opening_m"] - min(row["clear_opening_m"] for row in rows)
        command_span = max(args.commands) - min(args.commands)

        return {
            "status": "passed",
            "title": "Robotiq 2F-85 collision envelope in the wrist_3_link frame",
            "evidence_type": "simulation_kinematic_measurement",
            "protocol": {
                "task": args.task,
                "robustness_level": 0,
                "measurement": "USD collision-geometry bounds carried through the PhysX body pose",
                "frame": "wrist_3_link, the frame every tool offset in this project is expressed in",
                "blade": "parked 5 m below the scene so the fingers close unobstructed",
                "environments": num_envs,
                "settle_s": args.settle_s,
                "seed": args.seed,
                "bodies_without_collision_geometry": skipped,
                "bodies_approximated_by_bounding_box": approximated,
                "point_stride": args.point_stride,
                "resolved_body_prim_paths": {name: body_paths[name] for name in sorted(body_names)},
            },
            "axes": {
                "closing_axis": AXIS_NAMES[closing_axis],
                "approach_axis": AXIS_NAMES[approach_axis],
                "third_axis": AXIS_NAMES[third_axis],
                "note": (
                    "The closing axis is chosen by measurement: it is the wrist axis on which the two "
                    "pad centres move apart as the command opens. The approach axis is +z because every "
                    "tool offset in this project is a +z translation of the flange."
                ),
            },
            "by_command": rows,
            "throat_profile": {
                "third_axis_half_widths_m": list(args.throat_y_half),
                "note": (
                    "Nearest approach to the tool axis on the closing axis, by depth from the flange, "
                    "keyed by the third-axis half-width searched. A stepped head can only seat behind "
                    "the pads where 'other_bodies' leaves room, because the knuckles swing through that "
                    "volume as the fingers close."
                ),
                "sampling_caveat": (
                    "Distances come from collision-mesh vertices, so a reported obstruction is real but "
                    "a null slice is not proof of clearance: a coarse hull has no vertices between its "
                    "corners. Read positive detections, not absences."
                ),
                "by_command": throat,
            },
            "derived": {
                "widest_clear_opening_m": widest["clear_opening_m"],
                "widest_at_finger_joint_rad": widest["commanded_finger_joint_rad"],
                "narrowest_clear_opening_m": min(row["clear_opening_m"] for row in rows),
                "opening_per_radian_m": round(opening_span / command_span, 6) if command_span > 0.0 else None,
                "pad_leading_edge_from_flange_m": round(
                    max(row["pad_reach_along_approach_m"][1] for row in rows), 6
                ),
                "pad_trailing_edge_from_flange_m": round(
                    min(row["pad_reach_along_approach_m"][0] for row in rows), 6
                ),
                "gripper_envelope_min_m": [round(float(value), 6) for value in gripper_low],
                "gripper_envelope_max_m": [round(float(value), 6) for value in gripper_high],
            },
            "bodies": bodies,
            "scope_and_limitations": [
                "Kinematics only. This is the volume the gripper sweeps, not a grasp or a holding capacity.",
                "Collision geometry, which may be a convex hull larger than the visual pad surface.",
                "Measured with the fingers unobstructed; a workpiece between them stops them short.",
            ],
        }
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    report: dict[str, object]
    try:
        report = main()
    except BaseException as exc:
        traceback.print_exc()
        report = {"status": "failed", "task": args.task, "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {args.report}")
        simulation_app.close()
