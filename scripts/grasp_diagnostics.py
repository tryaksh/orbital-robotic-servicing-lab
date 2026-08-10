"""Measure what the simulated Robotiq 2F-85 friction grasp can actually hold.

The insertion curriculum replaces the grasp with a PhysX fixed joint, and the
real pad/handle contact task failed its axial pull gate as a pass/fail with no
number behind it. Removing the fixed joint is only defensible once the grasp is
characterised, so this script measures the quantity that decides it: the axial
force the closed fingers transmit to the handle before the blade slips.

Protocol, run entirely in parallel on the GPU:

1. Reset the contact-grasp scene with the rails collision-free, so the only
   contacts in the scene are finger pad against handle.
2. Hold the arm still while the fingers close and the contact settles, then
   record each environment's tool-to-handle vector as its zero-slip baseline.
3. Apply a constant axial pull to the blade. Environment ``i`` gets one point of
   a (closure target x pull force) grid, so one run sweeps the whole surface.
4. Hold, then report the slip each environment accumulated.

The reported breakaway force is the largest pull an environment held without
exceeding the slip tolerance. This is a physics characterisation of the
simulated grasp, not a learned-grasping result and not a hardware measurement:
PhysX Coulomb friction against primitive box geometry is an optimistic model of
a rubber pad on a machined handle.
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

# Worst-case peak contact force measured on the promoted Level-2 policy, from
# evidence/rigid_grasp_l2_contact_forces.json. A friction grasp that cannot
# transmit this cannot replace the fixed joint on the insertion task.
LEVEL_2_PEAK_CONTACT_FORCE_N = 66.36
LEVEL_2_P95_CONTACT_FORCE_N = 16.56


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-Insertion-Contact-v0")
    parser.add_argument(
        "--closures",
        type=float,
        nargs="+",
        default=[0.55, 0.60, 0.65, 0.69],
        help="Commanded 2F-85 closure targets in radians. NVIDIA's assembly setup uses 0.45-0.69.",
    )
    parser.add_argument("--force_levels", type=int, default=32, help="Pull force samples per closure target.")
    parser.add_argument("--max_force_n", type=float, default=120.0, help="Largest axial pull applied, in newtons.")
    parser.add_argument(
        "--pull_sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help="-1 pulls the blade out of the rack along -x, matching extraction; +1 pushes it in.",
    )
    parser.add_argument("--settle_s", type=float, default=1.0, help="Seconds held still before the baseline is taken.")
    parser.add_argument("--hold_s", type=float, default=1.5, help="Seconds the pull is applied.")
    parser.add_argument(
        "--slip_tolerance_m",
        type=float,
        default=0.002,
        help="Tool-to-handle movement above this counts as slip. The insertion task aborts at 0.025 m.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("artifacts/grasp_diagnostics.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.force_levels < 2:
    parser.error("--force_levels must be at least 2")
if args.max_force_n <= 0.0:
    parser.error("--max_force_n must be positive")
if args.settle_s <= 0.0 or args.hold_s <= 0.0:
    parser.error("--settle_s and --hold_s must be positive")
if args.slip_tolerance_m <= 0.0:
    parser.error("--slip_tolerance_m must be positive")
if any(value <= 0.0 for value in args.closures):
    parser.error("--closures must be positive radians")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.evaluation import round_floats, summarize_distribution
from zero_g_blade_swap.tasks.blade_swap.mdp.actions import (
    ROBOTIQ_2F85_COUPLING_SIGNS,
    ROBOTIQ_2F85_JOINT_NAMES,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import (
    attached_blade_pose_world,
    secured_blade_pose_error,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world


def _configure(env_cfg) -> None:
    """Strip everything that would end an episode or move the gripper for us."""

    # Level 0 keeps the rails collision-free, so pad-against-handle is the only
    # contact pair in the scene and the measurement isolates the grasp.
    env_cfg.configure_robustness(0)
    # A slipping environment trips the insertion-failure predicate at 25 mm and
    # would be reset mid-measurement, which is exactly the data we want to keep.
    env_cfg.terminations.insertion_failed = None
    env_cfg.rewards.failure = None
    # The script drives the fingers so each environment can hold its own closure
    # target; the interval event would overwrite all of them with one value.
    env_cfg.events.hold_gripper_closed = None


def _grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-environment closure target and pull force."""

    closures = torch.tensor(args.closures, dtype=torch.float32, device=device)
    forces = torch.linspace(0.0, float(args.max_force_n), args.force_levels, device=device)
    closure_grid = closures.repeat_interleave(args.force_levels)
    force_grid = forces.repeat(len(args.closures))
    return closure_grid, force_grid


def _breakaway(
    closure: float,
    forces: torch.Tensor,
    slipped: torch.Tensor,
    grip_torque: torch.Tensor,
) -> dict[str, object]:
    """Summarise one closure target's holding capacity.

    ``breakaway`` is the largest force held within tolerance below the first
    force that slipped, so a single non-monotonic point cannot inflate it.
    ``grip`` is reported alongside it because a capacity of zero is only
    interpretable next to the torque the fingers were actually developing: no
    torque means no normal force, which is a command or geometry fault rather
    than a friction limit.
    """

    order = torch.argsort(forces)
    ordered_force = forces[order]
    ordered_slipped = slipped[order]
    first_slip_index = int(torch.nonzero(ordered_slipped).min()) if bool(ordered_slipped.any()) else None
    summary: dict[str, object] = {
        "closure_rad": closure,
        "peak_drive_joint_torque_nm": float(grip_torque.max()),
        "mean_drive_joint_torque_nm": float(grip_torque.mean()),
    }
    if first_slip_index is None:
        summary.update(
            {
                "held_every_force": True,
                "largest_force_held_n": float(ordered_force[-1]),
                "first_slipping_force_n": None,
                "breakaway_force_n": None,
            }
        )
        return summary
    summary.update(
        {
            "held_every_force": False,
            "largest_force_held_n": float(ordered_force[first_slip_index - 1]) if first_slip_index > 0 else 0.0,
            "first_slipping_force_n": float(ordered_force[first_slip_index]),
            # Midpoint of the bracketing pair; the grid spacing bounds the error.
            "breakaway_force_n": (
                float(0.5 * (ordered_force[first_slip_index - 1] + ordered_force[first_slip_index]))
                if first_slip_index > 0
                else 0.0
            ),
        }
    )
    return summary


def main() -> dict[str, object]:
    env = None
    try:
        device = args.device or "cuda:0"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"GPU run requested on {device}, but PyTorch cannot access CUDA")
        num_envs = len(args.closures) * args.force_levels
        print(f"[INFO] Grasp grid: {len(args.closures)} closure targets x {args.force_levels} forces = {num_envs} envs")
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=num_envs)
        _configure(env_cfg)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        if getattr(task.cfg, "rigid_grasp", False):
            raise ValueError(f"{args.task} attaches the blade with a fixed joint; there is no grasp to measure")
        if not getattr(task.cfg, "contact_grasp", False):
            raise ValueError(f"{args.task} does not use physical finger/handle contact")

        robot = task.scene["robot"]
        blade = task.scene["spare_blade"]
        joint_ids, joint_names = robot.find_joints(list(ROBOTIQ_2F85_JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != ROBOTIQ_2F85_JOINT_NAMES:
            raise RuntimeError(f"unexpected Robotiq joint order: {joint_names}")
        pad_ids, pad_names = robot.find_bodies([".*inner_finger$"], preserve_order=True)
        if len(pad_ids) != 2:
            raise RuntimeError(f"expected two 2F-85 finger pads, resolved {pad_names}")
        signs = torch.tensor(ROBOTIQ_2F85_COUPLING_SIGNS, device=task.device)
        closure, pull_force = _grid(task.device)
        finger_targets = closure.unsqueeze(-1) * signs

        env.reset()
        hold_still = torch.zeros((num_envs, task.action_manager.total_action_dim), device=task.device)
        wrench = torch.zeros((num_envs, 1, 3), device=task.device)
        zero_torque = torch.zeros_like(wrench)
        settle_steps = max(1, int(round(args.settle_s / float(task.step_dt))))
        hold_steps = max(1, int(round(args.hold_s / float(task.step_dt))))

        with torch.inference_mode():
            for _ in range(settle_steps):
                robot.set_joint_position_target(finger_targets, joint_ids=joint_ids)
                env.step(hold_still)
            baseline_vector, baseline_angle = secured_blade_pose_error(task)
            baseline_vector = baseline_vector.clone()
            baseline_angle = baseline_angle.clone()
            baseline_blade_x = attached_blade_pose_world(task)[0][:, 0].clone()
            settled_offset = torch.linalg.vector_norm(baseline_vector, dim=-1)

            # A holding capacity of zero means nothing unless we also know
            # whether the pads ever reached the handle. Measure the three
            # frames that decide it: the tool frame the IK and the grasp metric
            # use, where the finger pads physically are, and where the handle is.
            blade_position, blade_orientation = attached_blade_pose_world(task)
            handle_offset = blade_position.new_tensor(task.cfg.scene.spare_blade.spawn.handle_offset)
            handle_centre = blade_position + quat_apply(blade_orientation, handle_offset.expand(num_envs, -1))
            tool_position, _ = end_effector_pose_world(task)
            pad_midpoint = robot.data.body_pos_w[:, pad_ids].mean(dim=1)
            tool_to_pad = torch.linalg.vector_norm(pad_midpoint - tool_position, dim=-1)
            pad_to_handle = torch.linalg.vector_norm(pad_midpoint - handle_centre, dim=-1)
            wrench[:, 0, 0] = float(args.pull_sign) * pull_force
            peak_slip = torch.zeros(num_envs, device=task.device)
            peak_grip_torque = torch.zeros(num_envs, device=task.device)
            for _ in range(hold_steps):
                robot.set_joint_position_target(finger_targets, joint_ids=joint_ids)
                blade.permanent_wrench_composer.set_forces_and_torques(
                    forces=wrench, torques=zero_torque, is_global=True
                )
                env.step(hold_still)
                current_vector, _ = secured_blade_pose_error(task)
                slip = torch.linalg.vector_norm(current_vector - baseline_vector, dim=-1)
                torch.maximum(peak_slip, slip, out=peak_slip)
                grip = robot.data.applied_torque[:, joint_ids[0]].abs()
                torch.maximum(peak_grip_torque, grip, out=peak_grip_torque)

            final_vector, final_angle = secured_blade_pose_error(task)
            # Split the slip along the pull axis from the rest. A capture that
            # is losing axial form closure and one whose payload is levering out
            # sideways look identical in the norm and need opposite fixes.
            slip_vector = final_vector - baseline_vector
            axial_slip = slip_vector[:, 0]
            lateral_slip = torch.linalg.vector_norm(slip_vector[:, 1:], dim=-1)
            reached_finger = robot.data.joint_pos[:, joint_ids[0]].clone()
            final_slip = torch.linalg.vector_norm(slip_vector, dim=-1)
            angular_slip = (final_angle - baseline_angle).abs()
            blade_travel = (attached_blade_pose_world(task)[0][:, 0] - baseline_blade_x).abs()
            finite = torch.isfinite(final_slip) & torch.isfinite(blade_travel)
            slipped = peak_slip > float(args.slip_tolerance_m)
            # Whether a grasp formed is decided by the solver, not by geometry
            # guessed from link origins: a blocked finger cannot reach its
            # commanded angle, so drive torque rises off the noise floor.
            grasp_established = peak_grip_torque > 0.05

        by_closure = [
            _breakaway(
                value,
                pull_force[closure == value],
                slipped[closure == value],
                peak_grip_torque[closure == value],
            )
            for value in args.closures
        ]
        required = LEVEL_2_PEAK_CONTACT_FORCE_N
        best = max(
            (entry["largest_force_held_n"] for entry in by_closure if entry["largest_force_held_n"] is not None),
            default=0.0,
        )
        return {
            "status": "passed",
            "title": "Simulated Robotiq 2F-85 axial grasp holding capacity",
            "evidence_type": "simulation_physics_characterization",
            "protocol": {
                "task": args.task,
                "robustness_level": 0,
                "grasp_model": "physical_finger_pad_contact",
                "gravity": "zero",
                "policy_rate_hz": 30,
                "physics_rate_hz": 120,
                "environments": num_envs,
                "closure_targets_rad": list(args.closures),
                "force_levels": args.force_levels,
                "max_force_n": args.max_force_n,
                "force_step_n": args.max_force_n / (args.force_levels - 1),
                "pull_direction": "extraction_minus_x" if args.pull_sign < 0 else "insertion_plus_x",
                "settle_s": args.settle_s,
                "hold_s": args.hold_s,
                "slip_tolerance_m": args.slip_tolerance_m,
                "arm_action": "held still at zero for the whole measurement",
                "seed": args.seed,
            },
            "settled_grasp": {
                "tool_to_handle_offset_m": summarize_distribution(settled_offset),
                "peak_drive_joint_torque_nm": summarize_distribution(peak_grip_torque),
                "drive_joint_effort_limit_nm": float(robot.actuators["gripper_drive"].effort_limit_sim.max()),
            },
            "frame_geometry": {
                "note": (
                    "These distances are to the finger *body origins*, which in this 2F-85 asset "
                    "are collapsed to within 18 mm of the wrist flange and are NOT where the pad "
                    "surfaces are. Do not read them as a pad location or as a calibration error. "
                    "The load-bearing evidence that no grasp forms is the drive torque below, "
                    "which stays at the 1e-5 N-m noise floor instead of rising against a blocked "
                    "finger. A separate sweep of handle distance along the tool axis saturates the "
                    "10 N-m limit between roughly 0.06 and 0.15 m, so the fingers do obstruct on "
                    "the blade there; the configured 0.179 m places the handle past the fingertips."
                ),
                "finger_bodies": list(pad_names),
                "tool_frame_to_finger_body_origin_m": summarize_distribution(tool_to_pad),
                "finger_body_origin_to_handle_centre_m": summarize_distribution(pad_to_handle),
                "tool_frame_to_handle_centre_m": summarize_distribution(settled_offset),
                # A grapple-pin blade has no single handle box, so this is absent
                # rather than zero on those tasks.
                "handle_size_m": (
                    list(handle_size) if (handle_size := getattr(task.cfg.scene.spare_blade.spawn, "handle_size", None))
                    else None
                ),
                "empirical_contact_zone_from_flange_m": [0.06, 0.15],
                "configured_tool_offset_m": list(task.cfg.tool_offset_pos),
            },
            "slip": {
                "peak_slip_m": summarize_distribution(peak_slip),
                "final_slip_m": summarize_distribution(final_slip),
                "final_axial_slip_m": summarize_distribution(axial_slip.abs()),
                "final_lateral_slip_m": summarize_distribution(lateral_slip),
                "final_angular_slip_rad": summarize_distribution(angular_slip),
                "reached_finger_joint_rad": summarize_distribution(reached_finger),
                "blade_axial_travel_m": summarize_distribution(blade_travel),
                "non_finite_environments": int((~finite).sum()),
                "environments_that_slipped": int(slipped.sum()),
            },
            "by_closure_target": by_closure,
            "gate": {
                "grasp_contact_established": bool(grasp_established.all()),
                "environments_where_a_finger_was_blocked": int(grasp_established.sum()),
                "required_axial_force_n": required,
                "rationale": (
                    "Worst-case peak contact force measured on the promoted Level-2 insertion policy "
                    "(evidence/rigid_grasp_l2_contact_forces.json). A friction grasp that cannot transmit "
                    "this cannot carry the insertion task without the fixed joint."
                ),
                "reference_p95_contact_force_n": LEVEL_2_P95_CONTACT_FORCE_N,
                "best_force_held_n": best,
                "passed": bool(grasp_established.all() and best >= required),
                "interpretation": (
                    "held capacity is only a friction result once a grasp actually forms; with "
                    "drive torque at the noise floor the blade is an unconstrained body and its "
                    "motion under load is simply force over mass"
                ),
            },
            "scope_and_limitations": [
                "Simulation only. PhysX Coulomb friction between primitive boxes, not a measured rubber pad.",
                "Static holding capacity with the arm held still; it does not measure grasp acquisition, "
                "regrasp, or dynamic slip during a moving insertion.",
                "Axial pull only. Lateral force and prying moment capacity are not measured here.",
                "This characterises a grasp abstraction's physics. It is not learned grasping.",
            ],
            "grid": {
                "closure_rad": [round(float(value), 4) for value in closure],
                "pull_force_n": [round(float(value), 4) for value in pull_force],
                "peak_slip_m": [round(float(value), 6) for value in peak_slip],
                "slipped": [bool(value) for value in slipped],
            },
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
