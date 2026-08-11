"""Run capture, extraction, transit, and re-insertion in one episode.

Three separately trained checkpoints drive one continuous episode on one module.
The driver switches between them on **measured conditions**, never on a timer:
the capture hands over when the drive torque says the pads are loaded on the pin,
the pull hands over when the module's rear face is clear of the rack mouth, and
the transit hands over when the module reaches the pose the insert policy was
trained from.

What is learned and what is not, stated plainly, because a demonstration that
blurs this is worthless:

* capture, extraction and insertion are trained policies, run deterministically
  from their checkpoints;
* the transit between "clear of the rack" and "lined up to go back in" is
  **scripted**, because there is no contact in it and nothing for a policy to
  learn. It retraces the path the extraction actually took, in reverse. A blind
  axial command does not work and the reason is worth recording: the extracted
  pose leaves the wrist about 200 mm in front of the robot's own base, folded,
  and driving straight back out from there takes the damped-least-squares IK
  through a near-singularity. Measured, it swings the shoulder 74 degrees, drives
  the elbow into its limit, and levers the module out of the pads. Retracing a
  path the arm has already flown is feasible by construction;
* the module is held by real pad-against-pin contact throughout. There is no
  fixed joint and no software fixture in this scene.

The policies are loaded straight from their checkpoints rather than through
RL-Games, because three players in one process would each need their own vector
environment. The network is a three-layer MLP and its observation normaliser is
in the same file, so running it directly is both simpler and easier to audit.
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

TASK = "Isaac-ZeroG-Blade-GrapplePin-Workflow-v0"
#: Control steps between recorded transit waypoints. Four is about 30 mm of pull
#: at the extraction scale, close enough that the return follows the same arc.
TRANSIT_WAYPOINT_STRIDE = 4
#: Control steps spent letting the closure drive the pin against its collar
#: before the pull starts. One second, which is what the pull gate needed to
#: settle and what the extract task's own action term waits out.
SEAT_STEPS = 30
#: Grip error the capture must reach before extraction takes over. The seating
#: feed adds about 3 mm on top, landing near the 12.4 mm the extract task starts
#: its own episodes from.
HANDOVER_GRIP_M = 0.010


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp_checkpoint", type=Path, required=True)
    parser.add_argument("--extract_checkpoint", type=Path, required=True)
    parser.add_argument("--insert_checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4070)
    parser.add_argument(
        "--curriculum_stage",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Reset distance for the capture. 2 is the widest pose error the grasp policy trained on.",
    )
    parser.add_argument(
        "--workflow",
        choices=("remove", "install", "full"),
        default="install",
        help=(
            "remove: capture a fully installed module and pull it clear of the rack, both learned. "
            "install: capture a module at the rack mouth and seat it, both learned. "
            "full: remove, fly back, and re-install. The return leg is blocked by the pin's yaw limitation, "
            "not by the controller: a single-point pin does not constrain rotation once the rails release the "
            "module, and the grip degrades from 15 mm to 35 mm during the return whatever speed it is flown at."
        ),
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--inspection_view",
        choices=("task", "grasp", "side", "top", "workcell"),
        default="side",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_dir", type=Path, default=Path("artifacts/demo/workflow"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/demo/workflow_report.json"))
    parser.add_argument(
        "--transit_slowdown",
        type=int,
        default=3,
        help=(
            "Replay the return path this many times slower than the pull. A single-point pin does not constrain "
            "yaw once the rails release the module, so a full-speed replay rotates it in the pads."
        ),
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=0,
        help="Extra steps to hold still after the workflow finishes, so a recording does not cut on the last frame.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import nn

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.evaluation import round_floats
from zero_g_blade_swap.tasks.blade_swap.mdp.grapple import (
    capture_established,
    grapple_grip_error_metrics,
    grapple_insertion_conditions,
    grip_drive_torque,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import insertion_error_metrics
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world
from zero_g_blade_swap.tasks.blade_swap.workflow_demo_env_cfg import (
    EXTRACT_ACTION_SCALE,
    GRASP_ACTION_SCALE,
    INSERT_ACTION_SCALE,
    TRANSIT_TARGET_BLADE_X,
)
from zero_g_blade_swap.grapple_geometry import EXTRACTED_BLADE_CENTRE_X


class CheckpointPolicy:
    """One RL-Games actor, loaded without RL-Games.

    Reproduces exactly what a deterministic player does: clip the observation to
    the configured range, apply the running mean/variance normaliser the policy
    was trained with, run the trunk, and take the mean action.
    """

    def __init__(self, path: Path, device: str, clip_observations: float = 10.0) -> None:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        weights = checkpoint["model"]
        self.device = device
        self.clip_observations = clip_observations
        self.mean = weights["running_mean_std.running_mean"].to(device).float()
        self.variance = weights["running_mean_std.running_var"].to(device).float()
        self.observation_dim = int(self.mean.shape[0])

        layers: list[nn.Module] = []
        index = 0
        while f"a2c_network.trunk.{index}.weight" in weights:
            weight = weights[f"a2c_network.trunk.{index}.weight"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(weights[f"a2c_network.trunk.{index}.bias"])
            layers.extend((layer, nn.ELU()))
            index += 2
        self.trunk = nn.Sequential(*layers).to(device).eval()

        mu_weight = weights["a2c_network.mu.weight"]
        self.mu = nn.Linear(mu_weight.shape[1], mu_weight.shape[0]).to(device)
        self.mu.weight.data.copy_(mu_weight)
        self.mu.bias.data.copy_(weights["a2c_network.mu.bias"])
        self.mu.eval()
        self.action_dim = int(mu_weight.shape[0])
        self.epoch = int(checkpoint.get("epoch", -1))
        self.path = path

    @torch.inference_mode()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise RuntimeError(
                f"{self.path.name} expects {self.observation_dim} observation values, "
                f"received {observation.shape[-1]}. The observation group does not match the policy."
            )
        clipped = observation.clamp(-self.clip_observations, self.clip_observations)
        normalized = ((clipped - self.mean) / torch.sqrt(self.variance + 1.0e-5)).clamp(-5.0, 5.0)
        return self.mu(self.trunk(normalized)).clamp(-1.0, 1.0)


def main() -> dict[str, object]:
    env = None
    try:
        device = args.device or "cuda:0"
        env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
        env_cfg.configure_robustness(0)
        env_cfg.seed = args.seed
        inspection_views = {
            "grasp": ((0.18, -1.05, 1.02), (0.52, 0.0, 0.72)),
            "side": ((0.52, -1.30, 0.86), (0.50, 0.0, 0.72)),
            "top": ((0.50, -0.05, 1.60), (0.50, 0.0, 0.70)),
            "workcell": ((-0.50, -1.80, 1.25), (0.45, 0.0, 0.72)),
        }
        if args.inspection_view in inspection_views:
            env_cfg.viewer.eye, env_cfg.viewer.lookat = inspection_views[args.inspection_view]

        policies = {
            "capture": CheckpointPolicy(args.grasp_checkpoint, device),
            "extract": CheckpointPolicy(args.extract_checkpoint, device),
            "insert": CheckpointPolicy(args.insert_checkpoint, device),
        }
        for name, policy in policies.items():
            print(f"[INFO] {name:8s} obs={policy.observation_dim:3d} act={policy.action_dim} "
                  f"epoch={policy.epoch} <- {policy.path.name}", flush=True)

        env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        task = env.unwrapped
        # The reset event picks the arm and blade pose from this buffer, so it has
        # to exist and hold the wanted stage *before* the first reset. Nothing has
        # created it yet: with no curriculum term, only the reset itself would,
        # and by then it would have already chosen stage 0.
        task._insertion_curriculum_stage = torch.full(
            (task.num_envs,), args.curriculum_stage, dtype=torch.long, device=task.device
        )
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(args.video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args.steps,
                disable_logger=True,
            )
        env.reset()
        task._insertion_curriculum_stage.fill_(args.curriculum_stage)

        arm = task.action_manager.get_term("arm")
        scales = {
            "capture": GRASP_ACTION_SCALE,
            "extract": EXTRACT_ACTION_SCALE,
            "transit": EXTRACT_ACTION_SCALE,
            "insert": INSERT_ACTION_SCALE,
        }

        def set_phase(name: str) -> None:
            arm._scale[:] = torch.tensor(scales[name], device=task.device)

        phase = "capture"
        set_phase(phase)
        held = 0
        seat_until = 0
        seated_at = 0
        seated = False
        transit_started = 0
        # Tool positions visited during the pull, sampled so the transit can fly
        # them backwards. Every one of them was reachable a moment ago.
        extraction_path: list[torch.Tensor] = []
        waypoint = -1
        required_hold = max(1, int(round(0.30 / float(task.step_dt))))
        timeline: list[dict[str, object]] = []
        actions = torch.zeros((1, task.action_manager.total_action_dim), device=task.device)

        def note(event: str, step: int) -> None:
            blade_x = float(task.scene["spare_blade"].data.root_pos_w[0, 0] - task.scene.env_origins[0, 0])
            grip, attitude = grapple_grip_error_metrics(task)
            entry = {
                "event": event,
                "step": step,
                "time_s": round(step * float(task.step_dt), 3),
                "blade_centre_x_m": blade_x,
                "grip_error_m": float(grip[0]),
                "grip_attitude_rad": float(attitude[0]),
                "drive_torque_nm": float(grip_drive_torque(task)[0]),
            }
            timeline.append(entry)
            print(f"[PHASE] {event:22s} step {step:4d}  t={entry['time_s']:6.2f}s  "
                  f"blade_x={blade_x:.4f}  grip={entry['grip_error_m'] * 1000:6.2f}mm  "
                  f"torque={entry['drive_torque_nm']:5.2f}Nm", flush=True)

        note("start:capture", 0)
        for step in range(args.steps):
            observations = task.observation_manager.compute()
            if phase == "capture":
                command = policies["capture"].act(observations["grasp"])
                actions[:] = command
                # Hand over on the *next* skill's precondition, not this one's
                # success criterion. The grasp task counts a capture from 20 mm
                # of grip error, and the extract policy has never started from
                # worse than about 12 mm, so handing over at the first qualifying
                # instant puts it 10 mm out of distribution and it reverses into
                # the rack. The grasp policy keeps closing to a 9-to-12 mm median
                # if simply allowed to finish.
                grip_error = float(grapple_grip_error_metrics(task)[0][0])
                if bool(capture_established(task)[0]) and grip_error <= HANDOVER_GRIP_M:
                    held += 1
                    if held >= required_hold:
                        # Latch the holding closure. TwoStageRobotiqAction drops
                        # back to the gentler capture command whenever the grip
                        # error exceeds its tolerance, which is right while
                        # capturing and catastrophic afterwards: measured, the
                        # error drifts past 20 mm during the pull, the fingers
                        # open by about 21 mm, and the module is released in
                        # mid-transit. A real servicer does not relax its grip
                        # once the part is taken.
                        gripper = task.action_manager.get_term("gripper")
                        gripper.cfg.closed_position = gripper.cfg.hold_position
                        phase = "seat"
                        seat_until = step + SEAT_STEPS
                        note("capture -> seat", step)
                else:
                    held = 0
            elif phase == "seat":
                # Hold the arm still and let the closure drive the pin against
                # the collar. The extract skill gets this for free: its task
                # resets with the fingers apart and runs the two-stage capture
                # inside a 1.0 s window while the action term holds the arm, so
                # it only ever sees a *seated* grip. The grasp policy declares
                # capture as soon as the pads are loaded, which is measurably
                # earlier -- finger angle 0.085 against the 0.223 the pin sits at
                # once it is home. Handing that shallow grip to extraction lets
                # the wedge cam the module round, and the attitude error grows
                # from 0.03 to 0.40 rad until the policy is out of distribution
                # and reverses into the rack.
                actions[:] = 0.0
                actions[:, 6] = 1.0
                if step >= seat_until:
                    phase = "insert" if args.workflow == "install" else "extract"
                    set_phase(phase)
                    note(f"seat -> {phase}", step)
            elif phase == "extract":
                actions[:, :6] = policies["extract"].act(observations["extract"])
                # Keep commanding closure so the two-stage action holds the pin.
                actions[:, 6] = 1.0
                if step % TRANSIT_WAYPOINT_STRIDE == 0:
                    extraction_path.append(end_effector_pose_world(task)[0][0].clone())
                if step % 40 == 0:
                    grip_e, grip_a = grapple_grip_error_metrics(task)
                    print(
                        f"[EXTRACT] step {step:4d} blade_x="
                        f"{float(task.scene['spare_blade'].data.root_pos_w[0, 0] - task.scene.env_origins[0, 0]):.4f} "
                        f"a0={float(actions[0, 0]):+.3f} grip={float(grip_e[0]) * 1000:6.2f}mm "
                        f"att={float(grip_a[0]):.3f} torque={float(grip_drive_torque(task)[0]):5.2f} "
                        f"finger={float(task.scene['robot'].data.joint_pos[0, task._grapple_finger_joint_ids[0]]):.3f}",
                        flush=True,
                    )
                blade_x = float(task.scene["spare_blade"].data.root_pos_w[0, 0] - task.scene.env_origins[0, 0])
                if blade_x <= EXTRACTED_BLADE_CENTRE_X:
                    if args.workflow == "remove":
                        note("extract: module clear of the rack", step)
                        seated_at = step
                        phase = "done"
                    else:
                        phase = "transit"
                        set_phase(phase)
                        waypoint = len(extraction_path) - 1
                        transit_started = step
                        note("extract -> transit (scripted)", step)
            elif phase == "transit":
                # Scripted, and the only segment no policy drives: fly the tool
                # back along the waypoints the extraction just visited. Closing
                # the loop on position each step, so a waypoint that is slightly
                # off does not accumulate.
                actions[:] = 0.0
                actions[:, 6] = 1.0
                tool = end_effector_pose_world(task)[0][0]
                # Walk the recorded path backwards on the clock, not on proximity.
                # Advancing only when close stalls: the last waypoint was sampled
                # up to a stride before the hand-off, so the tool is already past
                # it and the follower sits there driving the module further out.
                # Replaying at the stride it was recorded at flies the same arc
                # at the same speed, which the arm has just demonstrated it can.
                if (step - transit_started) % (TRANSIT_WAYPOINT_STRIDE * args.transit_slowdown) == 0:
                    waypoint = max(waypoint - 1, 0)
                target = extraction_path[waypoint]
                scale = torch.tensor(scales["transit"][:3], device=task.device)
                actions[0, :3] = ((target - tool) / scale).clamp(-1.0, 1.0)
                blade_x = float(task.scene["spare_blade"].data.root_pos_w[0, 0] - task.scene.env_origins[0, 0])
                if step % 40 == 0:
                    print(
                        f"[TRANSIT] step {step:4d} wp={waypoint:3d}/{len(extraction_path)} "
                        f"blade_x={blade_x:.4f} tool_x={float(tool[0]):.4f} "
                        f"target_x={float(target[0]):.4f} "
                        f"grip={float(grapple_grip_error_metrics(task)[0][0]) * 1000:6.2f}mm "
                        f"torque={float(grip_drive_torque(task)[0]):5.2f}",
                        flush=True,
                    )
                if waypoint <= 0 and blade_x >= TRANSIT_TARGET_BLADE_X - 0.005:
                    phase = "insert"
                    set_phase(phase)
                    note("transit -> insert", step)
            elif phase == "insert":
                actions[:, :6] = policies["insert"].act(observations["insert"])
                actions[:, 6] = 1.0
                conditions = grapple_insertion_conditions(task)
                if all(bool(value[0]) for value in conditions.values()):
                    note("insert: seated", step)
                    seated_at = step
                    seated = True
                    phase = "done"
            else:
                # Hold position so a recording ends on the seated module rather
                # than mid-motion.
                actions[:] = 0.0
                actions[:, 6] = 1.0
                if step >= seated_at + args.settle_steps:
                    break

            _, _, terminated, truncated, _ = env.step(actions)
            if bool(terminated[0] or truncated[0]):
                note(f"episode ended during {phase}", step)
                break

        axial, lateral, orientation = insertion_error_metrics(task)
        grip, attitude = grapple_grip_error_metrics(task)
        conditions = grapple_insertion_conditions(task)
        result = {
            "task": TASK,
            "seed": args.seed,
            "curriculum_stage": args.curriculum_stage,
            "reached_phase": phase,
            "checkpoints": {name: str(policy.path) for name, policy in policies.items()},
            "learned_phases": {
                "remove": ["capture", "extract"],
                "install": ["capture", "insert"],
                "full": ["capture", "extract", "insert"],
            }[args.workflow],
            "scripted_phases": ["seat"] + (["transit"] if args.workflow == "full" else []),
            "timeline": timeline,
            "final": {
                "axial_error_m": float(axial[0]),
                "lateral_error_m": float(lateral[0]),
                "orientation_error_rad": float(orientation[0]),
                "grip_error_m": float(grip[0]),
                "grip_attitude_rad": float(attitude[0]),
            },
            "insertion_conditions": {name: bool(value[0]) for name, value in conditions.items()},
        }
        result["workflow"] = args.workflow
        if args.workflow == "remove":
            blade_x = float(task.scene["spare_blade"].data.root_pos_w[0, 0] - task.scene.env_origins[0, 0])
            result["completed"] = bool(phase == "done" and blade_x <= EXTRACTED_BLADE_CENTRE_X)
            result["final"]["blade_centre_x_m"] = blade_x
        else:
            # The task's own success predicate firing is what completion means.
            # Re-checking every condition after the settle steps is stricter than
            # the skill itself: the module stays seated, but the pin relaxes a
            # couple of hundredths of a radian in the pads afterwards and trips
            # the grip-retention check. The final conditions are reported in full
            # beside this so the distinction is visible rather than hidden.
            result["completed"] = bool(seated)
            result["conditions_still_held_after_settling"] = all(result["insertion_conditions"].values())
        return result
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    report: dict[str, object]
    try:
        report = main()
    except BaseException as exc:
        traceback.print_exc()
        report = {"task": TASK, "error": f"{type(exc).__name__}: {exc}", "completed": False}
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(round_floats(report), indent=2)[:2000], flush=True)
        simulation_app.close()
