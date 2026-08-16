"""Train the zero-g blade-swap task with RL-Games PPO."""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

# Tasks that expose cumulative robustness profiles.
ROBUST_FAMILY_TASKS = (
    "Insertion-Robust",
    "Insertion-Contact",
    "Insertion-RigidGrasp",
    "Insertion-ForceLimited",
    "Insertion-StrictForceLimited",
    "Insertion-ForceFeedback",
    "Insertion-Uncertain",
    "Insertion-UncertainBlind",
    "Insertion-Vision",
    "Insertion-GuidedSlot",
    "Blade-CaptureInSlot",
    "Blade-GrapplePin",
)

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-Insertion-v0")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument(
        "--robustness_level",
        type=int,
        choices=range(5),
        default=None,
        help="Insertion profile: 0=tight/6D, 1=pose, 2=mass, 3=friction, 4=mount wobble.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume an RL-Games .pth checkpoint.")
    parser.add_argument("--bc_checkpoint", type=Path, default=None, help="Initialize the vision actor from BC.")
    parser.add_argument("--smoke", action="store_true", help="Run two PPO epochs with small batches.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=300)
    parser.add_argument("--video_interval", type=int, default=10_000)
    parser.add_argument("--run_name", default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args = parser.parse_args()
if "Vision" in args.task or args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Omniverse-dependent imports must happen after AppLauncher constructed the app.
import math
from datetime import datetime

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.io import dump_yaml
from isaaclab.sim.utils import get_current_stage
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from pxr import UsdPhysics

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.agents import register_rl_games_networks
from zero_g_blade_swap.tasks.blade_swap.mdp.grapple import grapple_grip_error_metrics
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import (
    insertion_error_metrics,
    insertion_goal_error,
    secured_blade_error_metrics,
    secured_blade_pose_error,
)


def _fit_minibatch(agent_cfg: dict, num_envs: int) -> None:
    config = agent_cfg["params"]["config"]
    batch = num_envs * int(config["horizon_length"])
    requested = min(int(config["minibatch_size"]), batch)
    divisors = [value for value in range(requested, 0, -1) if batch % value == 0]
    config["minibatch_size"] = divisors[0]
    central = config.get("central_value_config")
    if central is not None:
        central["minibatch_size"] = config["minibatch_size"]


def _validate_robust_smoke_contract(env, robustness_level: int) -> None:
    """Fail a smoke run if a Phase-2 profile was only nominally configured."""

    task = env.unwrapped
    # Six Cartesian corrections, plus one more when the policy owns the gripper.
    # A grasp skill that cannot command the fingers is not learning to grasp.
    expected_action_dim = 6 + (1 if getattr(task.cfg.actions, "gripper", None) is not None else 0)
    if task.action_manager.total_action_dim != expected_action_dim:
        raise RuntimeError(
            f"Expected {expected_action_dim} actions, got {task.action_manager.total_action_dim}"
        )
    if tuple(float(value) for value in task.cfg.sim.gravity) != (0.0, 0.0, 0.0):
        raise RuntimeError(f"Phase-2 gravity is not zero: {task.cfg.sim.gravity}")
    # The force profiles legitimately carry a blade contact sensor and the
    # vision profile a tiled camera. Anything else is an accident, and a render
    # product on a state-only profile is the expensive kind.
    unexpected_sensors = tuple(name for name in task.scene.sensors if name not in ("blade_contact", "camera"))
    if unexpected_sensors:
        raise RuntimeError(f"Task unexpectedly created sensors: {unexpected_sensors}")
    if getattr(task.cfg.scene, "camera", None) is not None and "Vision" not in args.task:
        raise RuntimeError("A state-only profile allocated a tiled camera")
    if getattr(task.cfg, "contact_grasp", False):
        if task.cfg.events.secured_blade_constraint is not None:
            raise RuntimeError("Contact insertion unexpectedly enabled the secured-blade fixture")
        spawn = task.cfg.scene.spare_blade.spawn
        # Two blade geometries reach here. The contact task carries a single
        # handle box; the head-on capture task carries a three-part grapple pin.
        # Either way the point of the check is the same: the thing the fingers
        # are supposed to grip must actually be a live collider at runtime,
        # because this project has twice shipped a grasp task whose gripper
        # touched nothing.
        pin_parts = ("GrapplePin/Shaft", "GrapplePin/Collar", "GrapplePin/Wedge")
        parts = pin_parts if hasattr(spawn, "wedge_x") else ("Handle",)
        if parts == ("Handle",) and not spawn.handle_collision_enabled:
            raise RuntimeError("Contact insertion handle collision is disabled")
        stage = get_current_stage()
        for part in parts:
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/SpareBlade/{part}")
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = collision_api.GetCollisionEnabledAttr().Get() if collision_api else None
            if not prim.IsValid() or not collision_api or collision_enabled is False:
                raise RuntimeError(
                    f"Grasp interface collider '{part}' is not active at runtime: "
                    f"valid={prim.IsValid()}, collision_enabled={collision_enabled}"
                )
        print(f"[INFO] Physical-grasp contract passed: {', '.join(parts)} collidable, software fixture off")
    if getattr(task.cfg, "rigid_grasp", False):
        joint_prim = get_current_stage().GetPrimAtPath("/World/envs/env_0/GraspJoint/Joint")
        if not joint_prim.IsValid() or not UsdPhysics.FixedJoint(joint_prim):
            raise RuntimeError("Rigid-grasp task did not author its PhysX fixed joint")
        if task.cfg.events.secured_blade_constraint is not None:
            raise RuntimeError("Rigid-grasp task unexpectedly enabled the old wrench fixture")
        if task.cfg.scene.spare_blade.spawn.handle_collision_enabled:
            raise RuntimeError("Rigid-grasp task must disable redundant finger/handle collision")
        collision = {
            "floor": bool(task.cfg.scene.blade_slot.spawn.collision_props.collision_enabled),
            "left": bool(task.cfg.scene.blade_slot_left_guide.spawn.collision_props.collision_enabled),
            "right": bool(task.cfg.scene.blade_slot_right_guide.spawn.collision_props.collision_enabled),
        }
        expected = (
            {"floor": False, "left": False, "right": False}
            if robustness_level == 0
            else {"floor": False, "left": True, "right": True}
            if robustness_level in (1, 2)
            else {"floor": False, "left": True, "right": True}
        )
        if collision != expected:
            raise RuntimeError(
                f"Rigid-grasp rail profile mismatch at level {robustness_level}: {collision} != {expected}"
            )
        print(
            "[INFO] Rigid-grasp contract passed: PhysX fixed joint on, "
            f"redundant handle collision off, rail collision={collision}"
        )
    if robustness_level >= 2:
        masses = task.scene["spare_blade"].root_physx_view.get_masses()
        if not bool(((masses >= 5.0) & (masses <= 15.0)).all()):
            raise RuntimeError(f"Phase-2 blade mass escaped [5, 15] kg: {masses}")
    if robustness_level >= 3:
        material_asset = "blade_slot_left_guide" if getattr(task.cfg, "rigid_grasp", False) else "blade_slot"
        material = task.scene[material_asset].root_physx_view.get_material_properties()
        valid_material = (
            (material[..., 0] >= 0.25)
            & (material[..., 0] <= 2.0)
            & (material[..., 1] >= 0.20)
            & (material[..., 1] <= 1.5)
            & (material[..., 1] <= material[..., 0])
            & (material[..., 2] >= 0.0)
            & (material[..., 2] <= 0.05)
        )
        if not bool(valid_material.all()):
            raise RuntimeError(f"Phase-2 guide material escaped its configured buckets: {material}")
        breakaway, viscous = task._rail_stiction["spare_blade"]
        if not bool(((breakaway >= 10.0) & (breakaway <= 120.0)).all()):
            raise RuntimeError("Phase-2 breakaway force escaped [10, 120] N")
        if not bool(((viscous >= 2.0) & (viscous <= 25.0)).all()):
            raise RuntimeError("Phase-2 viscous drag escaped [2, 25] N s/m")
    print(
        "[INFO] Phase-2 smoke contract passed: six actions, zero gravity, no sensors"
        f", robustness level {robustness_level}"
    )


def main() -> None:
    env = None
    try:
        rl_device = args.device or "cuda:0"
        if rl_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"GPU training requested on {rl_device}, but PyTorch cannot access CUDA")
        device_description = (
            torch.cuda.get_device_name(torch.device(rl_device)) if rl_device.startswith("cuda") else "CPU"
        )
        print(f"[INFO] Simulation and PPO device: {rl_device} ({device_description})")
        env_cfg = parse_env_cfg(args.task, device=rl_device, num_envs=args.num_envs)
        if args.robustness_level is not None:
            valid_profile_task = any(
                label in args.task for label in ROBUST_FAMILY_TASKS
            )
            if not valid_profile_task:
                raise ValueError("--robustness_level is valid only for a robust/contact insertion task")
            env_cfg.configure_robustness(args.robustness_level)
            print(f"[INFO] Insertion robustness level: {args.robustness_level}")
        agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["seed"] = args.seed
        agent_cfg["params"]["config"]["device"] = rl_device
        agent_cfg["params"]["config"]["device_name"] = rl_device
        if args.max_iterations is not None:
            agent_cfg["params"]["config"]["max_epochs"] = args.max_iterations
        if args.smoke:
            agent_cfg["params"]["config"]["max_epochs"] = 2
            agent_cfg["params"]["config"]["save_best_after"] = 0
            agent_cfg["params"]["config"]["save_frequency"] = 1
        if args.bc_checkpoint is not None:
            if "Vision" not in args.task:
                raise ValueError("--bc_checkpoint is only valid for the Vision task")
            if not args.bc_checkpoint.is_file():
                raise FileNotFoundError(args.bc_checkpoint)
            agent_cfg["params"]["network"]["bc_checkpoint"] = str(args.bc_checkpoint.resolve())

        env_cfg.seed = args.seed
        num_envs = int(env_cfg.scene.num_envs)
        _fit_minibatch(agent_cfg, num_envs)
        name = agent_cfg["params"]["config"]["name"]
        run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = Path("logs") / "rl_games" / name
        run_dir = (log_root / run_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        agent_cfg["params"]["config"]["train_dir"] = str(log_root.resolve())
        agent_cfg["params"]["config"]["full_experiment_name"] = run_name
        env_cfg.log_dir = str(run_dir)
        dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)

        render_mode = "rgb_array" if args.video else None
        env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
        print(f"[INFO] Created Gym environment for {args.task}")
        if args.smoke:
            smoke_observations, _ = env.reset()
            if any(label in args.task for label in ROBUST_FAMILY_TASKS):
                _validate_robust_smoke_contract(env, int(env_cfg.robustness_level))
            if not all(torch.isfinite(value).all() for value in smoke_observations.values()):
                raise RuntimeError("Smoke reset produced a non-finite observation")
            print("[INFO] Smoke reset produced finite observations")
            smoke_actions = torch.zeros(
                (num_envs, env.unwrapped.action_manager.total_action_dim),
                device=env.unwrapped.device,
            )
            try:
                smoke_observations, smoke_rewards, _, _, _ = env.step(smoke_actions)
            except BaseException:
                traceback.print_exc()
                raise
            if not all(torch.isfinite(value).all() for value in smoke_observations.values()):
                raise RuntimeError("Smoke step produced a non-finite observation")
            if not torch.isfinite(smoke_rewards).all():
                raise RuntimeError("Smoke step produced a non-finite reward")
            print("[INFO] Smoke environment step produced finite observations and rewards")
            # Which frame convention is this task's grasp expressed in?
            # ``secured_blade_error_metrics`` measures orientation against the
            # top-down GRIPPER_GRASP_ROT, so on a head-on capture it reports the
            # 90 degrees between the two conventions as grasp error and every
            # check below fails for the wrong reason.
            head_on = getattr(env_cfg, "tool_target_rot", None) is not None
            grip_metrics = grapple_grip_error_metrics if head_on else secured_blade_error_metrics
            if getattr(env_cfg, "contact_grasp", False) or getattr(env_cfg, "rigid_grasp", False):
                stationary_reward = smoke_rewards.clone()
                # The chained-insert task opens every episode with a capture
                # prologue the learning policy does not act in, and that prologue
                # carries **exactly zero reward** by construction. Twenty steps of
                # standing still therefore sum to zero rather than to something
                # negative, and asserting otherwise would be asserting that the
                # prologue is not there. Run the phase machine to the hand-off
                # first, then apply the same contract to the phase the policy
                # actually owns -- which also makes this the check that the
                # hand-off fires at all.
                chain_phase = getattr(env.unwrapped, "chain_phase", None)
                if chain_phase is not None:
                    from zero_g_blade_swap.tasks.blade_swap.chained_insert_env_cfg import INSERT

                    handover_budget = int(round(float(env.unwrapped.chain_capture_deadline)))
                    for _ in range(handover_budget):
                        env.step(smoke_actions)
                        if bool((chain_phase == INSERT).any()):
                            break
                    reached = int((chain_phase == INSERT).sum())
                    if reached == 0:
                        raise RuntimeError(
                            "Chained-insert smoke failed: no environment reached the insert phase within the "
                            f"capture skill's own {handover_budget}-step budget, so the hand-off never fired."
                        )
                    print(
                        f"[INFO] Chained insert: {reached} of {num_envs} environments reached the hand-off",
                        flush=True,
                    )
                    inserting = chain_phase == INSERT
                    stationary_reward = torch.zeros_like(smoke_rewards)
                    for _ in range(20):
                        _, reward, _, _, _ = env.step(smoke_actions)
                        stationary_reward += reward
                    stationary_reward = stationary_reward[inserting]
                else:
                    for _ in range(19):
                        _, reward, _, _, _ = env.step(smoke_actions)
                        stationary_reward += reward
                if not bool((stationary_reward < 0.0).all()):
                    raise RuntimeError(
                        "Contact reward contract failed: standing still must have negative cumulative reward, "
                        f"got {stationary_reward}"
                    )
                print("[INFO] Contact reward contract passed: standing still is net-negative")
                grasp_vector, grasp_angle = secured_blade_pose_error(env.unwrapped)
                print(
                    "[INFO] Settled tool-to-handle error: "
                    f"xyz={grasp_vector[0].tolist()} m, orientation={float(grasp_angle[0]):.4f} rad",
                    flush=True,
                )
                # Pull a short distance away from the rails. The contact task
                # must transmit this through its pads; the secured-grasp task
                # must transmit it through its audited PhysX fixed joint.
                if head_on:
                    # This probe's thresholds encode the fixed-joint grasp it
                    # was written for, where tool-to-handle error is zero by
                    # construction. A head-on capture is not like that: closing
                    # drives the pin along the wedge until the collar catches
                    # it, so a correctly seated grip sits about 12.5 mm from the
                    # nominal grip point and trips a 12 mm tolerance while
                    # working exactly as designed.
                    #
                    # Holding capacity for these tasks is not asserted here at
                    # all. It is measured by scripts/grasp_diagnostics.py over a
                    # closure-by-force grid, which is a far better test than
                    # twenty scripted steps: see
                    # evidence/grapple_pin_axial_pull_gate.json, 69 N held
                    # against the 66.4 N requirement.
                    seated, seated_angle = grip_metrics(env.unwrapped)
                    print(
                        "[INFO] Skipping the scripted pull test on a head-on capture task. "
                        f"Seated grip offset {float(seated.max()):.4f} m, attitude {float(seated_angle.max()):.4f} rad. "
                        "Holding capacity is gated by scripts/grasp_diagnostics.py.",
                        flush=True,
                    )
                else:
                    contact_actions = torch.zeros_like(smoke_actions)
                    contact_actions[:, 0] = -0.50
                    blade_x_before_pull = env.unwrapped.scene["spare_blade"].data.root_pos_w[:, 0].clone()
                    for _ in range(20):
                        env.step(contact_actions)
                    blade_x_after_pull = env.unwrapped.scene["spare_blade"].data.root_pos_w[:, 0]
                    blade_pull_distance = blade_x_before_pull - blade_x_after_pull
                    grasp_position, grasp_orientation = grip_metrics(env.unwrapped)
                    max_position = float(grasp_position.max())
                    max_orientation = float(grasp_orientation.max())
                    min_pull_distance = float(blade_pull_distance.min())
                    contact_motion_failed = getattr(env_cfg, "contact_grasp", False) and min_pull_distance < 0.003
                    if contact_motion_failed or max_position > 0.012 or max_orientation > 0.12:
                        raise RuntimeError(
                            "Physical-grasp pull test failed: "
                            f"blade_motion={min_pull_distance:.4f} m, position={max_position:.4f} m, "
                            f"orientation={max_orientation:.4f} rad"
                        )
                    print(
                        "[INFO] Physical-grasp pull test passed: "
                        f"blade_motion={min_pull_distance:.4f} m, position={max_position:.4f} m, "
                        f"orientation={max_orientation:.4f} rad"
                    )
            # The scripted insertion probe below is exactly that: an insertion
            # probe. It servos toward the slot goal and reads the insertion
            # termination terms by name, so it is meaningless on a task whose
            # objective is capture or extraction, and those tasks do not define
            # the terms it asks for.
            #
            # It is scoped to the contact-grasp family, which is what it was
            # written and tuned for. docs/status.md recorded it exhausting its
            # 300-step budget with 23.5 mm of residual axial error on the
            # rigid-grasp task while the learned policy inserts in 35 control
            # steps, so running it there reported a probe defect as a task
            # failure. On a rigid grasp the blade is welded to the tool anyway,
            # so axial feasibility is true by construction and the probe tests
            # nothing. A head-on capture task defines neither term it reads.
            insertion_probe_applies = getattr(env_cfg, "contact_grasp", False)
            insertion_probe_applies &= getattr(env_cfg.terminations, "insertion_success", None) is not None
            insertion_probe_applies &= not head_on
            if insertion_probe_applies:
                env.reset()
                # Physics-feasibility check only: a slow axial command must be
                # able to complete the near insertion through pad contact.
                # This is never used by PPO or the live demonstration.
                for _ in range(10):
                    env.step(smoke_actions)
                insertion_actions = torch.zeros_like(smoke_actions)
                inserted = False
                insertion_reward = torch.zeros(num_envs, device=env.unwrapped.device)
                for insertion_step in range(300):
                    axial_error, lateral_error, orientation_error = insertion_error_metrics(env.unwrapped)
                    pose_error = insertion_goal_error(env.unwrapped)
                    # Quasi-static closed-loop insertion: taper the tool speed
                    # near the target and correct small rail-induced lateral
                    # and angular errors. This proves controllability only;
                    # PPO never receives these test actions.
                    insertion_speed = ((axial_error - 0.0105) / 0.0195 * 0.20).clamp(0.0, 0.20)
                    insertion_actions[:, 0] = torch.where(
                        axial_error > 0.0105,
                        insertion_speed.clamp_min(0.08),
                        torch.zeros_like(axial_error),
                    )
                    lateral_action = (0.20 * pose_error[:, 1:3] / 0.010).clamp(-0.25, 0.25)
                    lateral_action = torch.sign(lateral_action) * torch.where(
                        lateral_action.abs() > 0.0,
                        lateral_action.abs().clamp_min(0.08),
                        torch.zeros_like(lateral_action),
                    )
                    lateral_aligned = torch.linalg.vector_norm(pose_error[:, 1:3], dim=-1) <= 0.0023
                    insertion_actions[:, 1:3] = torch.where(
                        lateral_aligned.unsqueeze(-1), torch.zeros_like(lateral_action), lateral_action
                    )
                    angular_action = (0.15 * pose_error[:, 3:6] / 0.10).clamp(-0.20, 0.20)
                    angular_aligned = torch.linalg.vector_norm(pose_error[:, 3:6], dim=-1) <= 0.040
                    insertion_actions[:, 3:6] = torch.where(
                        angular_aligned.unsqueeze(-1), torch.zeros_like(angular_action), angular_action
                    )
                    geometry_valid = (
                        (axial_error <= 0.0115)
                        & (lateral_error <= 0.0023)
                        & (orientation_error <= 0.045)
                    )
                    insertion_actions[:] = torch.where(
                        geometry_valid.unsqueeze(-1), torch.zeros_like(insertion_actions), insertion_actions
                    )
                    _, reward, _, _, _ = env.step(insertion_actions)
                    insertion_reward += reward
                    if bool(env.unwrapped.termination_manager.get_term("insertion_failed").any()):
                        conditions = env.unwrapped._insertion_latest_failure_conditions
                        metrics = env.unwrapped._insertion_latest_failure_metrics
                        failed = [name for name, value in conditions.items() if bool(value.any())]
                        raise RuntimeError(
                            "Physical-grasp axial feasibility test failed at "
                            f"step {insertion_step}: conditions={failed}, "
                            f"axial={float(metrics['axial'].max()):.4f} m, "
                            f"lateral={float(metrics['lateral'].max()):.4f} m, "
                            f"grasp_position={float(metrics['grasp_position'].max()):.4f} m, "
                            f"grasp_orientation={float(metrics['grasp_orientation'].max()):.4f} rad"
                        )
                    if bool(env.unwrapped.termination_manager.get_term("insertion_success").all()):
                        inserted = True
                        break
                if not inserted:
                    axial, lateral, orientation = insertion_error_metrics(env.unwrapped)
                    grasp_position, grasp_orientation = secured_blade_error_metrics(env.unwrapped)
                    conditions = getattr(env.unwrapped, "_insertion_latest_success_conditions", {})
                    unmet = [name for name, value in conditions.items() if not bool(value.all())]
                    raise RuntimeError(
                        "Physical-grasp axial feasibility test did not complete insertion: "
                        f"unmet={unmet}, axial={float(axial.max()):.4f} m, "
                        f"lateral={float(lateral.max()):.4f} m, orientation={float(orientation.max()):.4f} rad, "
                        f"grasp_position={float(grasp_position.max()):.4f} m, "
                        f"grasp_orientation={float(grasp_orientation.max()):.4f} rad"
                    )
                if not bool((insertion_reward > 0.0).all()):
                    raise RuntimeError(
                        "Insertion reward contract failed: a successful insertion must be net-positive, "
                        f"got {insertion_reward}"
                    )
                print(f"[INFO] Insertion reward contract passed: return={float(insertion_reward.min()):.3f}")
                print("[INFO] Physical-grasp axial feasibility test passed")
                env.reset()
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(run_dir / "videos" / "train"),
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            )

        rl_device = agent_cfg["params"]["config"]["device"]
        env_options = agent_cfg["params"]["env"]
        env = RlGamesVecEnvWrapper(
            env,
            rl_device,
            env_options.get("clip_observations", math.inf),
            env_options.get("clip_actions", math.inf),
            env_options.get("obs_groups"),
            env_options.get("concate_obs_groups", True),
        )
        print("[INFO] Wrapped environment for RL-Games")
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: env})
        agent_cfg["params"]["config"]["num_actors"] = num_envs

        register_rl_games_networks()
        runner = Runner(IsaacAlgoObserver())
        print("[INFO] Loading RL-Games PPO configuration")
        runner.load(agent_cfg)
        runner.reset()
        print("[INFO] RL-Games runner initialized")
        run_args = {"train": True, "play": False, "sigma": None}
        if args.checkpoint is not None:
            if not args.checkpoint.is_file():
                raise FileNotFoundError(args.checkpoint)
            run_args["checkpoint"] = str(args.checkpoint.resolve())
            # rl-games treats ``max_epochs`` as an ABSOLUTE epoch number, not a
            # count of additional epochs, and a resume that has already passed it
            # stops before the first update while still writing a checkpoint --
            # ``..._ep_3201_rew_-inf.pth``, which is indistinguishable from a
            # trained one until something evaluates it. Measured on the
            # chained-insert run: resuming insert v6 at epoch 3200 with
            # ``--max_iterations 1200`` produced exactly that, and the surrounding
            # script went on to spend a certification on it.
            #
            # Fail loudly instead. The epoch comes out of the checkpoint rather
            # than its filename, so a renamed file cannot defeat the check.
            resumed_epoch = int(torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("epoch", 0))
            budget = int(agent_cfg["params"]["config"]["max_epochs"])
            if resumed_epoch >= budget:
                raise ValueError(
                    f"{args.checkpoint.name} is at epoch {resumed_epoch}, but max_epochs is {budget}. "
                    "rl-games counts epochs absolutely, so this run would stop before its first update and "
                    f"still write a checkpoint. Pass --max_iterations {resumed_epoch} + the epochs you want."
                )
            print(f"[INFO] Resuming at epoch {resumed_epoch}; training to absolute epoch {budget}")
        print(f"[INFO] Training {args.task} with {num_envs} environments; logs: {run_dir}")
        runner.run(run_args)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
