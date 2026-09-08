"""Scripted simulator checks for the peg adapter. No learned policy is evaluated."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def expected_outcomes_match(controller, jobs, simulated_seconds, deadline_s):
    """A probe running to completion does not establish its intended behavior."""
    if not jobs:
        return False
    if controller == "release":
        return all(job["outcome"] == "lost_grasp" for job in jobs)
    if controller == "hold":
        expected = "deadline" if simulated_seconds >= deadline_s else "probe_end"
        return all(job["outcome"] == expected for job in jobs)
    return controller in {"insert_withdraw", "retry", "continue"}


def information_probe(env):
    """Perturb cached information only: never simulator state or part poses."""
    import torch

    baseline = env._get_observations()
    held, fixed = env.held_pos.clone(), env.fixed_pos.clone()
    try:
        env.held_pos += 0.01
        env.fixed_pos += 0.01
        changed = env._get_observations()
    finally:
        env.held_pos[:] = held
        env.fixed_pos[:] = fixed
    actor_delta = float((baseline["policy"] - changed["policy"]).abs().max())
    critic_delta = float((baseline["critic"] - changed["critic"]).abs().max())
    frame, noise = env.fixed_pos_obs_frame.clone(), env.init_fixed_pos_obs_noise.clone()
    original = env.generate_ctrl_signals
    targets = []

    def capture(**kwargs):
        targets.append(torch.cat([kwargs["ctrl_target_fingertip_midpoint_pos"],
                                  kwargs["ctrl_target_fingertip_midpoint_quat"]], dim=-1).clone())

    env.generate_ctrl_signals = capture
    try:
        env._apply_action()
        env.fixed_pos_obs_frame += 0.002
        env.init_fixed_pos_obs_noise -= 0.002
        same_estimate = env._get_observations()
        env._apply_action()
    finally:
        env.fixed_pos_obs_frame[:] = frame
        env.init_fixed_pos_obs_noise[:] = noise
        env.generate_ctrl_signals = original
    decomposition_actor_delta = float((baseline["policy"] - same_estimate["policy"]).abs().max())
    target_delta = float((targets[0] - targets[1]).abs().max())
    return {"hidden_pose_actor_delta": actor_delta, "hidden_pose_critic_delta": critic_delta,
            "same_estimate_actor_delta": decomposition_actor_delta, "same_estimate_target_delta": target_delta,
            "passed": actor_delta < 1e-6 and critic_delta > 0 and decomposition_actor_delta < 1e-6 and target_delta < 1e-6,
            "scope": "Specific actor and action-frame pathways; critic has privileged state. Ideal robot proprioception remains."}


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10070)
    parser.add_argument("--gravity", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--velocity", choices=("corrected", "upstream"), default="corrected")
    parser.add_argument("--controller", choices=("hold", "insert_withdraw", "release", "retry", "continue"), default="hold")
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--retry-config", type=Path)
    parser.add_argument("--fault-plan", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-env", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 35:
        parser.error("Probe duration must be 1..35 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "seed": args.seed, "controller": "scripted_" + args.controller,
              "held_part_gravity": args.gravity, "velocity_mode": args.velocity}
    app = env = video = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.evaluation import JobCriteria, JobEvaluator, PhysicsSample
        from assembly_recovery.peg_env import PegStudyEnv, WithinJobMutationError
        from assembly_recovery.retry_controller import ActorRetryController, RetrySettings
        from assembly_recovery.reward_ledger import discounted_job_return

        study = json.loads((ROOT / "configs/study.json").read_text())
        fault_cases = None
        if args.fault_plan:
            from assembly_recovery.faults import development_cases

            plan = json.loads(args.fault_plan.read_text())
            if hashlib.sha256((ROOT / "configs/study.json").read_bytes()).hexdigest() != plan["study_sha256"]:
                raise ValueError("Study differs from the predeclared fault plan")
            fault_cases = development_cases(study, args.seed)
            if args.seconds != 30 or args.controller not in {"retry", "continue"}:
                raise ValueError("Fault matrix requires paired full-deadline scripted jobs")
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=len(fault_cases) if fault_cases else 4)
        cfg.seed = args.seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = args.gravity == "disabled"
        protocol = study["job_protocol"]
        criteria = JobCriteria(physics_dt=cfg.sim.dt, deadline_s=protocol["deadline_seconds"],
                               seated_dwell_s=protocol["seated_dwell_seconds"], force_budget_n=protocol["force_budget_n"])
        if cfg.obs_order != ActorRetryController.observation_order:
            raise RuntimeError("Unexpected actor observation layout")
        cfg.viewer.resolution = (960, 720)
        env_type, fault_kwargs = PegStudyEnv, {}
        if fault_cases:
            from assembly_recovery.fault_env import PegFaultEnv

            env_type = PegFaultEnv
            fault_kwargs = {"cases": fault_cases, "nominal": study["fault_support"]["nominal"]}
        env = env_type(cfg, criteria=criteria, velocity_mode=args.velocity,
                       render_mode="rgb_array" if args.video else None, **fault_kwargs)
        env.reset()
        if fault_cases:
            report["fault_cases"] = fault_cases
            report["initialization_validity"] = env.initialization_report()
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        report.update(clone_in_fabric=cfg.scene.clone_in_fabric, criteria=asdict(criteria), geometry=env.geometry.report(), world_gravity=list(cfg.sim.gravity),
                      robot_gravity_disabled=cfg.robot.spawn.rigid_props.disable_gravity,
                      held_prim_paths=env._held_asset.root_physx_view.prim_paths,
                      fixed_prim_paths=env._fixed_asset.root_physx_view.prim_paths,
                      held_body_names=env._held_asset.body_names, fixed_body_names=env._fixed_asset.body_names,
                      initial_fixed_pos=env.fixed_pos.cpu().tolist(),
                      initial_part_pos=env.held_pos.cpu().tolist(), initial_fingertip_pos=env.fingertip_midpoint_pos.cpu().tolist(),
                      initial_base_clearance_m=(env.held_pos[:, 2] - env.fixed_pos[:, 2] - env.geometry.hole_height_m).cpu().tolist(),
                      information_probe=information_probe(env))
        initial_action = env.actions.clone().clamp(-1, 1)
        report["initial_action"] = initial_action.cpu().tolist()
        report["initial_action_outside_bounds"] = (env.actions.abs() > 1).any(dim=1).cpu().tolist()
        paired_controller = args.controller in {"retry", "continue"}
        settings = RetrySettings(**json.loads(args.retry_config.read_text())) if args.retry_config else RetrySettings()
        actor_obs = env._get_observations()["policy"].cpu().tolist()
        controllers = [ActorRetryController(row, step_dt=env.step_dt, position_bounds=cfg.ctrl.pos_action_bounds,
                                           seated_height_m=env.geometry.seated_fingertip_above_hole_top_m,
                                           retry=args.controller == "retry", settings=settings) for row in actor_obs]
        initial_state = {}
        for name in ("held_pos", "held_quat", "fixed_pos", "fixed_quat", "fingertip_midpoint_pos", "fingertip_midpoint_quat",
                     "init_fixed_pos_obs_noise", "ema_factor", "task_prop_gains", "pos_threshold", "rot_threshold",
                     "dead_zone_thresholds", "contact_penalty_thresholds", "actions", "force_sensor_world_smooth"):
            initial_state[name] = getattr(env, name).cpu().tolist()
        initial_state["actor_observation"] = actor_obs
        initial_state["robot_joint_pos"] = env._robot.data.joint_pos.cpu().tolist()
        initial_state["robot_joint_vel"] = env._robot.data.joint_vel.cpu().tolist()
        for name, asset in (("held", env._held_asset), ("fixed", env._fixed_asset)):
            initial_state[name + "_root_state"] = asset.data.root_state_w.cpu().tolist()
            initial_state[name + "_material"] = asset.root_physx_view.get_material_properties().cpu().tolist()
            initial_state[name + "_mass"] = asset.root_physx_view.get_masses().cpu().tolist()
        report["initial_state"] = initial_state
        report["initial_state_sha256"] = hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest()
        report["runtime"] = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
        report["resolved_reward_scales"] = {name: getattr(cfg.task, name) for name in (
            "action_penalty_ee_scale", "action_grad_penalty_scale", "action_penalty_asset_scale",
            "contact_penalty_scale", "delay_until_ratio")}
        if cfg.task.action_grad_penalty_scale != 0.0:
            raise RuntimeError("Resolved peg action-change reward coefficient must remain zero")
        if args.video:
            import imageio.v2 as imageio

            target = env._fixed_asset.data.root_pos_w[args.video_env].cpu().tolist()
            env.sim.set_camera_view(eye=[target[0] + 0.20, target[1] - 0.22, target[2] + 0.18],
                                    target=[target[0], target[1], target[2] + 0.04])
            video = imageio.get_writer(str(args.output / "trajectory.mp4"), fps=round(1 / env.step_dt), macro_block_size=8)
            report["video"] = {"job_index": args.video_env, "fps": round(1 / env.step_dt), "frames": 0,
                               "scope": "Actual simulator frames; camera positioning uses evaluator geometry only."}
        env.begin_jobs(args.output.parent.name)
        for i, validity in enumerate(report.get("initialization_validity", [])):
            if not validity["valid"]:
                env.jobs[i].finish("initialization_invalid")
                env.finished_mask[i] = True
        rollout_started = time.monotonic()
        resets = 0
        controls = []
        control_steps = math.ceil(args.seconds / env.step_dt)
        for step in range(control_steps):
            elapsed = step * env.step_dt
            action = initial_action.clone()
            phase = "hold"
            controller_states = []
            if paired_controller:
                decisions = [controller.act(row, elapsed) for controller, row in zip(controllers, actor_obs, strict=True)]
                action = torch.tensor([decision[0] for decision in decisions], device=env.device)
                controller_states = [decision[1] for decision in decisions]
                phase = [state["phase"] for state in controller_states]
            if args.controller == "insert_withdraw" and elapsed >= 1:
                phase = "insert" if elapsed < 4 else "withdraw"
                action[:, :2] = 0
                target = env.geometry.seated_fingertip_above_hole_top_m if phase == "insert" else 0.045
                action[:, 2] = target / cfg.ctrl.pos_action_bounds[2]
            if args.controller == "release" and elapsed >= 1:
                phase = "release"
                env.release_gripper = True
            env.commanded_down = [p in {"insert", "search", "seat"} for p in phase] if paired_controller else phase == "insert"
            active = [job.outcome is None for job in env.jobs]
            observation_before = actor_obs
            with torch.inference_mode():
                obs, reward, terminated, truncated, _ = env.step(action)
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite observations or reward")
            resets += int((terminated | truncated).sum())
            actor_obs = obs["policy"].cpu().tolist()
            controls.append({"step": step + 1, "phase": phase, "active_before_step": active,
                             "actor_observation_before": observation_before, "commanded_action": action.cpu().tolist(),
                             "applied_action": env.actions.cpu().tolist(), "controller_state": controller_states,
                             "reward_terms": {k: v.cpu().tolist() for k, v in env.reward_terms.items()},
                             "success_prediction_scale": env.success_pred_scale,
                             "reward": reward.cpu().tolist(), "part_pos": env.held_pos.cpu().tolist(),
                             "part_quat": env.held_quat.cpu().tolist(),
                             "fingertip_pos": env.fingertip_midpoint_pos.cpu().tolist(),
                             "finger_joint_positions": env._robot.data.joint_pos[:, -2:].cpu().tolist()})
            if video is not None:
                video.append_data(env.render())
                report["video"]["frames"] += 1
            if step % 75 == 0:
                print(json.dumps({"control_step": step + 1, "time_s": (step + 1) * env.step_dt,
                                  "phase": phase, "outcomes": [job.outcome for job in env.jobs]}), flush=True)
        jobs = env.end_jobs()
        rollout_wall_s = time.monotonic() - rollout_started
        report["resources"] = {"rollout_wall_s": rollout_wall_s, "control_transitions": control_steps * env.num_envs,
                               "control_transitions_per_wall_s": control_steps * env.num_envs / rollout_wall_s,
                               "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
                               "scope": "Instrumented scripted rollout including recording; Torch memory excludes PhysX and renderer. Not training capacity."}
        report["controllers"] = [controller.report() for controller in controllers] if paired_controller else []
        report["returns"] = [discounted_job_return(controls, i, job.last_step, cfg.decimation)
                             for i, job in enumerate(env.jobs)]
        replay = []
        with (args.output / "physics_samples.jsonl").open("x") as handle:
            for job, samples, contacts in zip(jobs, env.samples, env.contact_samples, strict=True):
                evaluator = JobEvaluator(job["job_id"], criteria)
                for sample, contact in zip(samples, contacts, strict=True):
                    handle.write(json.dumps({"job_id": job["job_id"], "peg_fixture_contact_force_n": contact, **sample}) + "\n")
                    evaluator.observe(PhysicsSample(**sample))
                evaluator.finish("initialization_invalid" if job["outcome"] == "initialization_invalid" else "probe_end")
                replay.append(evaluator.result())
        with (args.output / "control_samples.jsonl").open("x") as handle:
            for control in controls:
                handle.write(json.dumps(control) + "\n")
        contact_reports = []
        for contacts in env.contact_samples:
            longest = current = 0
            for force in contacts:
                current = current + 1 if force > 0.1 else 0
                longest = max(longest, current)
            contact_reports.append({"peak_pair_force_n": max(contacts, default=0),
                                    "contact_seconds_above_0p1n": sum(x > 0.1 for x in contacts) * criteria.physics_dt,
                                    "longest_contact_seconds_above_0p1n": longest * criteria.physics_dt})
        report["peg_fixture_contact"] = contact_reports
        report["contact_filter_paths"] = env.contact_filter_paths
        report["finger_contact"] = []
        for samples in env.finger_contact_samples:
            both = [min(forces) > 0.01 for forces in samples]
            current = longest = 0
            for contact in both:
                current = 0 if contact else current + 1
                longest = max(longest, current)
            report["finger_contact"].append({"bilateral_contact_fraction_above_0p01n": sum(both) / max(1, len(both)),
                                             "longest_without_bilateral_contact_s": longest * criteria.physics_dt,
                                             "final_finger_forces_n": samples[-1] if samples else None})
        report["contact_reporting_observed"] = any(c["peak_pair_force_n"] > 0.1 for c in contact_reports)
        report["contact_sensor_body_count"] = env.fixture_contact.body_physx_view.count
        report["expected_physics_steps_if_unfinished"] = control_steps * cfg.decimation
        if any(len(samples) != control_steps * cfg.decimation for job, samples in zip(jobs, env.samples, strict=True) if job["outcome"] == "probe_end"):
            raise RuntimeError("Physics sampling missed steps")
        report.update(jobs=jobs, replay_matches=replay == jobs, automatic_resets=resets,
                      initializations_during_jobs=env.initializations - env.job_initializations,
                      completed_control_steps=control_steps, simulated_seconds=control_steps * env.step_dt,
                      velocity_identity_max_error_m_s=max(env.velocity_errors, default=0),
                      final_part_pos=env.held_pos.cpu().tolist())
        guards = {}
        for operation in ("reset", "part_pose_write"):
            env.begin_jobs("deliberate_guard_probe_" + operation)
            before = env._held_asset.data.root_pos_w.clone()
            blocked = False
            try:
                if operation == "reset":
                    env._reset_idx(torch.arange(env.num_envs, device=env.device))
                else:
                    pose = env._held_asset.data.root_pose_w.clone()
                    pose[:, 0] += 0.01
                    env._held_asset.write_root_pose_to_sim(pose)
            except WithinJobMutationError:
                blocked = True
            guards[operation] = {"blocked": blocked, "part_unchanged": bool(torch.equal(before, env._held_asset.data.root_pos_w)),
                                 "jobs": env.end_jobs()}
        report["mutation_guards"] = guards
        report["scope"] = ["Development scripted probes, not a policy or recovery benchmark.",
                           "Held-part gravity is varied; robot gravity remains idealized compensation.",
                           "Wrist incoming joint load includes grasp, inertia and gravity; it is not isolated fixture contact.",
                           "Peg-side tensor sensor filters the actual fixture rigid body discovered from USD APIs; actor does not receive this measurement. A zero-only trial cannot establish sensor validity.",
                           "The stall detector remains a wrist-load/progress proxy; sustained contact alone does not prove a jam.",
                           "Force threshold triggers a commanded hold after observation, not a guaranteed force ceiling.",
                           "Loss of grasp uses gross separation or 0.1 s without bilateral finger contact above 0.01 N. Contact thresholds remain development choices.",
                           "API guards do not sandbox arbitrary low-level PhysX/USD mutation.",
                           "Per-physics-step CPU traces are for validation, not training throughput."]
        passed = report["replay_matches"] and resets == 0 and report["initializations_during_jobs"] == 0
        passed &= report["information_probe"]["passed"]
        report["expected_controller_outcomes"] = expected_outcomes_match(args.controller, jobs, control_steps * env.step_dt, criteria.deadline_s)
        passed &= report["expected_controller_outcomes"]
        passed &= env.fixture_contact.body_physx_view.count == env.num_envs
        if args.controller == "insert_withdraw":
            passed &= report["contact_reporting_observed"]
        passed &= all(g["blocked"] and g["part_unchanged"] for g in guards.values())
        if not passed:
            raise RuntimeError("Validation checks failed; inspect report")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if video is not None:
            video.close()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
