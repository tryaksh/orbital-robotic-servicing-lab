"""Conditional common20mm/s-reference scripted-prefix diagnostic; no training or final tests."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", type=int, choices=(4, 8), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(10071,), required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.controller = "frozen_scripted_prefix"
    args.mode = "frozen_script_prefix"
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "controller": args.controller, "seed": args.seed,
               "research_result": False, "reward_version": "completion_credit_v3", "diagnostic_protocol": "approach_slew_v1", "mode": args.mode, "refinement": args.refinement, "training_run_status": "completed", "checkpoint_selection": "Final registered 22-cohort budget checkpoint; no selection by score"}
    report["controller_condition"] = {
        "external_servo_hz": 480, "external_sensor_hz": 120, "policy_hz": 15,
        "native_physics_hz": 120 * args.refinement, "initialization_physics_hz": 120,
        "high_level_controller": "Frozen unchanged scripted first-attempt prefix",
        "change": "Common Euclidean XYZ requested-reference slew20mm/s at480Hz, followed by unchanged position-error clamps. Native physics480/960Hz; policy15Hz, sensor120Hz, gains, gripper and raw20N abort unchanged.",
        "cartesian_reference_slew_m_s": 0.020,
        "reference_initialization": "Actual tool pose captured between jobs",
        "reference_limit_order": "Requested world reference, Euclidean slew, original per-axis position-error clamp, original terminal hold",
        "actor_memory_limitation": "The interpolated reference is shared actuator state absent from the unchanged24-scalar actor vector. Previous actions still report original applied EMA actions. Audit this partial observability in any later common training comparison.",
    }
    app = env = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.approach_slew_env_v1 import PegApproachSlewEnvV1
        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.faults import development_cases
        from assembly_recovery.policy_diagnosis import capture_normal_draws
        from assembly_recovery.protocol import sha256
        from assembly_recovery.retry_controller import ActorRetryController, RetrySettings
        from assembly_recovery.study_ppo import StudyPolicy
        from scripts.probe_training import replay_physics

        torch.set_num_threads(1)
        study = json.loads((ROOT / "configs/study.json").read_text())
        if sha256(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError("Checkpoint changed after prelaunch capture")
        cases = development_cases(study, args.seed)
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=28)
        cfg.seed = args.seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        c = study["job_protocol"]
        criteria = JobCriteria(physics_dt=cfg.sim.dt, deadline_s=c["deadline_seconds"],
                               seated_dwell_s=c["seated_dwell_seconds"], force_budget_n=c["force_budget_n"])
        env = PegApproachSlewEnvV1(cfg, criteria=criteria, cases=cases, nominal=study["fault_support"]["nominal"], audit=True, refinement=args.refinement)
        start = time.monotonic()
        observation, _ = env.reset()
        report["initialization_wall_s"] = time.monotonic() - start
        dump_yaml(str(args.output / "initialization_environment.yaml"), cfg)
        validity = env.initialization_report()
        report.update(cases=cases, initialization=validity, criteria=asdict(criteria))
        if not all(v["valid"] for v in validity):
            raise RuntimeError("Invalid initialization; retain failed requests")
        # Instantiation occurs after reset in every arm, exactly as pilot evaluation.
        model = StudyPolicy(observation["policy"].shape[-1], observation["critic"].shape[-1]).to(env.device)
        loaded = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        if loaded["status"] != "budget_complete" or loaded["completed_cohorts"] != 22 or loaded["spec"] != model.spec or loaded.get("optimizer_version") != "value_unclipped_v4":
            raise ValueError("Expected the registered complete-budget value-unclipped-v4 checkpoint")
        model.load_state_dict(loaded["model"])
        model.eval()
        report.update(checkpoint_sha256=args.checkpoint_sha256, policy_spec=model.spec,
                      training_cost=loaded["cost"], action_std=model.log_std.clamp(-5, 2).exp().detach().cpu().tolist())
        initial = {name: getattr(env, name).detach().cpu().clone() for name in (
            "held_pos", "held_quat", "fixed_pos", "fixed_quat", "fingertip_midpoint_pos", "fingertip_midpoint_quat",
            "init_fixed_pos_obs_noise", "ema_factor", "task_prop_gains", "pos_threshold", "rot_threshold",
            "dead_zone_thresholds", "contact_penalty_thresholds", "actions", "force_sensor_world_smooth")}
        initial.update(actor_observation=observation["policy"].cpu().clone(),
                       robot_joint_pos=env._robot.data.joint_pos.cpu().clone(),
                       robot_joint_vel=env._robot.data.joint_vel.cpu().clone(),
                       material=env._fixed_asset.root_physx_view.get_material_properties().cpu().clone())
        settings = RetrySettings(**json.loads((ROOT / "configs/retry_unload_realign_v1.json").read_text()))
        controllers = [ActorRetryController(row, step_dt=env.step_dt, position_bounds=cfg.ctrl.pos_action_bounds,
                       seated_height_m=env.geometry.seated_fingertip_above_hole_top_m, retry=False, settings=settings)
                       for row in observation["policy"].cpu().tolist()]
        if abs(env.step_dt * 60 - 4.) > 1e-9 or settings.first_attempt_s != 4.:
            raise ValueError("The fixed prefix must equal the unchanged four-second first-attempt interval")
        report.update(controller_position_bounds=list(cfg.ctrl.pos_action_bounds),
                      controller_seated_height_m=env.geometry.seated_fingertip_above_hole_top_m,
                      controller_step_dt=env.step_dt, controller_settings=asdict(settings),
                      prefix_controls=60, prefix_physics_steps=env.timing.handoff_step, prefix_seconds=4.,
                      controller_schedule="Unchanged first-attempt script through control 59; end four-second diagnostic without a controller handoff.",
                      policy_phase_label="Frozen unchanged scripted first-attempt prefix")
        sensor_initial = {name: getattr(env, name).detach().cpu().clone() for name in (
            "_previous_noisy_pos", "_previous_noisy_quat", "ee_linvel_fd", "ee_angvel_fd",
            "noisy_fingertip_pos", "noisy_fingertip_quat", "noisy_force", "force_sensor_smooth",
            "force_sensor_world", "force_sensor_world_smooth", "flip_quats", "fixed_pos_obs_frame")}
        physical_initial = {}
        for label, asset in (("robot", env._robot), ("held", env._held_asset), ("fixed", env._fixed_asset)):
            for key, method in (("mass", "get_masses"), ("material", "get_material_properties")):
                physical_initial[label + "_" + key] = getattr(asset.root_physx_view, method)().cpu().clone()
        rng_before_prepare = torch.cuda.get_rng_state(env.device)
        cpu_rng_before_prepare = torch.get_rng_state()
        time_before_prepare = env.sim.current_time
        sensor_time_before_prepare = env._study_timestamp
        env.prepare_job_resolution()
        criteria = env.criteria
        before_begin = env._get_observations()
        report["preparation_checks"] = {
            "initial_actor_unchanged": torch.equal(before_begin["policy"], observation["policy"]),
            "initial_critic_unchanged": torch.equal(before_begin["critic"], observation["critic"]),
            "rng_unchanged": torch.equal(rng_before_prepare, torch.cuda.get_rng_state(env.device)),
            "cpu_rng_unchanged": torch.equal(cpu_rng_before_prepare, torch.get_rng_state()),
            "no_physics_step": time_before_prepare == env.sim.current_time,
            "sensor_timestamp_unchanged": sensor_time_before_prepare == env._study_timestamp,
            "sensor_filter_state_unchanged": all(torch.equal(v, getattr(env, k).cpu()) for k, v in sensor_initial.items()),
            "physical_pose_unchanged": all(torch.equal(initial[k], getattr(env, k).cpu()) for k in (
                "held_pos", "held_quat", "fixed_pos", "fixed_quat", "fingertip_midpoint_pos", "fingertip_midpoint_quat")),
        }
        if not all(report["preparation_checks"].values()):
            raise RuntimeError("Resolution preparation changed initialized state or RNG")
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        report.update(criteria=asdict(criteria), geometry=env.geometry.report(),
                      contact_filter_paths=env.contact_filter_paths,
                      contact_force_conversion_dt=env.fixture_contact._sim_physics_dt,
                      actual_solver_dt=env.timing.physics_dt,
                      scene_metadata_dt=env.sim.get_physics_dt(),
                      stepping_backend="PhysicsContext simulate(dt,current_time)/fetch_results with explicit native dt; no USD edits",
                      held_part_gravity_enabled=not cfg.task.held_asset.spawn.rigid_props.disable_gravity,
                      robot_gravity_disabled=cfg.robot.spawn.rigid_props.disable_gravity,
                      world_gravity=env.sim.get_physics_context().get_gravity(),
                      sensor_noise_parameters={"position": cfg.obs_rand.fingertip_pos,
                          "rotation_deg": cfg.obs_rand.fingertip_rot_deg, "force": cfg.obs_rand.ft_force,
                          "force_alpha": cfg.ft_smoothing_factor})
        env.begin_jobs("diagnosis")
        report["begin_checks"] = {
            "observations_unchanged": all(torch.equal(v, env._get_observations()[k]) for k, v in observation.items()),
            "rng_unchanged": torch.equal(rng_before_prepare, torch.cuda.get_rng_state(env.device)),
            "sensor_filter_state_unchanged": all(torch.equal(v, getattr(env, k).cpu()) for k, v in sensor_initial.items()),
        }
        if not all(report["begin_checks"].values()):
            raise RuntimeError("begin_jobs changed the confirmed observation or RNG path")
        report["status"] = "running"
        report["cost"] = env.cost_report()
        (args.output / "progress.json").write_text(json.dumps({"status": "initialized", "cost": report["cost"]}) + "\n")
        names = ("actor_before", "mean_action", "requested_action", "applied_action", "target_xyz", "tool_xyz",
                 "part_relative_xyz", "part_quat", "raw_reward", "job_reward", "active", "dead_zone", "completion_credit", "upstream_job_reward", "critic_before", "normalized_critic_before", "critic_value")
        arrays = {k: [] for k in names}
        terms, draws, rng_states, phases = {}, [], [], []
        done_counts = torch.zeros(28, dtype=torch.int64, device=env.device)
        returns = torch.zeros(28, dtype=torch.float64, device=env.device)
        rollout_start = interval_start = time.monotonic()
        throughput = []
        for step in range(60):
            with torch.no_grad():
                mean, _, critic_value, normalized = model.act(observation, deterministic=True)
                decisions = [ctrl.act(row, step * env.step_dt) for ctrl, row in
                             zip(controllers, observation["policy"].cpu().tolist(), strict=True)]
                action = torch.tensor([d[0] for d in decisions], device=env.device)
                phases.append([d[1]["phase"] for d in decisions])
                rng_states.append(torch.cuda.get_rng_state(env.device))
                step_draws = []
                with capture_normal_draws(step_draws):
                    next_obs, reward, done, _, extra = env.step(action)
                # Four unchanged draws per 120 Hz sensor tick; eight ticks per control at either resolution.
                if len(step_draws) != 32:
                    raise RuntimeError(f"Unexpected sensor draw count: {len(step_draws)}")
                draws.append(torch.stack([torch.cat((step_draws[i], step_draws[i+1],
                              step_draws[i+2][:, None], step_draws[i+3]), dim=-1) for i in range(0, 32, 4)]))
                done_counts += done
                returns += reward.double() * 0.995 ** step
                target = env.actions[:, :3] * torch.tensor(cfg.ctrl.pos_action_bounds, device=env.device)
                values = (observation["policy"], mean, action, env.actions, target,
                          env.fingertip_midpoint_pos - env.fixed_pos_obs_frame - env.init_fixed_pos_obs_noise,
                          env.held_pos - env.fixed_pos, env.held_quat, extra["raw_reward"], reward,
                          extra["valid"], env.dead_zone_thresholds, extra["completion_credit"], extra["upstream_job_reward"],
                          observation["critic"], normalized["critic"], critic_value)
                for name, value in zip(names, values, strict=True):
                    arrays[name].append(value.detach().clone())
                for name, value in env.reward_terms.items():
                    terms.setdefault(name, []).append(value.detach().clone())
                observation = next_obs
            if (step + 1) % 30 == 0:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - interval_start
                throughput.append({"end_control": step + 1, "wall_s": elapsed,
                    "control_transitions_per_s": 30 * 28 / elapsed,
                    "native_physics_env_steps_per_s": 30 * 28 * cfg.decimation / elapsed})
                interval_start = time.monotonic()
                progress = {"mode": args.mode, "refinement": args.refinement, "control_step": step + 1,
                            "cost": env.cost_report()}
                (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
        report.update(rollout_wall_s=time.monotonic() - rollout_start, throughput=throughput)
        active_at_end = env.tensor_jobs.active.clone()
        jobs = env.end_jobs()
        witness = env.contact_witness.results(jobs)
        if not torch.equal(done_counts, (~active_at_end).long()):
            raise RuntimeError("Expected one notification for each native terminal and none for active prefix jobs")
        artifact = {k: torch.stack(v).cpu() for k, v in arrays.items()}
        def stack_records(records):
            return {k: (torch.stack([r[k] for r in records]).cpu() if isinstance(records[0][k], torch.Tensor)
                        else torch.tensor([r[k] for r in records], dtype=torch.bool if isinstance(records[0][k], bool) else torch.float64)) for k in records[0]}
        artifact.update(initial=initial, sensor_initial=sensor_initial, physical_initial=physical_initial,
                        native=stack_records(env.native_records), sensors=stack_records(env.sensor_records),
                        controller_initial={"slew_reference": env.slew_initial_reference.cpu().clone()},
                        physics=env.audit_buffer[:env.executed_job_steps].cpu(), sensor_draws=torch.cat(draws).cpu(),
                        cuda_rng_before_steps=torch.stack(rng_states),
                        reward_terms={k: torch.stack(v).cpu() for k, v in terms.items()})
        torch.save(artifact, args.output / "trajectory.pt")
        report.update(jobs=jobs, recovery_witness=witness, discounted_job_returns=returns.cpu().tolist(),
                      cost=env.cost_report(), cpu_replay=replay_physics(artifact["physics"], criteria, jobs, "diagnosis", witness),
                      script_phases=phases, status="completed")
        report.update(
            prefix_summary={
                "requests": 28,
                "completions_before_prefix_end": sum(j["success"] for j in jobs),
                "active_at_prefix_end": int(active_at_end.sum()),
                "terminal_outcomes_and_probe_end": dict(Counter(j["outcome"] for j in jobs)),
                "force_aborts": sum(j["outcome"] == "force_abort" for j in jobs),
                "forbidden_events": sum(len(j["forbidden_events"]) for j in jobs),
            },
            servo_updates=env.servo_updates,
            sensor_updates=env.sensor_updates,
            native_physics_steps=env.executed_job_steps,
            expected_total_simulated_seconds=4.,
            force_signal_semantics={
                "raw_wrist": "Incoming joint 6D wrench in the child joint frame; installed PhysX tensor API. Inherited force_sensor_world name retained for compatibility. Norm drives unchanged native 20 N abort.",
                "contact_force": "World-frame normal contact vectors on held part, ordered fixed asset, left finger, right finger. Friction is absent from the ContactSensor normal-force matrix.",
                "contact_impulse": "Same contact matrix requested with dt=1; impulse/native_dt equals contact_force.",
                "force_sensor_body_quat": "Body orientation, not a verified child-joint transform; do not rotate wrench with it without checking USD joint local frame.",
            },
            scope_and_limitations=[
                "Only the unchanged four-second scripted first-attempt prefix acts. The checkpoint is instantiated and evaluated to preserve the confirmed RNG and initialization path; it produces no applied actions.",
                "All 28 fixed seed-10071 development cases. No optimization, controller tuning, full learned job evaluation, final-test use or task correction.",
                "Active requests at four seconds end as probe_end for bounded diagnostic bookkeeping; they are censored prefix observations, not failed 30-second study jobs.",
                "Native raw 20 N wrist abort, held-part gravity, retained grasp and existing success dwell remain unchanged. Finished jobs enter the existing commanded hold on the next 480 Hz servo tick.",
                "All processes initialize at 120 Hz. Explicit native dt changes only during live jobs; 15 Hz policy interface, 480 Hz external servo and 120 Hz sensors, no extra RNG draws or within-job state writes.",
                "Only evidence through each native terminal belongs to the active-prefix outcome. Post-terminal absorbing forces describe commanded-hold mechanics separately.",
                "This is the final prospective common-reference-slew cause test. Compare both480/960Hz native resolutions with their fixed480Hz-servo no-slew controls. No additional rate tuning, compliance or force regulator.",
                "The20mm/s limit applies to the interpolated Cartesian reference before unchanged error clamps. It does not certify actual tool speed or contact force. Requested, interpolated, clamped and final terminal-held targets are recorded separately.",
                "The controller reference is extra hidden actuator state shared across all controllers. The actor vector and previous-action definition remain unchanged; no privileged policy observation is added.",
                "This is an impact diagnostic. Agreement of sampled resolutions alone establishes neither hardware calibration nor a reliable learned recovery method.",
            ],
        )
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if env is not None:
            report["cost"] = env.cost_report()
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
