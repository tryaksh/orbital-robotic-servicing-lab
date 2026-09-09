"""Confirm the frozen prefix assay on registered development seeds; v1 stays frozen."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller", choices=("learned", "retry", "continue"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(10071, 10072), required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "controller": args.controller, "seed": args.seed,
               "research_result": False, "reward_version": "completion_credit_v3", "diagnostic_protocol": "failure_prefix_confirmation_v2", "training_run_status": "completed", "checkpoint_selection": "Final registered 22-cohort budget checkpoint; no selection by score"}
    app = env = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.completion_env import PegCompletionEnv as PegTrainingEnv
        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.faults import development_cases
        from assembly_recovery.policy_diagnosis import capture_normal_draws
        from assembly_recovery.post_stall import post_stall_completion, summarize_endpoints
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
        env = PegTrainingEnv(cfg, criteria=criteria, cases=cases, nominal=study["fault_support"]["nominal"], audit=True)
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        observation, _ = env.reset()
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
                       seated_height_m=env.geometry.seated_fingertip_above_hole_top_m, retry=args.controller == "retry", settings=settings)
                       for row in observation["policy"].cpu().tolist()]
        if abs(env.step_dt * 60 - 4.) > 1e-9 or settings.first_attempt_s != 4.:
            raise ValueError("The fixed prefix must equal the unchanged four-second first-attempt interval")
        report.update(controller_position_bounds=list(cfg.ctrl.pos_action_bounds),
                      controller_seated_height_m=env.geometry.seated_fingertip_above_hole_top_m,
                      controller_step_dt=env.step_dt, controller_settings=asdict(settings),
                      prefix_controls=60, prefix_physics_steps=480, prefix_seconds=4.,
                      controller_schedule="Unchanged scripted insertion for controls 0..59, then the named controller through the original 30-second job deadline.",
                      policy_phase_label="Frozen learned Gaussian mean" if args.controller == "learned" else "Unchanged scripted " + args.controller)
        env.begin_jobs("diagnosis")
        names = ("actor_before", "mean_action", "requested_action", "applied_action", "target_xyz", "tool_xyz",
                 "part_relative_xyz", "part_quat", "raw_reward", "job_reward", "active", "dead_zone", "completion_credit", "upstream_job_reward", "critic_before", "normalized_critic_before", "critic_value")
        arrays = {k: [] for k in names}
        terms, draws, rng_states, phases = {}, [], [], []
        done_counts = torch.zeros(28, dtype=torch.int64, device=env.device)
        returns = torch.zeros(28, dtype=torch.float64, device=env.device)
        for step in range(450):
            with torch.no_grad():
                mean, _, critic_value, normalized = model.act(observation, deterministic=True)
                if step < 60 or args.controller != "learned":
                    decisions = [ctrl.act(row, step * env.step_dt) for ctrl, row in
                                 zip(controllers, observation["policy"].cpu().tolist(), strict=True)]
                    action = torch.tensor([d[0] for d in decisions], device=env.device)
                    phases.append([d[1]["phase"] for d in decisions])
                else:
                    action = mean
                    phases.append(["learned_mean"] * 28)
                rng_states.append(torch.cuda.get_rng_state(env.device))
                step_draws = []
                with capture_normal_draws(step_draws):
                    next_obs, reward, done, _, extra = env.step(action)
                # Pinned FORGE: four draws per native step, shapes (N,3),(N,3),(N,),(N,3).
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
            if (step + 1) % 128 == 0:
                print(json.dumps({"controller": args.controller, "control_step": step + 1}), flush=True)
        jobs = env.end_jobs()
        witness = env.contact_witness.results(jobs)
        if not bool((done_counts == 1).all()):
            raise RuntimeError("Expected exactly one terminal per request")
        artifact = {k: torch.stack(v).cpu() for k, v in arrays.items()}
        artifact.update(initial=initial, physics=env.audit_buffer.cpu(), sensor_draws=torch.cat(draws).cpu(),
                        cuda_rng_before_steps=torch.stack(rng_states),
                        reward_terms={k: torch.stack(v).cpu() for k, v in terms.items()})
        torch.save(artifact, args.output / "trajectory.pt")
        report.update(jobs=jobs, recovery_witness=witness, discounted_job_returns=returns.cpu().tolist(),
                      cost=env.cost_report(), cpu_replay=replay_physics(artifact["physics"], criteria, jobs, "diagnosis", witness),
                      script_phases=phases, status="completed",
                      scope_and_limitations=["All 28 development requests retained, including prefix completions, force aborts and timeouts. No optimization or final-test use.",
                          "Fixed four-second action-only prefix; handoff does not depend on evaluator contact or simulator truth.",
                          "The retry arm is unchanged throughout; the shared prefix precedes its first permitted retry. This separate recovery assay does not replace the original 84-case competence gate.",
                          "Upstream rewards retain v2 cutoff; v3 adds declared completion credit at the native-terminal control notification. Original v2 results remain unchanged."])
        endpoints = [post_stall_completion(job, w, criteria) for job, w in zip(jobs, witness, strict=True)]
        report.update(post_stall_endpoint=endpoints, post_stall_summary=summarize_endpoints(endpoints, jobs))
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
