"""Versioned ordinary/exposure PPO worker; original 120 Hz physics only.

Training-only worker: no evaluation seeds, inherited checkpoint or final-test
access. A launcher must capture exact source/environment provenance before
starting it. Diagnostic mode permits one bounded cohort while physics is open.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("ordinary", "exposure"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--cohorts", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registration", type=Path, default=ROOT / "configs/recovery_teaching_registration_v1.json")
    parser.add_argument("--expected-study-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--expected-charged-transitions", type=float, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--max-minutes", type=float, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.cohorts <= 22 or not 0 < args.max_minutes <= 180:
        parser.error("Use 1..22 cohorts and a positive wall limit of at most 180 minutes")
    if args.diagnostic and (args.cohorts != 1 or args.max_minutes > 30):
        parser.error("A diagnostic is exactly one cohort with at most 30 minutes")
    args.output.mkdir(parents=True, exist_ok=False)
    start_wall = time.monotonic()
    report = {
        "status": "starting", "mode": "train", "arm": args.arm, "seed": args.seed,
        "cohorts": [], "research_result": False, "diagnostic": args.diagnostic,
        "controller": "project_ordinary_ppo_recovery_teaching_v1" if args.arm == "ordinary" else "script_prefix_then_project_exposure_ppo_v1",
        "policy_controller": "project_feedforward_ppo_v1",
        "prefix_controller": "unchanged_ActorRetryController_first_attempt" if args.arm == "exposure" else None,
        "reward_version": "completion_credit_v3", "optimizer_version": "value_unclipped_v4",
        "physics_hz": 120,
        "scope": "Fresh training under a registered ordinary/exposure design; diagnostic until shared physics and launch provenance are accepted.",
    }
    app = env = model = None
    metrics, durations = [], []
    try:
        from assembly_recovery.protocol import sha256, validate_study

        if sha256(ROOT / "configs/study.json") != args.expected_study_sha256:
            raise RuntimeError("Study changed after prelaunch capture")
        if sha256(args.protocol) != args.expected_protocol_sha256 or sha256(args.registration) != args.expected_registration_sha256:
            raise RuntimeError("Launch protocol or teaching registration changed after capture")
        study = json.loads((ROOT / "configs/study.json").read_text())
        protocol = json.loads(args.protocol.read_text())
        registration = json.loads(args.registration.read_text())
        if validate_study(study):
            raise RuntimeError("Existing finite-job implementation gates must pass")
        if args.seed not in registration["training_seeds"] or args.seed not in study["training"]["seeds"]:
            raise ValueError("Only registered training seeds may enter this worker")
        candidate = registration["candidate"]
        if candidate["fault_prefix_fraction"] != 0.5 or candidate["prefix_s"] != 4 or candidate["nominal_fraction"] != 0.25:
            raise ValueError("Worker implements only the registered fixed half-fault/four-second exposure")
        if not args.diagnostic:
            physical = protocol.get("physics_validation", {})
            if protocol.get("launch_permitted") is not True or physical.get("accepted") is not True or physical.get("physics_hz") != 120:
                raise RuntimeError("Substantial training requires accepted shared 120 Hz physics in the launch protocol")
            gate_path = ROOT / physical["evidence_path"]
            if sha256(gate_path) != physical["evidence_sha256"]:
                raise RuntimeError("Accepted physics evidence changed")
            if not protocol.get("behavior_source_sha256"):
                raise RuntimeError("Substantial training requires frozen behavior source hashes")
        for name, digest in protocol.get("behavior_source_sha256", {}).items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Frozen behavior source changed: " + name)
        report.update(
            study_sha256=args.expected_study_sha256, protocol_sha256=args.expected_protocol_sha256,
            registration_sha256=args.expected_registration_sha256,
            requested_cohorts=args.cohorts, num_envs=args.num_envs,
            budget_selection="Final predeclared cohort and charged cost; never selected by evaluation score.",
            optimizer_accounting="Same v4 update schedule and hyperparameters; actual active samples/optimizer steps reported, not identical FLOPs.",
        )
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.completion_env import PegCompletionEnv
        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.recovery_teaching_v1 import (
            exposure_mask,
            make_prefix_action_targets,
            mask_scripted_rollout,
            prefix_action_at_step,
            scripted_prefix_mask,
        )
        from assembly_recovery.retry_controller import RetrySettings
        from assembly_recovery.study_ppo import StudyPolicy
        from assembly_recovery.training_cases import uniform_training_cases
        from assembly_recovery.unclipped_ppo import ppo_update

        torch.set_num_threads(1)
        cases = uniform_training_cases(study, args.seed, 0, args.num_envs)
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        if abs(cfg.sim.dt - 1 / 120) > 1e-12 or cfg.decimation != 8:
            raise ValueError("This version implements exactly native 120 Hz physics and 15 Hz policy updates")
        c = study["job_protocol"]
        criteria = JobCriteria(
            physics_dt=cfg.sim.dt, deadline_s=c["deadline_seconds"],
            seated_dwell_s=c["seated_dwell_seconds"], force_budget_n=c["force_budget_n"],
        )
        if criteria.deadline_s != 30 or criteria.seated_dwell_s != 0.5 or criteria.force_budget_n != 20:
            raise ValueError("The frozen 30-second/0.5-second/20 N job definition is required")
        env = PegCompletionEnv(cfg, criteria=criteria, cases=cases, nominal=study["fault_support"]["nominal"], audit=False)
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        settings_path = ROOT / "configs/retry_unload_realign_v1.json"
        settings = RetrySettings(**json.loads(settings_path.read_text()))
        if settings.hold_s != 1 or settings.first_attempt_s != 4 or abs(env.step_dt * 60 - 4) > 1e-9:
            raise ValueError("The unchanged first attempt must hold one second and hand off at four seconds")
        report.update(
            criteria=asdict(criteria), prefix_controls=60,
            controller_settings=asdict(settings), controller_settings_sha256=sha256(settings_path),
            controller_position_bounds=list(cfg.ctrl.pos_action_bounds),
            controller_seated_height_m=env.geometry.seated_fingertip_above_hole_top_m,
        )
        optimizer = None
        for cohort in range(args.cohorts):
            if time.monotonic() - start_wall >= args.max_minutes * 60:
                raise TimeoutError("Worker wall-clock budget expired before next cohort")
            env.fault_cases = uniform_training_cases(study, args.seed, cohort, args.num_envs)
            selected_exposure = exposure_mask(
                env.fault_cases, seed=args.seed, cohort=cohort, training_seeds=registration["training_seeds"],
            )
            if args.arm == "ordinary":
                selected_exposure = [False] * args.num_envs
            exposed = torch.tensor(selected_exposure, dtype=torch.bool, device=env.device)
            init_start = time.monotonic()
            observation, _ = env.reset()
            torch.cuda.synchronize()
            init_wall = time.monotonic() - init_start
            validity = env.initialization_report()
            if not all(v["valid"] for v in validity):
                report["failed_cohort"] = {"cohort": cohort, "initialization": validity, "cases": env.fault_cases}
                raise RuntimeError("Invalid initialization; retain the failed cohort and stop")
            if model is None:
                model = StudyPolicy(observation["policy"].shape[-1], observation["critic"].shape[-1]).to(env.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                report["policy_spec"] = model.spec
                initial_path = args.output / "initial_model.pt"
                torch.save({"model": model.state_dict(), "spec": model.spec, "seed": args.seed}, initial_path)
                report["initial_model_sha256"] = sha256(initial_path)
            hold_action, insert_action = make_prefix_action_targets(
                observation["policy"], position_bounds=cfg.ctrl.pos_action_bounds,
                seated_height_m=env.geometry.seated_fingertip_above_hole_top_m,
            )
            env.begin_jobs(f"train-{args.arm}-{args.seed}-{cohort}")
            rollout, diagnostic_rows = [], []
            selected = {}
            for index, case in enumerate(env.fault_cases):
                selected.setdefault((case["bin_id"], selected_exposure[index]), index)
            diagnostic_indices = list(selected.values())
            diagnostic_ids = torch.tensor(diagnostic_indices, device=env.device)
            job_returns = torch.zeros(args.num_envs, dtype=torch.float64, device=env.device)
            upstream_returns, credit_returns = torch.zeros_like(job_returns), torch.zeros_like(job_returns)
            done_counts = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
            action_counts = torch.zeros((3,), dtype=torch.int64, device=env.device)
            start = interval_start = time.monotonic()
            for step in range(450):
                if time.monotonic() - start_wall >= args.max_minutes * 60:
                    raise TimeoutError("Worker wall-clock budget expired inside a cohort")
                with torch.no_grad():
                    policy_action, log, value, normalized = model.act(observation)
                    script_owned = scripted_prefix_mask(exposed, step=step, prefix_controls=60) & env.tensor_jobs.active
                    if step < 60 and args.arm == "exposure":
                        script_action = prefix_action_at_step(
                            hold_action, insert_action, step=step, step_dt=env.step_dt,
                            hold_s=settings.hold_s, first_attempt_s=settings.first_attempt_s,
                        )
                        requested_action = torch.where(script_owned[:, None], script_action, policy_action)
                    else:
                        requested_action = policy_action
                    next_obs, reward, done, _, extra = env.step(requested_action)
                    next_value = model.value(model.critic_norm(next_obs["critic"]))
                    done_counts += done
                    action_counts += torch.stack((extra["valid"].sum(), script_owned.sum(), (extra["valid"] & ~script_owned).sum()))
                    job_returns += reward.double() * 0.995 ** step
                    upstream_returns += extra["upstream_job_reward"].double() * 0.995 ** step
                    credit_returns += extra["completion_credit"].double() * 0.995 ** step
                    diagnostic_rows.append(torch.cat((
                        env.held_pos - env.fixed_pos, env.held_quat, env.force_sensor_world[:, :3],
                        env.actions, observation["policy"], extra["raw_reward"][:, None], reward[:, None],
                        extra["valid"][:, None], extra["completion_credit"][:, None],
                        extra["upstream_job_reward"][:, None], script_owned[:, None],
                    ), dim=-1)[diagnostic_ids].clone())
                    rollout.append({
                        "actor": normalized["policy"], "critic": normalized["critic"],
                        "raw_actor": observation["policy"], "raw_critic": observation["critic"],
                        "action": policy_action, "log_prob": log, "value": value, "next_value": next_value,
                        "reward": reward, "done": done, "valid": extra["valid"], "scripted": script_owned,
                    })
                    observation = next_obs
                if (step + 1) % 128 == 0 or step == 449:
                    torch.cuda.synchronize()
                    duration = time.monotonic() - interval_start
                    count = 128 if (step + 1) % 128 == 0 else 66
                    durations.append({
                        "cohort": cohort, "end_control_step": step + 1,
                        "rollout_wall_s": duration, "transitions_per_s": args.num_envs * count / duration,
                    })
                    data = {key: torch.stack([row[key] for row in rollout]) for key in rollout[0]}
                    script_mask = data.pop("scripted")
                    physical_active_samples = int(data["valid"].sum())
                    script_active_samples = int(script_mask.sum())
                    data = mask_scripted_rollout(data, script_mask)
                    update_start = time.monotonic()
                    metric = ppo_update(model, optimizer, data)
                    torch.cuda.synchronize()
                    metric.update(
                        wall_s=time.monotonic() - update_start, cohort=cohort, end_control_step=step + 1,
                        physical_active_samples=physical_active_samples, scripted_active_samples=script_active_samples,
                    )
                    metrics.append(metric)
                    rollout = []
                    interval_start = time.monotonic()
                    print(json.dumps({**durations[-1], "arm": args.arm, "active_policy_samples": metric["active_samples"]}), flush=True)
            jobs = env.end_jobs()
            witness = env.contact_witness.results(jobs)
            if not bool((done_counts == 1).all()):
                raise RuntimeError("Every finite job must emit exactly one terminal")
            counts = action_counts.cpu().tolist()
            if counts[0] != counts[1] + counts[2]:
                raise RuntimeError("Physical action ownership does not partition active transitions")
            prefix_terminals = [selected_exposure[i] and job["elapsed_s"] <= 4 for i, job in enumerate(jobs)]
            exposure_stalls = [
                selected_exposure[i] and row["witnessed_failure_step"] is not None and row["witnessed_failure_step"] <= 480
                for i, row in enumerate(witness)
            ]
            item = {
                "cohort": cohort, "cases": env.fault_cases, "initialization": validity,
                "initialization_wall_s": init_wall, "rollout_and_optimizer_wall_s": time.monotonic() - start,
                "jobs": jobs, "recovery_witness": witness, "exposure_selected": selected_exposure,
                "physical_active_controls": counts[0], "scripted_active_controls": counts[1], "policy_active_controls": counts[2],
                "selected_exposure_jobs": sum(selected_exposure),
                "exposed_witnessed_stalls_by_handoff": sum(exposure_stalls),
                "all_witnessed_stalls": sum(row["witnessed_failure_step"] is not None for row in witness),
                "prefix_script_completions": sum(prefix_terminals[i] and job["success"] for i, job in enumerate(jobs)),
                "prefix_script_failures": sum(prefix_terminals[i] and not job["success"] for i, job in enumerate(jobs)),
                "discounted_job_returns": job_returns.cpu().tolist(),
                "upstream_discounted_job_returns": upstream_returns.cpu().tolist(),
                "completion_credit_discounted_returns": credit_returns.cpu().tolist(),
                "policy_action_std": model.log_std.clamp(-5, 2).exp().detach().cpu().tolist(),
                "training_endpoint_scope": "Changing stochastic policies; contact-witness training statistics are not replay-verified frozen evaluation or a guarantee that every exposed job stalled.",
            }
            export_start = time.monotonic()
            torch.save({
                "case_indices": diagnostic_indices, "controls": torch.stack(diagnostic_rows).cpu(),
                "columns": ["part_relative_xyz:3", "part_quaternion:4", "control_endpoint_raw_wrist_xyz:3",
                            "applied_actions:7", "actor_observation_before:24", "raw_reward:1", "job_reward:1",
                            "active_before:1", "completion_credit:1", "upstream_job_reward:1", "script_owned:1"],
            }, args.output / f"diagnostics_{cohort}.pt")
            item["diagnostic_export_wall_s"] = time.monotonic() - export_start
            report["cohorts"].append(item)
            report["cost"] = env.cost_report()
            at_target = cohort + 1 == args.cohorts
            exact_cost = report["cost"]["charged_control_equivalent_transitions"] == args.expected_charged_transitions
            within_wall_limit = time.monotonic() - start_wall < args.max_minutes * 60
            final = at_target and exact_cost and within_wall_limit
            numpy_state = np.random.get_state()
            checkpoint = {
                "status": "budget_complete" if final else "partial",
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "spec": model.spec, "cost": report["cost"],
                "completed_cohorts": cohort + 1, "requested_cohorts": args.cohorts, "seed": args.seed, "arm": args.arm,
                "cpu_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
                "success_prediction_latch": bool(env.success_latch), "protocol_sha256": report["protocol_sha256"],
                "registration_sha256": report["registration_sha256"],
                "reward_version": "completion_credit_v3", "optimizer_version": "value_unclipped_v4",
                "exposure_selector": "stateless_recovery_teaching_v1_seed_cohort_bin",
                "diagnostic": args.diagnostic, "physics_hz": 120,
                "scope": "Fresh registered training. Complete optimizer/RNG capture; simulator restart/resume equivalence is not verified.",
            }
            path = args.output / ("budget_checkpoint.pt" if final else f"partial_cohort_{cohort + 1}.pt")
            torch.save(checkpoint, path)
            if not within_wall_limit:
                raise TimeoutError('Worker wall-clock budget expired; saved weights remain partial')
            if at_target and not exact_cost:
                raise RuntimeError("Charged initialization/rollout cost differs from registration; weights remain partial")
            if final:
                reloaded = torch.load(path, map_location=env.device, weights_only=True)
                rebuilt = StudyPolicy(
                    reloaded["spec"]["actor_size"], reloaded["spec"]["critic_size"],
                    reloaded["spec"]["action_size"], reloaded["spec"]["widths"],
                ).to(env.device)
                rebuilt.load_state_dict(reloaded["model"])
                with torch.no_grad():
                    equal = torch.equal(model.act(observation, deterministic=True)[0], rebuilt.act(observation, deterministic=True)[0])
                report["checkpoint_validation"] = {
                    "finite": all(bool(torch.isfinite(tensor).all()) for tensor in reloaded["model"].values()),
                    "identical_loaded_actions": equal, "sha256": sha256(path),
                }
                if not report["checkpoint_validation"]["finite"] or not equal:
                    raise RuntimeError("Saved checkpoint failed finite/action parity validation")
            (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        report.update(status="completed", optimizer_updates=metrics, throughput_intervals=durations)
        report["scope_and_limitations"] = [
            "Every nominal job is direct policy practice; fixed prefixes are on selected training faults only.",
            "Prefix actions, rewards and scripted completions never enter PPO losses or normalization statistics.",
            "All initialization, script, active policy and absorbing simulator work is charged.",
            "Scripted contact exposure need not yield an actual stall; realized training witness counts are reported.",
            "Original 120 Hz native simulation only; diagnostics do not resolve the known timestep sensitivity.",
            "No development/final evaluation, warm start, simulator-state recovery, or hardware-transfer claim.",
        ]
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if env is not None:
            report["cost"] = env.cost_report()
        raise
    finally:
        report["optimizer_updates"] = metrics
        report["throughput_intervals"] = durations
        report["elapsed_worker_wall_s"] = time.monotonic() - start_wall
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
