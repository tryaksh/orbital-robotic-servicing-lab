"""Bounded uniform-fault pilot or development evaluation of budget-selected weights."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--cohorts", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-study-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-charged-transitions", type=float, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "mode": args.mode, "seed": args.seed, "cohorts": [],
              "controller": "project_uniform_fault_ppo_v1", "research_result": False,
              "scope": "Bounded learning pilot/development evaluation, not an equal-cost method comparison or final test."}
    app = env = model = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.faults import development_cases
        from assembly_recovery.protocol import sha256, validate_study
        from assembly_recovery.study_ppo import StudyPolicy, ppo_update
        from assembly_recovery.training_cases import uniform_training_cases
        from assembly_recovery.training_env import PegTrainingEnv

        torch.set_num_threads(1)
        if sha256(ROOT / "configs/study.json") != args.expected_study_sha256 or sha256(args.protocol) != args.expected_protocol_sha256:
            raise RuntimeError("Protocol/config changed after prelaunch capture")
        study = json.loads((ROOT / "configs/study.json").read_text())
        if validate_study(study):
            raise RuntimeError("All scientific gates must pass before the pilot/evaluation path")
        training = args.mode == "train"
        if training:
            cases = uniform_training_cases(study, args.seed, 0, args.num_envs)
        else:
            cases = development_cases(study, args.seed)
            args.num_envs, args.cohorts = len(cases), 1
            if not args.checkpoint or sha256(args.checkpoint) != args.checkpoint_sha256:
                raise ValueError("Evaluation checkpoint must match its prelaunch hash")
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        c = study["job_protocol"]
        criteria = JobCriteria(physics_dt=cfg.sim.dt, deadline_s=c["deadline_seconds"],
                               seated_dwell_s=c["seated_dwell_seconds"], force_budget_n=c["force_budget_n"])
        env = PegTrainingEnv(cfg, criteria=criteria, cases=cases, nominal=study["fault_support"]["nominal"], audit=not training)
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        report.update(num_envs=args.num_envs, requested_cohorts=args.cohorts,
                      budget_selection="Final predeclared cohort, never development/test performance.",
                      protocol_sha256=sha256(args.protocol))
        optimizer = None
        metrics, durations = [], []
        for cohort in range(args.cohorts):
            if training:
                env.fault_cases = uniform_training_cases(study, args.seed, cohort, args.num_envs)
            init_start = time.monotonic()
            observation, _ = env.reset()
            torch.cuda.synchronize()
            init_wall = time.monotonic() - init_start
            validity = env.initialization_report()
            if not all(v["valid"] for v in validity):
                report["failed_cohort"] = {"cohort": cohort, "initialization": validity, "cases": env.fault_cases}
                raise RuntimeError("A sampled pilot initialization is invalid; retain the failure and stop training")
            if model is None:
                model = StudyPolicy(observation["policy"].shape[-1], observation["critic"].shape[-1]).to(env.device)
                if training:
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                else:
                    loaded = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
                    if loaded["status"] != "budget_complete" or loaded["spec"] != model.spec:
                        raise ValueError("Only a complete matching budget-selected checkpoint may be evaluated")
                    model.load_state_dict(loaded["model"])
                    model.eval()
                    report["checkpoint_sha256"] = args.checkpoint_sha256
                    report["training_cost"] = loaded["cost"]
                report["policy_spec"] = model.spec
            env.begin_jobs(f"{args.mode}-{args.seed}-{cohort}")
            rollout = []
            selected = {}
            for index, case in enumerate(env.fault_cases):
                selected.setdefault(case["bin_id"], index)
            diagnostic_indices = list(selected.values()) if training else list(range(args.num_envs))
            diagnostic_ids = torch.tensor(diagnostic_indices, device=env.device)
            diagnostic_rows = []
            job_returns = torch.zeros(args.num_envs, dtype=torch.float64, device=env.device)
            done_counts = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
            start = interval_start = time.monotonic()
            for step in range(450):
                with torch.no_grad():
                    action, log, value, normalized = model.act(observation, deterministic=not training)
                    next_obs, reward, done, truncated, extra = env.step(action)
                    next_value = model.value(model.critic_norm(next_obs["critic"]))
                    done_counts += done
                    job_returns += reward.to(torch.float64) * 0.995 ** step
                    diagnostic_rows.append(torch.cat((env.held_pos - env.fixed_pos, env.held_quat,
                        env.force_sensor_world[:, :3], env.actions, observation["policy"],
                        extra["raw_reward"][:, None], reward[:, None], extra["valid"][:, None]), dim=-1)[diagnostic_ids].clone())
                    if training:
                        rollout.append({"actor": normalized["policy"], "critic": normalized["critic"],
                                        "raw_actor": observation["policy"], "raw_critic": observation["critic"],
                                        "action": action, "log_prob": log, "value": value, "next_value": next_value,
                                        "reward": reward, "done": done, "valid": extra["valid"]})
                    observation = next_obs
                if (step + 1) % 128 == 0 or step == 449:
                    torch.cuda.synchronize()
                    duration = time.monotonic() - interval_start
                    count = 128 if (step + 1) % 128 == 0 else 66
                    durations.append({"cohort": cohort, "end_control_step": step + 1,
                                      "rollout_wall_s": duration, "transitions_per_s": args.num_envs * count / duration})
                    if training:
                        data = {k: torch.stack([r[k] for r in rollout]) for k in rollout[0]}
                        update_start = time.monotonic()
                        metric = ppo_update(model, optimizer, data)
                        torch.cuda.synchronize()
                        metric["wall_s"] = time.monotonic() - update_start
                        metrics.append(metric)
                        rollout = []
                    interval_start = time.monotonic()
                    print(json.dumps(durations[-1]), flush=True)
            jobs = env.end_jobs()
            if not bool((done_counts == 1).all()):
                raise RuntimeError("Each finite job must emit exactly one done")
            item = {"cohort": cohort, "cases": env.fault_cases, "initialization": validity,
                    "initialization_wall_s": init_wall, "rollout_and_optimizer_wall_s": time.monotonic() - start,
                    "jobs": jobs, "recovery_witness": env.contact_witness.results(jobs)}
            item["discounted_job_returns"] = job_returns.cpu().tolist()
            export_start = time.monotonic()
            diagnostic = {"case_indices": diagnostic_indices, "controls": torch.stack(diagnostic_rows).cpu(),
                          "columns": ["part_relative_xyz:3", "part_quaternion:4", "control_endpoint_raw_wrist_xyz:3",
                                      "applied_actions:7", "actor_observation_before:24", "raw_reward:1", "job_reward:1", "active_before:1"]}
            if not training:
                from scripts.probe_training import replay_physics

                diagnostic["physics"] = env.audit_buffer.cpu()
                item["cpu_replay"] = replay_physics(diagnostic["physics"], criteria, jobs,
                                      f"{args.mode}-{args.seed}-{cohort}", item["recovery_witness"])
            torch.save(diagnostic, args.output / f"diagnostics_{cohort}.pt")
            item["diagnostic_export_wall_s"] = time.monotonic() - export_start
            report["cohorts"].append(item)
            report["cost"] = env.cost_report()
            if training:
                at_target = cohort + 1 == args.cohorts
                exact_cost = report["cost"]["charged_control_equivalent_transitions"] == args.expected_charged_transitions
                final = at_target and exact_cost
                checkpoint = {"status": "budget_complete" if final else "partial", "model": model.state_dict(),
                              "optimizer": optimizer.state_dict(), "spec": model.spec, "cost": report["cost"],
                              "completed_cohorts": cohort + 1, "requested_cohorts": args.cohorts, "seed": args.seed,
                              "cpu_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                              "success_prediction_latch": bool(env.success_latch), "protocol_sha256": report["protocol_sha256"],
                              "scope": "Uniform-fault pilot. Resume equivalence is not yet verified."}
                path = args.output / ("budget_checkpoint.pt" if final else f"partial_cohort_{cohort + 1}.pt")
                torch.save(checkpoint, path)
                if at_target and not exact_cost:
                    raise RuntimeError("Initialization cost differs from the predeclared charged budget; weights remain partial")
                if final:
                    reloaded = torch.load(path, map_location=env.device, weights_only=True)
                    rebuilt = StudyPolicy(reloaded["spec"]["actor_size"], reloaded["spec"]["critic_size"],
                                          reloaded["spec"]["action_size"], reloaded["spec"]["widths"]).to(env.device)
                    rebuilt.load_state_dict(reloaded["model"])
                    with torch.no_grad():
                        equal = torch.equal(model.act(observation, deterministic=True)[0], rebuilt.act(observation, deterministic=True)[0])
                    report["checkpoint_validation"] = {"finite": all(bool(torch.isfinite(v).all()) for v in reloaded["model"].values()),
                                                        "identical_loaded_actions": equal, "sha256": sha256(path)}
                    if not report["checkpoint_validation"]["finite"] or not equal:
                        raise RuntimeError("Saved checkpoint validation failed")
            (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        report.update(status="completed", optimizer_updates=metrics, throughput_intervals=durations)
        report["scope_and_limitations"] = [
            "One training seed and development-only evaluation; no adaptive-versus-uniform comparison or final-test claim.",
            "Project feedforward PPO, distinct from the upstream recurrent reference.",
            "All rollout slots and initialization physics are charged; failed or absorbing slots cannot contribute additional PPO samples.",
            "Controller-independent recovery witness requires contact failure, physical withdrawal and later completed dwell.",
            "No hardware transfer, camera perception or force-certified safety claim."]
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
