"""Representative capacity-only PPO worker; frozen historical workers stay unchanged."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def finite_tensor_tree(value):
    """Check optimizer state as well as model weights after a real disk reload."""
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensor_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensor_tree(item) for item in value)
    return True


def reloaded_actions_match(model, loaded, observation, device):
    """Checkpoint inspection must not consume the following cohort's RNG stream."""
    import torch

    from assembly_recovery.study_ppo import StudyPolicy

    target = torch.device(device)
    devices = [target] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        spec = loaded["spec"]
        rebuilt = StudyPolicy(spec["actor_size"], spec["critic_size"], spec["action_size"], spec["widths"]).to(target)
        rebuilt.load_state_dict(loaded["model"])
        with torch.no_grad():
            return torch.equal(model.act(observation, deterministic=True)[0], rebuilt.act(observation, deterministic=True)[0])


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from assembly_recovery.protocol import sha256, write_json

    report = {"schema": 1, "version": "training_capacity_v1", "status": "starting",
              "controller": "project_uniform_fault_ppo_value_unclipped_v4_capacity_only",
              "research_result": False, "cohorts": [], "optimizer_updates": [], "throughput_intervals": []}
    app = env = None
    start = time.monotonic()
    try:
        if sha256(args.manifest) != args.manifest_sha256:
            raise RuntimeError("Prelaunch manifest changed")
        manifest = json.loads(args.manifest.read_text())
        for name, digest in manifest["source_hashes"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Source changed after exact snapshot: " + name)
        specification = manifest["specification"]
        registration = json.loads((ROOT / "configs/training_capacity_v1.json").read_text())
        study = json.loads((ROOT / "configs/study.json").read_text())
        n, seed = specification["num_envs"], specification["seed"]
        report.update(specification=specification, manifest_sha256=args.manifest_sha256,
                      scope_and_limitations=registration["scope_and_limitations"])
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.completion_env import PegCompletionEnv
        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.study_ppo import StudyPolicy
        from assembly_recovery.training_cases import uniform_training_cases
        from assembly_recovery.unclipped_ppo import ppo_update

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=n)
        cfg.seed = seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        contract = study["job_protocol"]
        criteria = JobCriteria(physics_dt=cfg.sim.dt, deadline_s=contract["deadline_seconds"],
                               seated_dwell_s=contract["seated_dwell_seconds"], force_budget_n=contract["force_budget_n"])
        if cfg.sim.dt != 1 / 120 or cfg.decimation != 8:
            raise RuntimeError("Capacity benchmark requires frozen 120 Hz physics and 15 Hz policy")
        env = PegCompletionEnv(cfg, criteria=criteria, cases=uniform_training_cases(study, seed, 0, n),
                               nominal=study["fault_support"]["nominal"], audit=False)
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        torch.cuda.synchronize()
        report["app_and_environment_startup_wall_s"] = time.monotonic() - start
        report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(), "python": sys.version}
        torch.cuda.reset_peak_memory_stats()
        model = optimizer = None
        for cohort in range(specification["cohorts"]):
            cohort_start = time.monotonic()
            env.fault_cases = uniform_training_cases(study, seed, cohort, n)
            init_start = time.monotonic()
            observation, _ = env.reset()
            torch.cuda.synchronize()
            initialization_wall = time.monotonic() - init_start
            validity = env.initialization_report()
            if not all(item["valid"] for item in validity):
                report["failed_cohort"] = {"cohort": cohort, "cases": env.fault_cases, "initialization": validity}
                raise RuntimeError("Invalid sampled initialization; retain failed request and stop")
            if model is None:
                model = StudyPolicy(observation["policy"].shape[-1], observation["critic"].shape[-1]).to(env.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=registration["learning_rate"])
                report["policy_spec"] = model.spec
            env.begin_jobs(f"capacity-{n}-{seed}-{cohort}")
            selected = {}
            for i, case in enumerate(env.fault_cases):
                selected.setdefault(case["bin_id"], i)
            diagnostic_ids = torch.tensor(list(selected.values()), device=env.device)
            diagnostic_rows, active_counts, rollout = [], [], []
            done_counts = torch.zeros(n, dtype=torch.int64, device=env.device)
            job_returns = torch.zeros(n, dtype=torch.float64, device=env.device)
            all_terminal_at = None
            all_terminal_control = None
            # One scalar read is piggybacked on each existing PPO boundary, avoiding a new per-step device sync.
            interval_start = time.monotonic()
            interval_first = 0
            cohort_rollout = cohort_optimizer = trailing_wall = 0.0
            for step in range(specification["controls_per_cohort"]):
                with torch.no_grad():
                    action, log, value, normalized = model.act(observation)
                    next_obs, reward, done, _, extra = env.step(action)
                    next_value = model.value(model.critic_norm(next_obs["critic"]))
                    done_counts += done
                    job_returns += reward.double() * 0.995 ** step
                    active_counts.append(extra["valid"].sum())
                    diagnostic_rows.append(torch.cat((env.held_pos - env.fixed_pos, env.held_quat,
                        env.force_sensor_world[:, :3], env.actions, observation["policy"],
                        extra["raw_reward"][:, None], reward[:, None], extra["valid"][:, None],
                        extra["completion_credit"][:, None], extra["upstream_job_reward"][:, None]), dim=-1)[diagnostic_ids].clone())
                    rollout.append({"actor": normalized["policy"], "critic": normalized["critic"],
                                    "raw_actor": observation["policy"], "raw_critic": observation["critic"],
                                    "action": action, "log_prob": log, "value": value, "next_value": next_value,
                                    "reward": reward, "done": done, "valid": extra["valid"]})
                    observation = next_obs
                if (step + 1) % registration["ppo_horizon"] == 0 or step + 1 == specification["controls_per_cohort"]:
                    torch.cuda.synchronize()
                    rollout_wall = time.monotonic() - interval_start
                    cohort_rollout += rollout_wall
                    if all_terminal_at is not None:
                        trailing_wall += rollout_wall
                    data = {key: torch.stack([row[key] for row in rollout]) for key in rollout[0]}
                    active = int(data["valid"].sum())
                    update_start = time.monotonic()
                    metric = ppo_update(model, optimizer, data, mini_epochs=registration["ppo_mini_epochs"],
                                        minibatch=registration["ppo_minibatch"])
                    torch.cuda.synchronize()
                    optimizer_wall = time.monotonic() - update_start
                    cohort_optimizer += optimizer_wall
                    if all_terminal_at is not None:
                        trailing_wall += optimizer_wall
                    interval = {"cohort": cohort, "first_control_step": interval_first + 1,
                                "end_control_step": step + 1, "rollout_wall_s": rollout_wall,
                                "optimizer_wall_s": optimizer_wall,
                                "allocated_transitions": n * (step + 1 - interval_first),
                                "active_samples": active, "absorbing_samples": n * (step + 1 - interval_first) - active}
                    interval["allocated_rollout_transitions_per_s"] = interval["allocated_transitions"] / rollout_wall
                    interval["active_samples_per_rollout_optimizer_s"] = active / (rollout_wall + optimizer_wall)
                    metric.update(cohort=cohort, end_control_step=step + 1, wall_s=optimizer_wall)
                    report["optimizer_updates"].append(metric)
                    report["throughput_intervals"].append(interval)
                    if all_terminal_at is None and not bool(env.tensor_jobs.active.any()):
                        all_terminal_at = time.monotonic()
                        all_terminal_control = step + 1
                    report["cost"] = env.cost_report()
                    report["current_cohort"] = cohort
                    report["last_completed_control_step"] = step + 1
                    write_json(args.output / "progress.json", report)
                    print(json.dumps(interval), flush=True)
                    rollout = []
                    del data
                    interval_first = step + 1
                    interval_start = time.monotonic()
            jobs = env.end_jobs()
            if not bool((done_counts == 1).all()):
                raise RuntimeError("Each requested job must emit exactly one terminal")
            count_rows = torch.stack(active_counts).cpu().tolist()
            first_empty = next((i + 1 for i, count in enumerate(count_rows) if count == 0), None)
            item = {"cohort": cohort, "cases": env.fault_cases, "initialization": validity, "jobs": jobs,
                    "recovery_witness": env.contact_witness.results(jobs), "done_counts": done_counts.cpu().tolist(),
                    "discounted_job_returns": job_returns.cpu().tolist(), "active_count_by_control": count_rows,
                    "initialization_wall_s": initialization_wall, "rollout_wall_s": cohort_rollout,
                    "optimizer_wall_s": cohort_optimizer,
                    "scheduling": {"first_fully_absorbing_control": first_empty,
                        "avoidable_trailing_control_slots": 0 if first_empty is None else n * (451 - first_empty),
                        "all_terminal_observed_at_ppo_boundary": all_terminal_control,
                        "fully_absorbing_boundary_intervals_wall_s": trailing_wall,
                        "timing_scope": "Conservative measured lower bound from complete PPO intervals after an all-terminal boundary; no interpolation within a partially active interval.",
                        "partial_slot_reuse_enabled": False}}
            export_start = time.monotonic()
            diagnostic_path = args.output / f"diagnostics_{cohort}.pt"
            torch.save({"case_indices": list(selected.values()), "controls": torch.stack(diagnostic_rows).cpu(),
                        "columns": ["part_relative_xyz:3", "part_quaternion:4", "control_endpoint_raw_wrist_xyz:3",
                            "applied_actions:7", "actor_observation_before:24", "raw_reward:1", "job_reward:1",
                            "active_before:1", "completion_credit:1", "upstream_job_reward:1"]}, diagnostic_path)
            report["cost"] = env.cost_report()
            checkpoint_path = args.output / f"benchmark_cohort_{cohort + 1}.pt"
            torch.save({"status": "engineering_benchmark_only", "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "spec": model.spec, "cost": report["cost"],
                        "seed": seed, "completed_cohorts": cohort + 1, "manifest_sha256": args.manifest_sha256,
                        "cpu_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                        "success_prediction_latch": bool(env.success_latch), "research_training_initializer": False}, checkpoint_path)
            loaded = torch.load(checkpoint_path, map_location=env.device, weights_only=True)
            reloaded_optimizer = torch.optim.Adam(model.parameters(), lr=registration["learning_rate"])
            reloaded_optimizer.load_state_dict(loaded["optimizer"])
            equal = reloaded_actions_match(model, loaded, observation, env.device)
            diagnostics = torch.load(diagnostic_path, map_location="cpu", weights_only=True)
            validation = {"model_and_optimizer_finite": finite_tensor_tree(loaded), "identical_loaded_actions": equal,
                          "optimizer_state_nonempty": bool(reloaded_optimizer.state_dict()["state"]),
                          "diagnostic_shape": list(diagnostics["controls"].shape),
                          "diagnostics_finite": finite_tensor_tree(diagnostics)}
            if not all(validation[key] for key in ("model_and_optimizer_finite", "identical_loaded_actions",
                                                  "optimizer_state_nonempty", "diagnostics_finite")):
                raise RuntimeError("Exported benchmark artifact validation failed")
            item["checkpoint_validation"] = validation
            item["exports"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                               for path in (diagnostic_path, checkpoint_path)]
            item["export_and_reload_wall_s"] = time.monotonic() - export_start
            item["complete_cohort_wall_s"] = time.monotonic() - cohort_start
            item["scheduling"]["measured_trailing_wall_fraction"] = trailing_wall / item["complete_cohort_wall_s"]
            report["cohorts"].append(item)
            report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            write_json(args.output / "progress.json", report)
            del loaded, reloaded_optimizer, diagnostics, diagnostic_rows
        if report["cost"]["charged_control_equivalent_transitions"] != specification["expected_charged_transitions"]:
            raise RuntimeError("Measured initialization/rollout cost differs from the preregistration")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if env is not None:
            report["cost"] = env.cost_report()
        raise
    finally:
        report["worker_wall_s_before_close"] = time.monotonic() - start
        write_json(args.output / "report.json", report)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
