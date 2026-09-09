"""Bounded runtime checks for the project finite-job adapter and masked PPO."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def replay_physics(buffer, criteria, expected, prefix, expected_witness=None):
    from assembly_recovery.contact_witness import ContactRecoveryWitness, WitnessCriteria
    from assembly_recovery.evaluation import JobEvaluator, PhysicsSample

    jobs = [JobEvaluator(f"{prefix}-{i}", criteria) for i in range(len(expected))]
    witnesses = [ContactRecoveryWitness(WitnessCriteria(physics_dt=criteria.physics_dt,
                 stall_window_s=criteria.stall_window_s)) for _ in jobs]
    for step, rows in enumerate(buffer.tolist(), 1):
        for i, row in enumerate(rows):
            active = jobs[i].outcome is None
            jobs[i].observe(PhysicsSample(step, bool(row[0]), bool(row[1]), row[2], row[3],
                                          bool(row[4]), row[5], row[6], bool(row[7]), bool(row[8]), (row[9], row[10])))
            if expected_witness is not None:
                witnesses[i].observe(step, row[11], row[12], row[13], jobs[i].first_stall_step or 0, active)
    for job in jobs:
        job.finish("probe_end")
    actual = [job.result() for job in jobs]
    if actual != expected:
        differences = [{"i": i, "cpu": a, "tensor": b} for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b]
        raise ValueError(f"CPU/tensor physics replay mismatch: {differences[:2]}")
    if expected_witness is not None and [w.result(j) for w, j in zip(witnesses, actual, strict=True)] != expected_witness:
        raise ValueError("Controller-independent witness differs between CPU and tensor")
    return {"jobs": len(jobs), "exact_cpu_evaluator_parity": True,
            "contact_witness_parity": expected_witness is not None}


def reward_probe(env):
    import torch
    from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

    saved_previous = env.prev_actions.clone()
    saved_latch = env.success_pred_scale
    # A reference evaluation outside the rollout hot path; same pre-reward latch.
    env.success_pred_scale = float(env.success_latch)
    actual = env._get_rewards().clone()
    actual_terms = {k: v.clone() for k, v in env.reward_terms.items()}
    env.prev_actions.copy_(saved_previous)
    reference = ForgeEnv._get_rewards(env).clone()
    ref_terms = env.reward_terms
    error = max(float((actual_terms[k] - ref_terms[k]).abs().max()) for k in actual_terms)
    env.prev_actions.copy_(saved_previous)
    env.success_pred_scale = saved_latch
    return {"max_total_error": float((actual - reference).abs().max()), "max_term_error": error,
            "passed": error < 1e-5 and bool(torch.allclose(actual, reference, atol=1e-5, rtol=0))}


def policy_probe(env, model):

    from scripts.validate_peg import information_probe

    cached_probe = information_probe(env)
    baseline = env._get_observations()
    normalized = model.normalize(baseline)
    before = model.distribution(normalized["policy"]).mean
    modified = {k: v.clone() for k, v in baseline.items()}
    modified["critic"] += 100
    changed = model.normalize(modified)
    actor_delta = float((before - model.distribution(changed["policy"]).mean).abs().max())
    critic_delta = float((model.value(normalized["critic"]) - model.value(changed["critic"])).abs().max())
    return {**cached_probe, "instantiated_network": model.spec,
            "privileged_input_actor_delta": actor_delta, "privileged_input_value_delta": critic_delta,
            "passed": cached_probe["passed"] and actor_delta == 0 and critic_delta > 0}


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10070)
    parser.add_argument("--cohorts", type=int, default=1)
    parser.add_argument("--uniform-dev-mix", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--controller", choices=("hold", "insert", "policy"), default="policy")
    parser.add_argument("--optimize", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "research_result": False, "mode": "development_contract_probe",
              "controller": "project_untrained_ppo" if args.controller == "policy" else "scripted_" + args.controller,
              "source_provenance": "Parent manifest and prelaunch source.zip",
              "seed": args.seed, "num_envs": args.num_envs, "cohorts": []}
    app = env = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.faults import development_cases
        from assembly_recovery.study_ppo import StudyPolicy, ppo_update
        from assembly_recovery.training_env import PegTrainingEnv

        torch.set_num_threads(1)
        study = json.loads((ROOT / "configs/study.json").read_text())
        cases = development_cases(study, args.seed)
        if args.uniform_dev_mix:
            from assembly_recovery.training_cases import support_grid_cases

            cases = support_grid_cases(study, args.seed, "interior")
        cases = [dict(cases[i % len(cases)], case_id=f"probe-{args.seed}-{i}-{cases[i % len(cases)]['bin_id']}")
                 for i in range(args.num_envs)]
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        c = study["job_protocol"]
        criteria = JobCriteria(physics_dt=cfg.sim.dt, deadline_s=c["deadline_seconds"],
                               seated_dwell_s=c["seated_dwell_seconds"], force_budget_n=c["force_budget_n"])
        construction_start = time.monotonic()
        env = PegTrainingEnv(cfg, criteria=criteria, cases=cases, nominal=study["fault_support"]["nominal"], audit=args.audit)
        report["construction_wall_s"] = time.monotonic() - construction_start
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        report.update(criteria=asdict(criteria), geometry=env.geometry.report(), fault_cases=cases,
                      action_grad_penalty_scale=cfg.task.action_grad_penalty_scale,
                      held_part_gravity_enabled=not cfg.task.held_asset.spawn.rigid_props.disable_gravity,
                      contact_filter_paths=env.contact_filter_paths, finite_horizon=cfg.is_finite_horizon,
                      reward_convention="Endpoint-only v1: partial terminal interval omitted, absorbing zero, no terminal bootstrap.",
                      recurrent_state="No recurrence in the project feedforward policy; upstream recurrent reference unchanged.",
                      runtime={"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version})
        model = optimizer = None
        rollout = []
        metrics, durations, checks = [], [], []
        for cohort in range(args.cohorts):
            init_start = time.monotonic()
            observation, _ = env.reset()
            torch.cuda.synchronize()
            init_wall = time.monotonic() - init_start
            validity = env.initialization_report()
            if not all(v["valid"] for v in validity):
                report["invalid_initializations"] = validity
                raise RuntimeError("Invalid initialization; all requests preserved, no training update permitted")
            if model is None:
                model = StudyPolicy(observation["policy"].shape[-1], observation["critic"].shape[-1]).to(env.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                report["information_probe"] = policy_probe(env, model)
                if not report["information_probe"]["passed"]:
                    raise RuntimeError("Instantiated actor information check failed")
                report["policy_spec"] = model.spec
            initial_action = env.actions.clone().clamp(-1, 1)
            prefix = f"probe-{cohort}"
            env.begin_jobs(prefix)
            done_counts = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
            full_rewards, raw_rewards, valids, dones = [], [], [], []
            terminal_observation_errors = torch.zeros((), device=env.device)
            integration_error = torch.zeros((), dtype=torch.bool, device=env.device)
            cohort_start = interval_start = time.monotonic()
            for step in range(round(criteria.deadline_s / env.step_dt)):
                with torch.no_grad():
                    action, log_prob, value, norm = model.act(observation)
                    if args.controller != "policy":
                        action = initial_action.clone()
                        if args.controller == "insert" and step * env.step_dt >= 1:
                            action[:, :2] = 0
                            action[:, 2] = env.geometry.seated_fingertip_above_hole_top_m / cfg.ctrl.pos_action_bounds[2]
                    next_obs, reward, done, truncated, extra = env.step(action)
                    next_value = model.value(model.critic_norm(next_obs["critic"]))
                    terminal_observation_errors = torch.maximum(terminal_observation_errors,
                        torch.where(done[:, None], (next_obs["policy"] - extra["terminal_observation"]["policy"]).abs(), 0.).max())
                    integration_error |= truncated.any() | (extra["bootstrap"] != env.tensor_jobs.active).any()
                    integration_error |= ((~extra["valid"]) & done).any()
                    done_counts += done
                    row = {"actor": norm["policy"], "critic": norm["critic"], "raw_actor": observation["policy"],
                           "raw_critic": observation["critic"], "action": action, "log_prob": log_prob,
                           "value": value, "next_value": next_value, "reward": reward, "done": done, "valid": extra["valid"]}
                    if args.controller == "policy":
                        rollout.append(row)
                    if args.audit:
                        full_rewards.append(reward.clone())
                        raw_rewards.append(extra["raw_reward"].clone())
                        valids.append(extra["valid"].clone())
                        dones.append(done.clone())
                    observation = next_obs
                if (step + 1) % 128 == 0 or step + 1 == round(criteria.deadline_s / env.step_dt):
                    torch.cuda.synchronize()
                    end = time.monotonic()
                    chunk_steps = 128 if (step + 1) % 128 == 0 else (step + 1) % 128
                    durations.append({"cohort": cohort, "end_control": step + 1, "steps": chunk_steps,
                                      "wall_s": end - interval_start, "transitions_per_s": chunk_steps * args.num_envs / (end - interval_start)})
                    if args.optimize and rollout:
                        update_start = time.monotonic()
                        data = {k: torch.stack([r[k] for r in rollout]) for k in rollout[0]}
                        metrics.append(ppo_update(model, optimizer, data))
                        torch.cuda.synchronize()
                        metrics[-1]["wall_s"] = time.monotonic() - update_start
                    rollout = []
                    interval_start = time.monotonic()
                    print(json.dumps({"cohort": cohort, "control_step": step + 1, "transitions_per_s": durations[-1]["transitions_per_s"]}), flush=True)
            torch.cuda.synchronize()
            cohort_wall = time.monotonic() - cohort_start
            reward_check = reward_probe(env)
            jobs = env.end_jobs()
            general_witness = env.contact_witness.results(jobs)
            cohort_report = {"cohort": cohort, "jobs": jobs, "initialization": validity,
                             "initialization_wall_s": init_wall, "rollout_and_optimizer_wall_s": cohort_wall,
                             "raw_reward_reference_check": reward_check, "done_counts": done_counts.cpu().tolist(),
                             "terminal_observation_error": float(terminal_observation_errors), "general_recovery_witness": general_witness}
            checks.append(not bool(integration_error))
            cohort_report["terminal_bootstrap_and_truncation_masks_valid"] = not bool(integration_error)
            checks.extend([reward_check["passed"], bool((done_counts == 1).all()), float(terminal_observation_errors) == 0])
            export_start = time.monotonic()
            if args.audit:
                buffer = env.audit_buffer.cpu()
                cohort_report["replay"] = replay_physics(buffer, criteria, jobs, prefix, general_witness)
                rw, raw = torch.stack(full_rewards).cpu(), torch.stack(raw_rewards).cpu()
                terminal = env.tensor_jobs.terminal_step.cpu()
                endpoints = torch.arange(1, len(rw) + 1)[:, None] * cfg.decimation
                expected = torch.where(endpoints <= terminal[None, :], raw, 0.)
                if not torch.equal(rw, expected):
                    raise RuntimeError("Online reward differs from endpoint-only offline ledger")
                torch.save({"physics": buffer, "reward": rw, "raw_reward": raw, "valid": torch.stack(valids).cpu(),
                            "done": torch.stack(dones).cpu(), "terminal_step": terminal}, args.output / f"cohort_{cohort}.pt")
                cohort_report["endpoint_reward_parity"] = True
            cohort_report["export_and_cpu_replay_wall_s"] = time.monotonic() - export_start
            report["cohorts"].append(cohort_report)
        report["cost"] = env.cost_report()
        report["throughput_intervals"] = durations
        report["optimizer_updates"] = metrics
        report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["protocol_gates_passed"] = all(study["gates"].values())
        # Saving and reloading this development probe never supplies pilot weights.
        checkpoint = {"model": model.state_dict(), "spec": model.spec, "cost": report["cost"],
                      "seed": args.seed, "scope": "Discarded development integration checkpoint; not a trained baseline."}
        torch.save(checkpoint, args.output / "checkpoint.pt")
        loaded = torch.load(args.output / "checkpoint.pt", map_location=env.device, weights_only=True)
        rebuilt = StudyPolicy(loaded["spec"]["actor_size"], loaded["spec"]["critic_size"],
                              loaded["spec"]["action_size"], loaded["spec"]["widths"]).to(env.device)
        rebuilt.load_state_dict(loaded["model"])
        with torch.no_grad():
            a = model.act(observation, deterministic=True)[0]
            b = rebuilt.act(observation, deterministic=True)[0]
        report["checkpoint_validation"] = {"finite_parameters": all(bool(torch.isfinite(v).all()) for v in loaded["model"].values()),
                                           "identical_loaded_actions": bool(torch.equal(a, b))}
        checks.extend(report["checkpoint_validation"].values())
        report["status"] = "completed" if all(checks) else "check_failed"
        report["scope_and_limitations"] = [
            "Development integration/capacity probe; no uniform baseline, learned recovery or final-test result.",
            "Project feedforward PPO differs from the retained upstream recurrent reference.",
            "Exact replay compares evaluator labels on identical samples; it does not yet establish identical physical trajectories versus the reference driver.",
            "Controller-independent contact/withdrawal witness is new development instrumentation; historical scripted-phase labels are retained separately.",
            "Memory allocator totals exclude non-Torch engine allocations; parent records total device memory.",
        ]
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if env is not None:
            report["cost"] = env.cost_report() if hasattr(env, "tensor_jobs") else None
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
