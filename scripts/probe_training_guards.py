"""Deliberate API-failure controls for the instantiated finite-job environment."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10070)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "research_result": False, "controller": "deliberate_api_faults",
              "jobs": [], "initialization": [], "checks": {}, "guard_cohorts": []}
    app = env = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.faults import development_cases
        from assembly_recovery.peg_env import WithinJobMutationError
        from assembly_recovery.training_env import PegTrainingEnv

        study = json.loads((ROOT / "configs/study.json").read_text())
        cases = development_cases(study, args.seed)[:4]
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=4)
        cfg.seed = args.seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        env = PegTrainingEnv(cfg, criteria=JobCriteria(), cases=cases, nominal=study["fault_support"]["nominal"])
        snapshots = []
        actions = [
            ("pose_write", lambda: env._held_asset.write_root_pose_to_sim(env._held_asset.data.root_state_w[:, :7].clone()),
             "within_job_state_write"),
            ("reset", lambda: env.reset(), "within_job_simulator_reset"),
            ("initialization_ik", lambda: env.set_pos_inverse_kinematics(env.fingertip_midpoint_pos.clone(),
                env.fingertip_midpoint_quat.clone(), torch.arange(4, device=env.device)), "within_job_state_write"),
            ("initialization_physics", lambda: env.step_sim_no_action(), "within_job_state_write"),
        ]
        for name, action, outcome in actions:
            env.reset()
            report["initialization"].extend(env.initialization_report())
            env.begin_jobs(name)
            before = env._held_asset.data.root_state_w.clone()
            timestamp = env._robot._data._sim_timestamp
            initializations = env.initializations
            caught = False
            try:
                action()
            except WithinJobMutationError:
                caught = True
            after = env._held_asset.data.root_state_w.clone()
            jobs = env.end_jobs()
            check = (caught and torch.equal(before, after) and env._robot._data._sim_timestamp == timestamp
                     and env.initializations == initializations and all(j["outcome"] == outcome for j in jobs))
            report["checks"][name] = check
            report["jobs"].extend(jobs)
            report["guard_cohorts"].append({"name": name, "blocked_without_physics_or_pose_change": check})
            snapshots.append(torch.stack((before, after)).cpu())
            caught = False
            try:
                env.begin_jobs("illegal-restart")
            except RuntimeError as exc:
                caught = "fresh full-cohort" in str(exc)
            report["checks"][name + "_requires_fresh_initialization"] = caught
        env.reset()
        report["initialization"].extend(env.initialization_report())
        env.begin_jobs("nonfinite-action")
        action = env.actions.clone()
        action[0] = float("nan")
        with torch.no_grad():
            _, reward1, done1, _, _ = env.step(action)
            _, reward2, done2, _, _ = env.step(action)
        report["checks"]["live_nonfinite_action_terminates_once"] = bool(done1[0]) and not bool(done2[0])
        report["checks"]["nonfinite_absorbing_output_cannot_pollute_rewards"] = bool(torch.isfinite(reward1).all() & torch.isfinite(reward2).all()) and float(reward1[0]) == float(reward2[0]) == 0.
        jobs = env.end_jobs()
        report["checks"]["nonfinite_action_outcome"] = jobs[0]["outcome"] == "nonfinite_state"
        report["jobs"].extend(jobs)
        report["cost"] = env.cost_report()
        report["checks"]["all_initializations_valid"] = all(v["valid"] for v in report["initialization"])
        torch.save({"before_after_root_states": snapshots, "first_rewards": reward1.cpu(), "second_rewards": reward2.cpu()},
                   args.output / "guard_states.pt")
        report["status"] = "completed" if all(report["checks"].values()) else "check_failed"
        report["scope_and_limitations"] = [
            "Twenty deliberate development requests: sixteen blocked API faults and four jobs in the nonfinite-action control.",
            "Intended failed-job outcomes verify accounting, not assembly success or learned recovery.",
            "Guards cover the project/upstream asset APIs; they do not sandbox arbitrary low-level PhysX/USD access.",
            "No forbidden write/reset was actually executed; the before/after simulator states and timestamps match."]
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
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
