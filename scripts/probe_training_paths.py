"""Compare physical reference and tensor paths under an unchanged actor-only script."""
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


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference", "tensor"), required=True)
    parser.add_argument("--controller", choices=("retry", "continue"), default="retry")
    parser.add_argument("--grid", choices=("representative", "low", "interior", "high"), default="representative")
    parser.add_argument("--seed", type=int, default=10070)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "controller": "scripted_" + args.controller, "backend": args.backend,
              "seed": args.seed, "grid": args.grid, "research_result": False}
    app = env = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.contact_witness import ContactRecoveryWitness, WitnessCriteria
        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.fault_env import PegFaultEnv
        from assembly_recovery.faults import development_cases
        from assembly_recovery.retry_controller import ActorRetryController, RetrySettings
        from assembly_recovery.training_cases import support_grid_cases
        from assembly_recovery.training_env import PegTrainingEnv
        from scripts.compare_recovery import recovery_witness
        from scripts.probe_training import replay_physics

        torch.set_num_threads(1)
        study = json.loads((ROOT / "configs/study.json").read_text())
        cases = (development_cases(study, args.seed) if args.grid == "representative"
                 else support_grid_cases(study, args.seed, args.grid))
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=len(cases))
        cfg.seed = args.seed
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        c = study["job_protocol"]
        criteria = JobCriteria(physics_dt=cfg.sim.dt, deadline_s=c["deadline_seconds"],
                               seated_dwell_s=c["seated_dwell_seconds"], force_budget_n=c["force_budget_n"])
        class ReferencePathEnv(PegFaultEnv):
            def begin_jobs(self, prefix):
                super().begin_jobs(prefix)
                self.general_witness = [ContactRecoveryWitness(WitnessCriteria(physics_dt=criteria.physics_dt,
                                        stall_window_s=criteria.stall_window_s)) for _ in self.jobs]

            def _sample_jobs(self):
                active = [job.outcome is None for job in self.jobs]
                super()._sample_jobs()
                positions = torch.stack((self.held_pos[:, 2],
                            self.held_pos[:, 2] - self.fixed_pos[:, 2] - self.geometry.hole_height_m), dim=-1).cpu().tolist()
                for i, job in enumerate(self.jobs):
                    if active[i]:
                        self.general_witness[i].observe(job.last_step, self.contact_samples[i][-1], *positions[i],
                                                       job.first_stall_step or 0, True)

        cls = PegTrainingEnv if args.backend == "tensor" else ReferencePathEnv
        extra = {"audit": True} if args.backend == "tensor" else {}
        env = cls(cfg, criteria=criteria, cases=cases, nominal=study["fault_support"]["nominal"], **extra)
        init_start = time.monotonic()
        observation, _ = env.reset()
        init_wall = time.monotonic() - init_start
        validity = env.initialization_report()
        report.update(initialization=validity, fault_cases=cases, criteria=asdict(criteria), geometry=env.geometry.report(),
                      initial_fixed_pos=env.fixed_pos.cpu().tolist(), initialization_wall_s=init_wall)
        if not all(v["valid"] for v in validity):
            raise RuntimeError("Support initialization invalid; requests remain failed")
        initial = {name: getattr(env, name).cpu() for name in (
            "held_pos", "held_quat", "fixed_pos", "fixed_quat", "fingertip_midpoint_pos", "fingertip_midpoint_quat",
            "init_fixed_pos_obs_noise", "ema_factor", "task_prop_gains", "pos_threshold", "rot_threshold",
            "dead_zone_thresholds", "contact_penalty_thresholds", "actions", "force_sensor_world_smooth")}
        initial["actor_observation"] = observation["policy"].cpu()
        initial["robot_joint_pos"] = env._robot.data.joint_pos.cpu()
        initial["robot_joint_vel"] = env._robot.data.joint_vel.cpu()
        settings = RetrySettings(**json.loads((ROOT / "configs/retry_unload_realign_v1.json").read_text()))
        actor_obs = observation["policy"].cpu().tolist()
        controllers = [ActorRetryController(row, step_dt=env.step_dt, position_bounds=cfg.ctrl.pos_action_bounds,
                       seated_height_m=env.geometry.seated_fingertip_above_hole_top_m,
                       retry=args.controller == "retry", settings=settings) for row in actor_obs]
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        env.begin_jobs("paired")
        controls = []
        arrays = {"part_pos": [], "part_quat": [], "actor_obs": [], "commanded_action": [], "reward": []}
        start = time.monotonic()
        for step in range(450):
            decisions = [controller.act(row, step * env.step_dt) for controller, row in zip(controllers, actor_obs, strict=True)]
            action = torch.tensor([d[0] for d in decisions], device=env.device)
            phases = [d[1]["phase"] for d in decisions]
            down = torch.tensor([phase in {"insert", "search", "seat"} for phase in phases], device=env.device)
            active = env.tensor_jobs.active.cpu().tolist() if args.backend == "tensor" else [j.outcome is None for j in env.jobs]
            with torch.no_grad():
                if args.backend == "tensor":
                    obs, reward, _, _, _ = env.step(action, commanded_down=down)
                else:
                    env.commanded_down = down
                    obs, reward, _, _, _ = env.step(action)
                # Continue actor-only script even in absorbing slots as the reference did.
                actor_obs = env._get_observations()["policy"].cpu().tolist()
                for name, value in (("part_pos", env.held_pos), ("part_quat", env.held_quat),
                                    ("actor_obs", obs["policy"]), ("commanded_action", action), ("reward", reward)):
                    arrays[name].append(value.clone())
            controls.append({"step": step + 1, "phase": phases, "active_before_step": active,
                             "part_pos": env.held_pos.cpu().tolist()})
            if (step + 1) % 128 == 0:
                print(json.dumps({"step": step + 1, "backend": args.backend}), flush=True)
        torch.cuda.synchronize()
        report["rollout_wall_s"] = time.monotonic() - start
        jobs = env.end_jobs()
        report.update(jobs=jobs, controllers=[c.report() for c in controllers])
        physics = {}
        if args.backend == "tensor":
            buffer = env.audit_buffer.cpu()
            report["general_recovery_witness"] = env.contact_witness.results(jobs)
            report["cpu_replay"] = replay_physics(buffer, criteria, jobs, "paired", report["general_recovery_witness"])
            rows = buffer.tolist()
            for i, job in enumerate(jobs):
                terminal = round(job["elapsed_s"] / criteria.physics_dt)
                physics[job["job_id"]] = [{"peg_fixture_contact_force_n": r[i][11], "step": k + 1}
                                          for k, r in enumerate(rows[:terminal])]
            report["cost"] = env.cost_report()
        else:
            report["general_recovery_witness"] = [w.result(j) for w, j in zip(env.general_witness, jobs, strict=True)]
            for i, job in enumerate(jobs):
                physics[job["job_id"]] = [{"peg_fixture_contact_force_n": force, "step": sample["step"]}
                    for sample, force in zip(env.samples[i], env.contact_samples[i], strict=True)]
            buffer = None
            report["cost"] = {"rollout_control_transitions": len(cases) * 450,
                              "initialization_physics_env_steps": None,
                              "limitation": "Reference initialization physics not instrumented; full wall time is recorded."}
        report["witnesses"] = [recovery_witness(report, controls, physics, i) for i in range(len(jobs))]
        export_start = time.monotonic()
        artifact = {k: torch.stack(v).cpu() for k, v in arrays.items()}
        artifact["initial"] = initial
        artifact["physics"] = buffer
        torch.save(artifact, args.output / "trajectory.pt")
        report["export_wall_s"] = time.monotonic() - export_start
        report.update(status="completed", scope_and_limitations=[
            "Unchanged scripted development controller; no learned-policy evaluation or final-test case.",
            "Recovery witness retains the existing scripted trigger/withdrawal definition.",
            "Continuous support checks sample declared values; they cannot prove every point in a range.",
            "Full traces and CPU script decisions are enabled; this is not the optimized policy training throughput."])
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
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
