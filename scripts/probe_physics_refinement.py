"""Development-only registered physics-timing and contact controls."""
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
    parser.add_argument("--refinement", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, default=10070)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "research_result": False, "controller": "scripted_physics_control_v1",
              "seed": args.seed, "refinement": args.refinement}
    app = env = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.faults import development_cases
        from assembly_recovery.refinement_env import PegRefinementEnv
        from scripts.probe_training import replay_physics

        torch.set_num_threads(1)
        study = json.loads((ROOT / "configs/study.json").read_text())
        cases = development_cases(study, args.seed)[:4]  # nominal development only
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=4)
        cfg.seed = args.seed
        cfg.sim.dt = 1 / (120 * args.refinement)
        cfg.decimation = 8 * args.refinement
        cfg.sim.render_interval = cfg.decimation
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        c = study["job_protocol"]
        criteria = JobCriteria(physics_dt=cfg.sim.dt, deadline_s=c["deadline_seconds"],
                               seated_dwell_s=c["seated_dwell_seconds"], force_budget_n=c["force_budget_n"])
        env = PegRefinementEnv(cfg, criteria=criteria, cases=cases, nominal=study["fault_support"]["nominal"],
                               audit=True, refinement=args.refinement, sensor_seed=1000000 + args.seed)
        start = time.monotonic()
        env.reset()
        report["initialization_wall_s"] = time.monotonic() - start
        validity = env.initialization_report()
        report.update(initialization=validity, criteria=asdict(criteria), geometry=env.geometry.report(), fault_cases=cases)
        if not all(v["valid"] for v in validity):
            raise RuntimeError("Refinement initialization invalid; all requested cases preserved")
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        initial = {name: getattr(env, name).clone().cpu() for name in (
            "held_pos", "held_quat", "fixed_pos", "fixed_quat", "init_fixed_pos_obs_noise", "ema_factor",
            "task_prop_gains", "pos_threshold", "rot_threshold", "dead_zone_thresholds", "contact_penalty_thresholds")}
        for name, asset in (("robot", env._robot), ("held", env._held_asset), ("fixed", env._fixed_asset)):
            initial[name + "_mass"] = asset.root_physx_view.get_masses().cpu()
            initial[name + "_material"] = asset.root_physx_view.get_material_properties().cpu()
        env.begin_jobs("refinement")
        controls, poses = [], []
        start = time.monotonic()
        for step in range(450):
            t = step * env.step_dt
            if t < 1:
                target_z = study["fault_support"]["nominal"]["hand_height_above_fixture_m"] - env.geometry.hole_height_m
            elif t < 4 or t >= 7:
                target_z = env.geometry.seated_fingertip_above_hole_top_m
            else:
                target_z = 0.045
            action = torch.zeros((4, 7), device=env.device)
            action[:, 2] = target_z / cfg.ctrl.pos_action_bounds[2]
            action[:, 5], action[:, 6] = 1 / 3, -1
            with torch.no_grad():
                env.step(action, commanded_down=torch.full((4,), 1 <= t < 4 or t >= 7, dtype=torch.bool, device=env.device))
                controls.append(action.clone())
                poses.append(env.held_pos.clone())
            if (step + 1) % 128 == 0:
                print(json.dumps({"step": step + 1, "refinement": args.refinement}), flush=True)
        torch.cuda.synchronize()
        report["rollout_wall_s"] = time.monotonic() - start
        jobs = env.end_jobs()
        witness = env.contact_witness.results(jobs)
        physics = env.audit_buffer.cpu()
        report.update(jobs=jobs, general_recovery_witness=witness,
                      cpu_replay=replay_physics(physics, criteria, jobs, "refinement", witness),
                      cost=env.cost_report(), sensor_updates=env.sensor_updates, servo_updates=env.servo_updates,
                      native_physics_steps=env.tensor_jobs.step)
        torques = torch.stack(env.torque_records).cpu()
        report["contact_force_conversion_dt"] = env.fixture_contact._sim_physics_dt
        report["checks"] = {"native_step_count": env.tensor_jobs.step == 3600 * args.refinement,
                            "servo_120hz": env.servo_updates == 3600, "sensor_noise_120hz": env.sensor_updates == 3600,
                            "fixture_contact_positive_control": bool((physics[:, :, 11] > 0.1).any()),
                            "finger_contact_positive_control": bool((physics[:, :, 9:11] > criteria.finger_contact_threshold_n).all(-1).any()),
                            "native_force_finite": bool(torch.isfinite(physics[:, :, 2]).all()),
                            "contact_force_native_dt": abs(env.fixture_contact._sim_physics_dt - criteria.physics_dt) < 1e-12,
                            "commanded_efforts_held_between_fine_substeps": args.refinement == 1 or torch.equal(torques[::2], torques[1::2])}
        torch.save({"physics": physics, "noise_draws": torch.stack(env.noise_records).cpu(),
                    "commands": torch.stack(controls).cpu(), "torques": torques, "part_pos": torch.stack(poses).cpu(), "initial": initial},
                   args.output / "trajectory.pt")
        report["status"] = "completed" if all(report["checks"].values()) else "check_failed"
        report["scope_and_limitations"] = [
            "Four nominal development jobs per resolution, not final robustness tests or a hardware result.",
            "Both resolutions use this evaluation adapter with separate sensor RNG. Training and upstream reference files are unchanged.",
            "All native raw wrist force samples retain the 20 N abort; no spike is reclassified as artificial.",
            "External arm torque commands update at 120 Hz. The pinned implicit gripper drive retains identical targets/gains and is integrated by the native physics solver at both resolutions.",
            "Identical commands and noise draws do not imply identical realized trajectories or initial physical poses.",
            "This controls timing and tests discretization sensitivity; it does not certify convergence or physical realism."]
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
