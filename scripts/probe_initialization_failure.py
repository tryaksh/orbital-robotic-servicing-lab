"""Development diagnosis of initial actions from a failed pilot RNG state."""
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "research_result": False, "scope": "Initialization-only RNG diagnosis. A fresh simulator is not an exact restored PhysX trajectory."}
    app = env = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        from assembly_recovery.evaluation import JobCriteria
        from assembly_recovery.protocol import sha256
        from assembly_recovery.training_cases import uniform_training_cases
        from assembly_recovery.training_env import PegTrainingEnv

        if sha256(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError("Checkpoint hash changed")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["status"] != "partial" or saved["completed_cohorts"] != 2:
            raise ValueError("Expected the preserved two-cohort partial checkpoint")
        study = json.loads((ROOT / "configs/study.json").read_text())
        cases = uniform_training_cases(study, 170, 2, 1024)
        cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", device=args.device or "cuda:0", num_envs=1024)
        cfg.seed = 170
        cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
        env = PegTrainingEnv(cfg, criteria=JobCriteria(), cases=cases, nominal=study["fault_support"]["nominal"])
        torch.set_rng_state(saved["cpu_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        env.reset()
        validity = env.initialization_report()
        report["invalid"] = [{"index": i, "case": cases[i], "validity": v, "actions": env.actions[i].cpu().tolist(),
                              "fingertip_pos": env.fingertip_midpoint_pos[i].cpu().tolist(),
                              "target_estimate": (env.fixed_pos_obs_frame[i] + env.init_fixed_pos_obs_noise[i]).cpu().tolist(),
                              "fixed_position_noise": env.init_fixed_pos_obs_noise[i].cpu().tolist()}
                             for i, v in enumerate(validity) if not v["valid"]]
        report.update(status="completed", initialization=validity, cost=env.cost_report(), checkpoint_sha256=sha256(args.checkpoint))
        torch.save({"initial_actions": env.actions.cpu(), "held_pos": env.held_pos.cpu(), "held_quat": env.held_quat.cpu(),
                    "fingertip_pos": env.fingertip_midpoint_pos.cpu(), "fixed_pos_obs_frame": env.fixed_pos_obs_frame.cpu(),
                    "fixed_position_noise": env.init_fixed_pos_obs_noise.cpu()}, args.output / "initialization.pt")
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
