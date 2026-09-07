"""Bounded upstream FORGE smoke check; no learned-policy evaluation."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task", choices=("peg", "gear"), default="peg")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=170)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 16 or not 1 <= args.steps <= 256:
        parser.error("This smoke check supports 1..16 environments and 1..256 steps")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "Upstream FORGE smoke check; no policy trained or evaluated", "status": "starting",
              "seed": args.seed, "num_envs": args.num_envs}
    with args.report.open("x", encoding="utf8") as handle:
        json.dump(report, handle, indent=2)
    app = None
    env = None
    try:
        app = AppLauncher(args).app
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        task = "Isaac-Forge-PegInsert-Direct-v0" if args.task == "peg" else "Isaac-Forge-GearMesh-Direct-v0"
        cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make(task, cfg=cfg)
        base = env.unwrapped
        obs, _ = env.reset()
        force_peak = 0.0
        reset_count = 0
        for _ in range(args.steps):
            with torch.inference_mode():
                action = torch.zeros((args.num_envs, base.cfg.action_space), device=base.device)
                obs, reward, terminated, truncated, _ = env.step(action)
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite actor observations or reward")
            force = base.force_sensor_smooth[:, :3]
            if not torch.isfinite(force).all():
                raise RuntimeError("Non-finite simulated force signal")
            force_peak = max(force_peak, float(torch.linalg.vector_norm(force, dim=-1).max()))
            reset_count += int((terminated | truncated).sum())
        report.update(status="passed", task=task, completed_control_steps=args.steps,
                      actor_observation_shape=list(obs["policy"].shape), action_dimensions=base.cfg.action_space,
                      simulated_force_peak_n=force_peak, automatic_episode_resets=reset_count,
                      gravity=list(cfg.sim.gravity), held_asset_gravity_disabled=cfg.task.held_asset.spawn.rigid_props.disable_gravity,
                      scope=["Zero actions, not a policy test.", "Upstream task starts with a grasped part; no pickup or release tested.",
                             "Pose observations are simulator-derived with synthetic noise, not camera-derived.",
                             "Held-part gravity is disabled upstream; resolve before primary recovery experiments.", "No recovery or hardware-transfer claim."])
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
