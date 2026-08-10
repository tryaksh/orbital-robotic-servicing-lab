"""Benchmark safe parallel-environment counts in isolated Isaac Sim processes."""

# ruff: noqa: E402, I001 -- child-only Isaac imports must follow AppLauncher.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_CANDIDATES = {
    "state": (1024, 768, 512, 256),
    "vision": (256, 128, 64),
}

# The state profile benchmarks the force-feedback insertion task because that is
# the lineage every current and planned run sits on, and its contact sensor
# forbids Fabric cloning, which is exactly the cost worth measuring.
PROFILE_TASKS = {
    "state": "Isaac-ZeroG-Blade-Insertion-ForceFeedback-v0",
    "vision": "Isaac-ZeroG-Blade-Insertion-Vision-v0",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("state", "vision", "all"), default="all")
    parser.add_argument("--candidates", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--memory_budget_mib", type=int, default=10_752)
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Benchmark every candidate instead of stopping at the first passing descending candidate.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap warm-up/measured steps at 20/50 for installation validation.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest.json"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_envs", type=int, default=None, help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.warmup < 0 or args.steps < 1:
    parser.error("--warmup must be non-negative and --steps must be positive")
if args.memory_budget_mib < 1:
    parser.error("--memory_budget_mib must be positive")
if args.quick:
    args.warmup = min(args.warmup, 20)
    args.steps = min(args.steps, 50)


def _run_parent() -> int:
    if args.profile == "all" and args.candidates:
        parser.error("--candidates is profile-specific; run teacher and vision separately to override it")
    profiles = ("state", "vision") if args.profile == "all" else (args.profile,)
    profile_results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="blade_swap_benchmark_") as temp_dir:
        for profile in profiles:
            candidates = sorted(set(args.candidates or DEFAULT_CANDIDATES[profile]), reverse=True)
            if any(count < 1 for count in candidates):
                parser.error("all --candidates values must be positive")
            results: list[dict] = []
            for count in candidates:
                result_file = Path(temp_dir) / f"{profile}_{count}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--profile",
                    profile,
                    "--num_envs",
                    str(count),
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                    "--memory_budget_mib",
                    str(args.memory_budget_mib),
                    "--result",
                    str(result_file),
                    "--headless",
                ]
                if profile == "vision":
                    command.append("--enable_cameras")
                completed = subprocess.run(command, check=False, env=os.environ.copy())
                if result_file.is_file():
                    result = json.loads(result_file.read_text(encoding="utf-8"))
                else:
                    result = {"profile": profile, "num_envs": count, "ok": False, "return_code": completed.returncode}
                result.setdefault("return_code", completed.returncode)
                results.append(result)
                # Candidates are descending. The first pass is therefore the
                # largest safe configuration; failures must continue downward.
                if result.get("ok", False) and not args.run_all:
                    break
            profile_results[profile] = results

    selections: dict[str, int | None] = {}
    for profile, results in profile_results.items():
        accepted = [
            item for item in results if item.get("ok") and item.get("gpu_memory_mib", 10**9) <= args.memory_budget_mib
        ]
        selections[profile] = max((item["num_envs"] for item in accepted), default=None)
    output = {
        "profile": args.profile,
        "memory_budget_mib": args.memory_budget_mib,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "search_mode": "full_matrix" if args.run_all else "descending_first_fit",
        "candidate_matrices": {
            profile: list(sorted(set(args.candidates or DEFAULT_CANDIDATES[profile]), reverse=True))
            for profile in profiles
        },
        "selected_num_envs": selections,
        "results": profile_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if all(value is not None for value in selections.values()) else 2


if not args.child:
    raise SystemExit(_run_parent())

if args.num_envs is None or args.result is None:
    parser.error("internal benchmark child requires --num_envs and --result")
if args.profile == "all":
    parser.error("internal benchmark child requires a concrete profile")
if args.profile == "vision":
    args.enable_cameras = True
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401


def _gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _run_child() -> int:
    task = PROFILE_TASKS[args.profile]
    env = None
    result = {"profile": args.profile, "num_envs": args.num_envs, "ok": False}
    try:
        cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
        env = gym.make(task, cfg=cfg)
        env.reset()
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        actions = torch.empty((args.num_envs, action_dim), device=env.unwrapped.device).uniform_(-1.0, 1.0)
        for _ in range(args.warmup):
            env.step(actions)
        torch.cuda.synchronize()
        memory = _gpu_memory_mib()
        started = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        memory_samples = [value for value in (memory, _gpu_memory_mib()) if value is not None]
        memory = max(memory_samples) if memory_samples else round(torch.cuda.max_memory_reserved() / 2**20)
        within_budget = memory <= args.memory_budget_mib
        result.update(
            ok=within_budget,
            status="pass" if within_budget else "over_budget",
            memory_budget_mib=args.memory_budget_mib,
            elapsed_seconds=elapsed,
            environment_steps_per_second=args.steps * args.num_envs / elapsed,
            simulation_steps_per_second=args.steps / elapsed,
            gpu_memory_mib=memory,
            torch_peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            torch_peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
        )
    except Exception as exc:  # child must always leave a machine-readable failure
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    try:
        exit_code = _run_child()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
