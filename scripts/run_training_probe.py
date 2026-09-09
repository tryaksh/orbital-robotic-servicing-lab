"""Launch a single provenance-recorded finite-job development probe."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.protocol import LAB_COMMIT, assess_completion, sha256, validate_run_id, write_json  # noqa: E402
from scripts.run_experiment import git, snapshot_source, stop_owned_process  # noqa: E402


def free_ram_mib():
    if os.name != "nt":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                   "virtual_total", "virtual_available", "extended")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return status.available / 1024 ** 2
    return None


def run_resource_bounded(command, log_path, deadline, environment, samples):
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log_path.open("w", encoding="utf8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        printed = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    stop_owned_process(process)
                    return process.returncode, True, None
                if samples:
                    current = samples[-1]
                    if current["used_mib"] > 10240 or (current["free_ram_mib"] is not None and current["free_ram_mib"] < 4096):
                        stop_owned_process(process)
                        return process.returncode, False, current
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if time.monotonic() - printed >= 30:
                        print(f"Running PID {process.pid}; deadline in {int(deadline - time.monotonic())}s", flush=True)
                        printed = time.monotonic()
            return process.returncode, False, None
        except BaseException:
            if process.poll() is None:
                stop_owned_process(process)
            raise


def monitor_gpu(stop, samples):
    while not stop.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode == 0:
                memory, utilization = map(int, result.stdout.strip().splitlines()[0].split(","))
                samples.append({"time": time.time(), "used_mib": memory, "utilization_percent": utilization, "free_ram_mib": free_ram_mib()})
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        stop.wait(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=10070)
    parser.add_argument("--num-envs", type=int, choices=(4, 32, 64, 128, 256, 512, 1024), default=32)
    parser.add_argument("--cohorts", type=int, default=1)
    parser.add_argument("--uniform-dev-mix", action="store_true")
    parser.add_argument("--max-minutes", type=int, default=20)
    parser.add_argument("--controller", choices=("hold", "insert", "policy", "retry", "continue"), default="policy")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--path-backend", choices=("reference", "tensor"))
    parser.add_argument("--refinement", type=int, choices=(1, 2))
    parser.add_argument("--guards", action="store_true")
    parser.add_argument("--grid", choices=("representative", "low", "interior", "high"), default="representative")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--sim-python", type=Path, default=Path("C:/isaac-sim/python.bat"))
    args = parser.parse_args()
    study = json.loads((ROOT / "configs/study.json").read_text())
    if args.seed not in study["splits"]["development_seeds"] or not 1 <= args.cohorts <= 6 or not 1 <= args.max_minutes <= 90:
        parser.error("Only bounded development cohorts are permitted")
    if args.optimize and args.controller != "policy":
        parser.error("Cannot train on scripted actions using policy log probabilities")
    if args.path_backend:
        if args.controller not in {"retry", "continue"} or args.cohorts != 1 or args.optimize:
            parser.error("Path comparisons require one unchanged scripted cohort without PPO optimization")
        args.num_envs = 28 if args.grid == "representative" else 32
    elif args.controller in {"retry", "continue"}:
        parser.error("Scripted comparisons require --path-backend")
    if args.refinement:
        if args.path_backend or args.optimize or args.cohorts != 1:
            parser.error("Refinement controls require one separate non-learning cohort")
        args.num_envs = 4
    if args.guards:
        if args.path_backend or args.refinement or args.optimize or args.cohorts != 1:
            parser.error("Guards require a separate non-learning API control")
        args.num_envs = 4
    upstream = ROOT / ".deps/IsaacLab"
    if git(upstream, "rev-parse", "HEAD") != LAB_COMMIT or git(upstream, "status", "--porcelain", "--untracked-files=no"):
        parser.error("Pinned upstream source must be unchanged")
    run_id = validate_run_id(args.run_id)
    output = ROOT / "artifacts/assembly" / run_id
    output.mkdir(parents=True, exist_ok=False)
    command = [str(args.sim_python), str(ROOT / "scripts/probe_training.py"), "--headless", "--output", str(output / "probe"),
               "--seed", str(args.seed), "--num-envs", str(args.num_envs), "--cohorts", str(args.cohorts),
               "--controller", args.controller]
    command += ["--audit"] if args.audit else []
    command += ["--optimize"] if args.optimize else []
    command += ["--uniform-dev-mix"] if args.uniform_dev_mix else []
    if args.path_backend:
        command = [str(args.sim_python), str(ROOT / "scripts/probe_training_paths.py"), "--headless",
                   "--output", str(output / "probe"), "--seed", str(args.seed), "--backend", args.path_backend,
                   "--controller", args.controller, "--grid", args.grid]
    if args.refinement:
        command = [str(args.sim_python), str(ROOT / "scripts/probe_physics_refinement.py"), "--headless",
                   "--output", str(output / "probe"), "--seed", str(args.seed), "--refinement", str(args.refinement)]
    if args.guards:
        command = [str(args.sim_python), str(ROOT / "scripts/probe_training_guards.py"), "--headless",
                   "--output", str(output / "probe"), "--seed", str(args.seed)]
    lock = ROOT / "environment-lock.local.json"
    manifest = {"schema": 1, "run_id": run_id, "status": "starting", "research_result": False,
                "started_at_utc": datetime.now(UTC).isoformat(), "command": command, "seed": args.seed,
                "predeclared_rollout_control_transitions": 8 if args.guards else args.cohorts * 450 * args.num_envs,
                "initialization_cost": "Measured physics steps are charged in addition to the fixed rollout budget.",
                "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"),
                "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
                "source_hashes": snapshot_source(output / "source.zip"), "upstream_commit": LAB_COMMIT,
                "upstream_source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                    for folder in ("factory", "forge")
                    for p in (upstream / "source/isaaclab_tasks/isaaclab_tasks/direct" / folder).rglob("*.py")},
                "environment_lock": json.loads(lock.read_text()) if lock.exists() else None,
                "environment_lock_sha256": sha256(lock) if lock.exists() else None, "input_checkpoint": None,
                "environment_overrides": {"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
                "unmet_study_gates": [k for k, v in study["gates"].items() if not v],
                "scope_and_limitations": ["Bounded development integration/capacity check. All transitions are probe costs.",
                                         "Any saved development weights are excluded from pilot initialization.",
                                         "This launcher cannot start a learning campaign or open final-test cases."]}
    write_json(output / "manifest.json", manifest)
    samples, stop = [], threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
    monitor.start()
    start = time.monotonic()
    try:
        code, timeout, resource_stop = run_resource_bounded(command, output / "process.log", start + args.max_minutes * 60,
                                  {**os.environ, **manifest["environment_overrides"]}, samples)
        manifest["resource_guard_stop"] = resource_stop
        manifest["resource_limits"] = {"max_gpu_used_mib": 10240, "minimum_free_ram_mib": 4096}
        report_path = output / "probe/report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        checks = {"probe_report_completed": report.get("status") == "completed",
                  "checkpoint_loaded": all(report.get("checkpoint_validation", {"missing": False}).values()),
                  "all_requested_cohorts": len(report.get("cohorts", [])) == args.cohorts,
                  "rollout_cost_matches": report.get("cost", {}).get("rollout_control_transitions") == manifest["predeclared_rollout_control_transitions"]}
        if args.path_backend or args.refinement:
            checks = {"probe_report_completed": report.get("status") == "completed",
                      "all_requested_jobs": len(report.get("jobs", [])) == args.num_envs,
                      "trajectory_saved": (output / "probe/trajectory.pt").is_file(),
                      "all_initializations_valid": bool(report.get("initialization")) and all(v["valid"] for v in report["initialization"]),
                      "rollout_cost_matches": report.get("cost", {}).get("rollout_control_transitions") == manifest["predeclared_rollout_control_transitions"]}
        if args.guards:
            checks = {"probe_report_completed": report.get("status") == "completed",
                      "all_guard_requests": len(report.get("jobs", [])) == 20,
                      "all_guards_passed": bool(report.get("checks")) and all(report["checks"].values()),
                      "guard_states_saved": (output / "probe/guard_states.pt").is_file(),
                      "rollout_cost_matches": report.get("cost", {}).get("rollout_control_transitions") == 8}
        manifest.update(returncode=code, timed_out=timeout, checks=checks, status=assess_completion(code, timeout, checks))
    except BaseException as exc:
        manifest.update(status="launcher_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        monitor.join(timeout=8)
        write_json(output / "gpu_resources.json", {"samples": samples, "peak_used_mib": max((x["used_mib"] for x in samples), default=None),
                                                   "minimum_free_ram_mib": min((x["free_ram_mib"] for x in samples if x["free_ram_mib"] is not None), default=None),
                                                  "scope": "Whole-device GPU and free system RAM sampling every two seconds; short peaks may be missed."})
        manifest["elapsed_wall_s"] = time.monotonic() - start
        manifest["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                                 for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "process.log"}]
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "wall_s": manifest["elapsed_wall_s"]}))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
