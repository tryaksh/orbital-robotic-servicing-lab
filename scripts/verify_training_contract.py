"""Summarize verified bounded integration and capacity probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assembly_recovery.protocol import sha256  # noqa: E402


def verify_assets(path):
    m = json.loads((path / "manifest.json").read_text())
    for a in m["artifacts"]:
        if sha256(ROOT / a["path"]) != a["sha256"]:
            raise ValueError("Artifact mismatch: " + a["path"])
    with zipfile.ZipFile(path / "source.zip") as archive:
        for name, digest in m["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Source mismatch: " + name)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence")
    names = ["tensor_contract_32_r02", "tensor_contract_64_r01", "tensor_contract_128_r01",
             "tensor_capacity_256_r01", "tensor_capacity_512_r01", "tensor_capacity_1024_r01"]
    rows = []
    for name in names:
        p = ROOT / "artifacts/assembly" / name
        m = verify_assets(p)
        r = json.loads((p / "probe/report.json").read_text())
        g = json.loads((p / "gpu_resources.json").read_text())
        checks = {"process_complete": m["status"] == "completed", "report_complete": r["status"] == "completed",
                  "actor_transform_audit": r["information_probe"]["passed"],
                  "privileged_critic_does_not_change_actor": r["information_probe"]["privileged_input_actor_delta"] == 0,
                  "checkpoint_reloaded": all(r["checkpoint_validation"].values()),
                  "one_done_per_job": all(x == 1 for c in r["cohorts"] for x in c["done_counts"]),
                  "frozen_terminal_observations": all(c["terminal_observation_error"] == 0 for c in r["cohorts"]),
                  "upstream_reward_terms": all(c["raw_reward_reference_check"]["passed"] for c in r["cohorts"]),
                  "valid_initializations": all(v["valid"] for c in r["cohorts"] for v in c["initialization"])}
        if name.startswith("tensor_contract"):
            checks["cpu_native_trace_parity"] = all(c["replay"]["exact_cpu_evaluator_parity"] and c["endpoint_reward_parity"] for c in r["cohorts"])
        else:
            checks["terminal_bootstrap_masks"] = all(c["terminal_bootstrap_and_truncation_masks_valid"] for c in r["cohorts"])
        checkpoints = list((p / "probe").glob("*.pt"))
        for ckpt in checkpoints:
            data = torch.load(ckpt, map_location="cpu", weights_only=True)
            if "model" in data:
                checks["checkpoint_currently_finite"] = all(bool(torch.isfinite(v).all()) for v in data["model"].values())
        rates = [i["transitions_per_s"] for i in r["throughput_intervals"]]
        rows.append({"run_id": name, "manifest_sha256": sha256(p / "manifest.json"), "checks": checks,
                     "num_envs": r["num_envs"], "cost": r["cost"], "rollout_transitions_per_s_min_max": [min(rates), max(rates)],
                     "process_wall_s": m["elapsed_wall_s"], "peak_gpu_mib": g["peak_used_mib"],
                     "minimum_free_ram_mib": g.get("minimum_free_ram_mib"),
                     "initialization_wall_s": sum(c["initialization_wall_s"] for c in r["cohorts"]),
                     "export_wall_s": sum(c["export_and_cpu_replay_wall_s"] for c in r["cohorts"]),
                     "optimizer_wall_s": sum(x["wall_s"] for x in r["optimizer_updates"])})
    p = ROOT / "artifacts/assembly/training_guards_r01"
    m = verify_assets(p)
    guard = json.loads((p / "probe/report.json").read_text())
    guard_ok = m["status"] == "completed" and all(guard["checks"].values())
    failed = ROOT / "artifacts/assembly/tensor_contract_32_r01"
    failure = verify_assets(failed)
    result = {"schema": 1, "status": "verified" if guard_ok and all(all(r["checks"].values()) for r in rows) else "check_failed",
              "research_result": False, "capacity_runs": rows,
              "runtime_guards": {"run_id": m["run_id"], "manifest_sha256": sha256(p / "manifest.json"), "checks": guard["checks"], "cost": guard["cost"]},
              "preserved_failed_probe": {"run_id": failure["run_id"], "status": failure["status"],
                  "manifest_sha256": sha256(failed / "manifest.json"), "reason": "Overbroad interval-event guard rejected the pinned dead-zone event before rollout. Failed process retained; zero measured rollout."},
              "scope_and_limitations": [
                  "Six single-cohort capacities, four PPO update boundaries each; longer thermal/cross-cohort stability awaits the bounded pilot.",
                  "32/64/128 audit probes and later 256/512/1024 probes have distinct exact source snapshots. They are not a controlled algorithm speed comparison.",
                  "All development weights are discarded for pilot initialization. No learned recovery performance claim.",
                  "Partial terminal reward interval is deliberately omitted; no terminal bootstrap, absorbing losses or automatic reset.",
                  "All probe/initialization/absorbing transitions are development costs, separate from the predeclared pilot budget.",
                  "API guards do not sandbox arbitrary low-level simulator access. Actor audit covers tested observation and transform routes."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
