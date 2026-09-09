"""Verify saved physical path pairing; failures remain explicit JSON evidence."""
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


def verified_run(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("A failed process cannot supply a completed comparison")
    for artifact in manifest["artifacts"]:
        if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Artifact hash mismatch")
    with zipfile.ZipFile(path / "source.zip") as archive:
        for name, digest in manifest["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Archived source mismatch")
    report = json.loads((path / "probe/report.json").read_text())
    data = torch.load(path / "probe/trajectory.pt", map_location="cpu", weights_only=True)
    return manifest, report, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Comparison evidence already exists; preserve it")
    rm, rr, ra = verified_run(args.reference)
    tm, tr, ta = verified_run(args.tensor)
    checks = {
        "correct_backends": rr["backend"] == "reference" and tr["backend"] == "tensor",
        "same_task_and_cases": all(rr[k] == tr[k] for k in ("controller", "seed", "grid", "fault_cases", "criteria", "geometry")),
        "same_behavior_source": all(rm["source_hashes"][k] == tm["source_hashes"][k]
            for k in rm["source_hashes"] if k.startswith(("src/", "configs/")) or k in {
                "scripts/probe_training_paths.py", "scripts/compare_recovery.py", "scripts/probe_training.py"})
            and rm["upstream_source_sha256"] == tm["upstream_source_sha256"],
        "identical_initial_state": all(torch.equal(v, ta["initial"][k]) for k, v in ra["initial"].items()),
        "identical_job_outcomes_and_timing": rr["jobs"] == tr["jobs"],
        "identical_general_witness": rr["general_recovery_witness"] == tr["general_recovery_witness"],
        "identical_witnessed_recovery": rr["witnesses"] == tr["witnesses"],
        "identical_actor_script_state": rr["controllers"] == tr["controllers"],
    }
    differences = {}
    for name in ("part_pos", "part_quat", "commanded_action"):
        differences[name] = float((ra[name] - ta[name]).abs().max())
        checks[f"identical_{name}"] = torch.equal(ra[name], ta[name])
    terminal = torch.tensor([round(j["elapsed_s"] / rr["criteria"]["physics_dt"]) for j in rr["jobs"]])
    endpoints = torch.arange(1, 451)[:, None] * 8
    reference_reward = torch.where(endpoints <= terminal[None, :], ra["reward"], 0.)
    differences["job_reward"] = float((reference_reward - ta["reward"]).abs().max())
    checks["reward_within_1e_minus_5"] = torch.allclose(reference_reward, ta["reward"], atol=1e-5, rtol=0)
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "pairing_failed",
              "research_result": False, "checks": checks, "maximum_absolute_differences": differences,
              "runs": [{"run_id": m["run_id"], "manifest_sha256": sha256(d / "manifest.json")}
                       for m, d in ((rm, args.reference), (tm, args.tensor))],
              "jobs_per_path": len(rr["jobs"]), "completed_jobs": sum(j["success"] for j in tr["jobs"]),
              "witnessed_recoveries": sum(w["witnessed_complete_recovery"] for w in tr["witnesses"]),
              "reference_rollout_wall_s": rr["rollout_wall_s"], "tensor_rollout_wall_s": tr["rollout_wall_s"],
              "scope_and_limitations": [
                  "One unchanged scripted development cohort; no learned result or continuous-range proof.",
                  "Behavior source hashes must match; full archived snapshots may differ in offline verification scripts and documents.",
                  "Both paths run actor-only CPU scripts and export traces; this does not measure optimized PPO speedup.",
                  "Reward convention remains endpoint-only with omitted terminal partial intervals.",
                  "The reference and tensor drivers have different internal horizon metadata; both run the same external 30-second finite jobs without auto-reset."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
