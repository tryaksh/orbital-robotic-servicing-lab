"""Verify paired development physics timing without claiming convergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from verify_training_paths import verified_run

from assembly_recovery.force_diagnostics import TemporalLoadMonitor
from assembly_recovery.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence")
    cm, cr, ca = verified_run(args.coarse)
    fm, fr, fa = verified_run(args.fine)
    keys = [k for k in ca["initial"] if k not in {"held_pos", "held_quat", "fixed_pos", "fixed_quat"}]
    checks = {
        "resolutions": cr["refinement"] == 1 and fr["refinement"] == 2,
        "same_cases": cr["fault_cases"] == fr["fault_cases"],
        "same_behavior_source": all(cm["source_hashes"][k] == fm["source_hashes"][k] for k in cm["source_hashes"] if k.startswith("src/")),
        "same_driver_source": cm["source_hashes"]["scripts/probe_physics_refinement.py"] == fm["source_hashes"]["scripts/probe_physics_refinement.py"],
        "same_upstream_source": cm["upstream_source_sha256"] == fm["upstream_source_sha256"],
        "native_timing_checks": all(cr["checks"].values()) and all(fr["checks"].values()),
        "identical_commands": torch.equal(ca["commands"], fa["commands"]),
        "identical_sensor_noise_draws": torch.equal(ca["noise_draws"], fa["noise_draws"]),
        "identical_nuisance_and_materials": all(torch.equal(ca["initial"][k], fa["initial"][k]) for k in keys),
        "all_requested_initializations_valid": all(v["valid"] for r in (cr, fr) for v in r["initialization"]),
        "native_cpu_replay": all(r["cpu_replay"]["exact_cpu_evaluator_parity"] and r["cpu_replay"]["contact_witness_parity"] for r in (cr, fr)),
        "raw_abort_unchanged": cr["criteria"]["force_budget_n"] == fr["criteria"]["force_budget_n"] == 20,
    }
    diagnostics = []
    for report, data in ((cr, ca), (fr, fa)):
        for i, job in enumerate(report["jobs"]):
            dt = report["criteria"]["physics_dt"]
            end = round(job["elapsed_s"] / dt)
            monitor = TemporalLoadMonitor()
            for force in data["physics"][:end, i, 2].tolist():
                result = monitor.observe(force, dt)
            diagnostics.append({"refinement": report["refinement"], "case_id": report["fault_cases"][i]["case_id"],
                                "outcome": job["outcome"], "elapsed_s": job["elapsed_s"], **result})
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed",
              "research_result": False, "checks": checks,
              "initial_maximum_absolute_differences": {k: float((v - fa["initial"][k]).abs().max()) for k, v in ca["initial"].items()},
              "completion_coarse": sum(j["success"] for j in cr["jobs"]),
              "completion_fine": sum(j["success"] for j in fr["jobs"]),
              "joint_completion": sum(c["success"] and f["success"] for c, f in zip(cr["jobs"], fr["jobs"], strict=True)),
              "paired_requests": len(cr["jobs"]), "force_diagnostics_until_terminal": diagnostics,
              "runs": [{"run_id": m["run_id"], "manifest_sha256": sha256(d / "manifest.json"), "cost": r["cost"]}
                       for m, r, d in ((cm, cr, args.coarse), (fm, fr, args.fine))],
              "scope_and_limitations": [
                  "Four scripted nominal development pairs. Final policy robustness tests remain unopened.",
                  "Identical external timing and nuisance draws do not imply identical physical initialization or trajectory.",
                  "Pinned implicit gripper targets/gains match; its native solver integration changes with timestep.",
                  "Native raw 20 N abort remains decisive; temporal filtering is diagnostic only.",
                  "This verifies the comparison machinery and exposes sensitivity, not convergence or hardware realism."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "completion_coarse", "completion_fine", "joint_completion")}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
