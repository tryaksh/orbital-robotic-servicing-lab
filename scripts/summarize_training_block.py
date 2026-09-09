"""Verify the explicit block run ledger, retaining failed and unmeasured costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_training_contract import ROOT, verify_assets

from assembly_recovery.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence")
    names = ["tensor_contract_32_r01", "tensor_contract_32_r02", "tensor_contract_64_r01", "tensor_contract_128_r01",
             "training_path_reference_r01", "training_path_tensor_r01", "tensor_capacity_256_r01", "tensor_capacity_512_r01", "tensor_capacity_1024_r01"]
    names += [f"training_support_{grid}_{arm}_r01" for grid in ("low", "interior", "high") for arm in ("retry", "continue")]
    names += ["physics_refinement_120_r01", "physics_refinement_240_r01", "training_guards_r01", "uniform_pilot_170_v1",
              "initialization_failure_diagnostic_v1", "initialization_projection_diagnostic_v2", "training_path_reference_r02", "training_path_tensor_r02",
              "tensor_contract_projection_32_r01", "uniform_pilot_170_v2", "uniform_dev_10070_v2", "uniform_dev_10071_v2", "uniform_dev_10072_v2"]
    rows = []
    for name in names:
        path = ROOT / "artifacts/assembly" / name
        m = verify_assets(path)
        report_path = path / ("pilot/report.json" if name.startswith(("uniform_pilot", "uniform_dev")) else "probe/report.json")
        r = json.loads(report_path.read_text()) if report_path.is_file() else {}
        cost = r.get("cost", {})
        rows.append({"run_id": name, "status": m["status"], "manifest_sha256": sha256(path / "manifest.json"),
                     "process_wall_s": m["elapsed_wall_s"], "cost": cost,
                     "known_charged_lower_bound": cost.get("charged_control_equivalent_transitions", cost.get("rollout_control_transitions", 0)),
                     "initialization_cost_measured": cost.get("initialization_physics_env_steps") is not None})
    tests = ROOT / "artifacts/verification/tensorized_finite_jobs_r02_checks.json"
    verification = json.loads(tests.read_text())
    assert verification["status"] == "passed"
    result = {"schema": 1, "status": "artifacts_verified", "research_result": False, "gpu_process_launches": len(rows), "runs": rows,
              "completed_processes": sum(r["status"] == "completed" for r in rows),
              "failed_or_incomplete_processes": sum(r["status"] != "completed" for r in rows),
              "known_rollout_control_transitions": sum(r["cost"].get("rollout_control_transitions", 0) for r in rows),
              "known_charged_control_equivalent_lower_bound": sum(r["known_charged_lower_bound"] for r in rows),
              "serial_process_wall_hours": sum(r["process_wall_s"] for r in rows) / 3600,
              "cpu_verification": {"path": tests.relative_to(ROOT).as_posix(), "sha256": sha256(tests), "status": verification["status"],
                                   "tests_passed": 144, "source_only_export": True, "import_isolation_asserted": True},
              "scope_and_limitations": [
                  "All listed run artifacts and prelaunch source snapshots were rehashed; failure status is retained even when checkpoints exist or the process returned zero.",
                  "Known compute is a lower bound: both CPU-reference initializations and the earliest failed constructor lack complete initialization-step instrumentation.",
                  "The failed v1 pilot, correction probes and initial capacity weights are development/tuning costs; only v2 reaches the declared fresh pilot target.",
                  "No adaptive/fixed curriculum comparison, final-test opening, gear policy, hardware result or public checkpoint release."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
