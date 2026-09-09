"""Verify the six predeclared support runs and every sampled bin/value."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.evaluation import JobCriteria  # noqa: E402
from scripts.probe_training import replay_physics  # noqa: E402
from scripts.verify_training_paths import verified_run  # noqa: E402


def main():
    output = ROOT / "evidence/training_support_v1.json"
    if output.exists():
        raise ValueError("Preserve the existing support evidence")
    groups, all_checks = [], []
    totals = {"retry": Counter(), "continue": Counter()}
    for grid in ("low", "interior", "high"):
        runs = {}
        for controller in ("retry", "continue"):
            directory = ROOT / "artifacts/assembly" / f"training_support_{grid}_{controller}_r01"
            manifest, report, arrays = verified_run(directory)
            replay_physics(arrays["physics"], JobCriteria(**report["criteria"]), report["jobs"], "paired",
                           report["general_recovery_witness"])
            runs[controller] = (manifest, report, arrays)
            totals[controller].update(j["outcome"] for j in report["jobs"])
        rm, rr, ra = runs["retry"]
        cm, cr, ca = runs["continue"]
        source_names = [n for n in rm["source_hashes"] if n.startswith("src/assembly_recovery/")]
        source_names += ["scripts/probe_training_paths.py", "scripts/probe_training.py", "scripts/compare_recovery.py",
                         "configs/study.json", "configs/retry_unload_realign_v1.json"]
        checks = {"all_initializations_valid": all(v["valid"] for r in (rr, cr) for v in r["initialization"]),
                  "same_criteria_geometry_cases": all(rr[k] == cr[k] for k in ("criteria", "geometry", "fault_cases")),
                  "identical_initial_state": all(torch.equal(v, ca["initial"][k]) for k, v in ra["initial"].items()),
                  "same_behavior_source": all(rm["source_hashes"].get(n) == cm["source_hashes"].get(n) for n in source_names),
                  "pretrigger_trajectories_match": True, "saved_hashes_and_cpu_replay_verified": True}
        bins = defaultdict(lambda: {"jobs": 0, "retry_completions": 0, "continued_completions": 0,
                                    "scripted_witnesses": 0, "general_witnesses": 0})
        for i, (case, rj, cj) in enumerate(zip(rr["fault_cases"], rr["jobs"], cr["jobs"], strict=True)):
            trigger = rr["controllers"][i]["trigger"]
            prefix = round(trigger["time_s"] / (8 * rr["criteria"]["physics_dt"])) if trigger else 450
            checks["pretrigger_trajectories_match"] &= all(torch.equal(ra[n][:prefix, i], ca[n][:prefix, i])
                                                          for n in ("part_pos", "part_quat", "commanded_action"))
            b = bins[case["bin_id"]]
            b["jobs"] += 1
            b["retry_completions"] += rj["success"]
            b["continued_completions"] += cj["success"]
            b["scripted_witnesses"] += rr["witnesses"][i]["witnessed_complete_recovery"]
            b["general_witnesses"] += rr["general_recovery_witness"][i]["witnessed_complete_recovery"]
        checks["at_least_one_completion_per_bin_value"] = all(b["retry_completions"] > 0 for b in bins.values())
        all_checks += list(checks.values())
        groups.append({"grid": grid, "seed": rr["seed"], "checks": checks, "bins": dict(bins),
                       "run_ids": [rm["run_id"], cm["run_id"]],
                       "outcomes": {"retry": dict(Counter(j["outcome"] for j in rr["jobs"])),
                                    "continue": dict(Counter(j["outcome"] for j in cr["jobs"]))}})
    result = {"schema": 1, "status": "verified" if all(all_checks) else "support_gate_failed",
              "research_result": False, "jobs_per_arm": 96, "all_initializations": 192,
              "groups": groups, "total_outcomes": {k: dict(v) for k, v in totals.items()},
              "scope_and_limitations": [
                  "Unchanged scripted development controllers at sampled lower/interior/upper values; no learned recovery result.",
                  "Every requested failure and force abort remains in the denominator.",
                  "Sampled coverage and geometry feasibility do not prove every continuous point or Gaussian nuisance tail.",
                  "Historical scripted-phase and new controller-independent recovery witnesses remain separate metrics.",
                  "Final-test seeds and combined faults remain unopened."]}
    with output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
