"""CPU-only inspection of a completed cohort's checkpoint during a bounded run."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "pilot/progress.json").read_text())
    cohort = report["cohorts"][-1]
    count = len(report["cohorts"])
    name = "budget_checkpoint.pt" if count == 22 else f"partial_cohort_{count}.pt"
    checkpoint = torch.load(args.run / "pilot" / name, map_location="cpu", weights_only=True)
    tensors = list(checkpoint["model"].values())
    tensors.extend(value for state in checkpoint["optimizer"]["state"].values()
                   for value in state.values() if isinstance(value, torch.Tensor))
    result = {"completed_cohorts": count, "checkpoint": name,
              "checkpoint_status": checkpoint["status"],
              "checkpoint_and_optimizer_finite": all(bool(torch.isfinite(v).all()) for v in tensors),
              "checkpoint_matches_report_cost": checkpoint["cost"] == report["cost"],
              "charged_transitions": report["cost"]["charged_control_equivalent_transitions"],
              "active_transitions": report["cost"]["active_control_transitions"],
              "absorbing_fraction": report["cost"]["absorbing_control_transitions"] / report["cost"]["rollout_control_transitions"],
              "last_cohort_outcomes": dict(Counter(j["outcome"] for j in cohort["jobs"])),
              "last_cohort_nominal_outcomes": dict(Counter(j["outcome"] for c, j in zip(cohort["cases"], cohort["jobs"], strict=True) if c["bin_id"] == "nominal")),
              "all_completed_initializations_valid": all(v["valid"] for c in report["cohorts"] for v in c["initialization"]),
              "last_cohort_action_std": cohort["policy_action_std"],
              "scope": "Changing stochastic training policy; these outcomes are not development competence scores."}
    if args.output:
        with args.output.open("x") as f:
            f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["checkpoint_and_optimizer_finite"] and result["checkpoint_matches_report_cost"] and result["all_completed_initializations_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
