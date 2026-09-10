"""Audit a completed uniform training log; no simulation or counterfactual learning."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(cohorts):
    groups = {}
    for cohort in cohorts:
        for case, job, witness in zip(cohort["cases"], cohort["jobs"], cohort["recovery_witness"], strict=True):
            if case["split"] != "training" or not job["finished"]:
                raise ValueError("Only finalized training jobs may supply this exploratory audit")
            row = groups.setdefault(case["bin_id"], {"jobs": 0, "completions": 0, "outcomes": Counter(),
                                                   "logged_contact_stalls": 0, "logged_post_contact_completions": 0})
            row["jobs"] += 1
            row["completions"] += job["success"]
            row["outcomes"][job["outcome"]] += 1
            witnessed = witness["witnessed_failure_step"] is not None
            row["logged_contact_stalls"] += witnessed
            row["logged_post_contact_completions"] += bool(witnessed and job["success"] and not job["forbidden_events"])
    for row in groups.values():
        row["success_fraction"] = row["completions"] / row["jobs"]
        row["smoothed_p"] = (row["completions"] + 1) / (row["jobs"] + 2)
        row["p_times_one_minus_p"] = row["smoothed_p"] * (1 - row["smoothed_p"])
    fault_ids = sorted(k for k in groups if k != "nominal")
    score_sum = sum(groups[k]["p_times_one_minus_p"] for k in fault_ids)
    for k in fault_ids:
        groups[k]["illustrative_fault_share"] = .2 / len(fault_ids) + .8 * groups[k]["p_times_one_minus_p"] / score_sum
    return groups


def main():
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if args.output.exists():
        parser.error("Preserve the previous audit")
    if report["status"] != "completed" or report["seed"] != 170 or len(report["cohorts"]) != 22:
        parser.error("Expected the completed 22-cohort uniform v4 run, not partial weights")
    windows = []
    for end in range(1, len(report["cohorts"]) + 1):
        start = max(0, end - 2)
        rows = summarize(report["cohorts"][start:end])
        if sum(r["jobs"] for r in rows.values()) != (end - start) * 1024:
            raise ValueError("All requested training jobs must be retained")
        windows.append({"completed_cohorts": end, "window_cohorts": list(range(start + 1, end + 1)), "bins": rows})
    result = {"schema": 1, "status": "verified_log_accounting", "research_result": False,
              "source_report": args.report.as_posix(),
              "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "controller": report["controller"], "source_training_cost": report["cost"],
              "additional_simulator_transitions": 0, "all_training_jobs": summarize(report["cohorts"]),
              "windows": windows,
              "scope_and_limitations": [
                  "Exploratory summaries of one changing stochastic training policy, not frozen-policy development or final-test scores.",
                  "Illustrative shares use Beta(1,1) smoothing, the two most recent cohorts and a 20 percent uniform mixture. They describe a score only; no alternate policy was trained or evaluated.",
                  "Intermediate success can reflect noise or irreducible failure. This audit cannot estimate learning progress, physical recoverability or a causal sampler benefit.",
                  "Contact labels are inherited from the recorded tensor witness; this audit does not independently replay all training physics.",
                  "All source training cost remains charged as shared prior development; offline inspection adds no simulator transitions."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8", newline="\n") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "last_two_cohorts": windows[-1]["bins"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
