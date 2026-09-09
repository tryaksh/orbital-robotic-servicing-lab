"""Summarize all witnessed prefix failures and plot one declared descriptive case."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import sha256, write_json  # noqa: E402
from scripts.run_experiment import git, snapshot_source  # noqa: E402
from scripts.verify_training_contract import verify_assets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verified-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.verified_evidence.read_text())
    if evidence["status"] != "verified":
        raise ValueError("Verify the paired assay before reviewing its motion")
    args.output.mkdir(parents=True, exist_ok=False)
    reports, traces, manifests = {}, {}, {}
    manifest = {"status": "starting", "created_at_utc": datetime.now(UTC).isoformat(),
                "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"),
                "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
                "source_hashes": snapshot_source(args.output / "source.zip"),
                "input_evidence_sha256": sha256(args.verified_evidence),
                "python": sys.version, "torch": torch.__version__, "matplotlib": matplotlib.__version__,
                "simulator_transitions": 0, "command": [sys.executable, *sys.argv], "run_directory": str(args.run.resolve()),
                "example_rule": "Lexicographically first witnessed-prefix-failure case completed by learned actions and failed by continued insertion; all eligible cases remain in the table."}
    write_json(args.output / "analysis_manifest.json", manifest)
    torch.set_num_threads(1)
    for arm in ("learned", "retry", "continue"):
        path = args.run / arm
        manifests[arm] = verify_assets(path)
        if sha256(path / "manifest.json") != evidence["arms"][arm]["manifest_sha256"]:
            raise ValueError("Verified input run changed")
        reports[arm] = json.loads((path / "diagnosis/report.json").read_text())
        traces[arm] = torch.load(path / "diagnosis/trajectory.pt", map_location="cpu", weights_only=True)
    rows = []
    learned = reports["learned"]
    for i, (case, job, witness) in enumerate(zip(learned["cases"], learned["jobs"], learned["recovery_witness"], strict=True)):
        failure = witness["witnessed_failure_step"]
        if failure is None or failure > 480 or job["elapsed_s"] <= 4:
            continue
        row = {"case_id": case["case_id"], "case_index": i, "bin_id": case["bin_id"],
               "witnessed_failure_s": failure / 120, "arms": {}}
        for arm in reports:
            j = reports[arm]["jobs"][i]
            w = reports[arm]["recovery_witness"][i]
            end = round(j["elapsed_s"] * 120)
            native = traces[arm]["physics"][failure - 1:end, i]
            part = traces[arm]["part_relative_xyz"][:, i]
            row["arms"][arm] = {"outcome": j["outcome"], "completion": j["success"],
                "elapsed_s": j["elapsed_s"], "strict_withdrawal_recovery": w["witnessed_complete_recovery"],
                "maximum_rise_after_failure_mm": 1000 * float((native[:, 12] - native[0, 12]).max()),
                "lateral_part_shift_after_handoff_mm": 1000 * float(torch.linalg.vector_norm(part[(end - 1) // 8, :2] - part[59, :2]))}
        rows.append(row)
    successful = [r["arms"]["learned"] for r in rows if r["arms"]["learned"]["completion"]]
    candidates = sorted((r for r in rows if r["arms"]["learned"]["completion"] and not r["arms"]["continue"]["completion"]), key=lambda r: r["case_id"])
    result = {"schema": 1, "status": "verified", "research_result": False, "all_prefix_failure_cases": rows,
              "successful_learned_cases": len(successful),
              "maximum_rise_range_mm": [min(x["maximum_rise_after_failure_mm"] for x in successful), max(x["maximum_rise_after_failure_mm"] for x in successful)] if successful else None,
              "lateral_shift_range_mm": [min(x["lateral_part_shift_after_handoff_mm"] for x in successful), max(x["lateral_part_shift_after_handoff_mm"] for x in successful)] if successful else None,
              "existing_withdrawal_requirement_mm": 5., "input_evidence_sha256": manifest["input_evidence_sha256"],
              "interpretation": "Post-stall completion and the existing five-millimetre withdrawal witness are different measured endpoints. Preserve both; the original competence gate remains unchanged.",
              "scope": "All shared prefix failures retained. One descriptive example is selected by the recorded rule; it is not an independent trial or proof of a curriculum advantage."}
    if candidates:
        chosen = candidates[0]
        index = chosen["case_index"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        colors = {"learned": "#15803d", "retry": "#2563eb", "continue": "#b45309"}
        for arm, label in (("learned", "Learned mean after prefix"), ("retry", "Unchanged scripted retry"), ("continue", "Continued insertion")):
            j = reports[arm]["jobs"][index]
            end = round(j["elapsed_s"] * 120)
            native = traces[arm]["physics"][:end, index]
            time = torch.arange(1, end + 1).numpy() / 120
            axes[0].plot(time, -native[:, 13].numpy() * 1000, color=colors[arm], label=f"{label}: {j['outcome']}")
            axes[1].plot(time, native[:, 2].numpy(), color=colors[arm])
            controls = (end + 7) // 8
            xy = traces[arm]["part_relative_xyz"][:controls, index, :2]
            axes[2].plot(torch.arange(1, controls + 1).numpy() / 15, torch.linalg.vector_norm(xy, dim=-1).numpy() * 1000, color=colors[arm])
        for ax in axes:
            ax.axvspan(0, 4, color="gray", alpha=.12)
            ax.axvline(4, color="black", linestyle="--", linewidth=1)
            ax.axvline(chosen["witnessed_failure_s"], color="#be123c", linestyle=":", linewidth=1)
            ax.grid(alpha=.2)
            ax.set_xlim(0, 30)
        axes[1].axhline(20, color="#be123c", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Peg base below\nfixture top (mm)")
        axes[1].set_ylabel("Raw wrist load (N)")
        axes[2].set_ylabel("Lateral peg offset\nfrom fixture axis (mm)")
        axes[2].set_xlabel("Job time (s); shaded region is the common scripted prefix")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle("Completion after witnessed contact stall\n" + chosen["case_id"] + "; dotted line = stall, dashed line = fixed handoff", fontsize=12)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / ("matched_prefix_motion." + suffix), dpi=160)
        plt.close(fig)
        result["example"] = chosen["case_id"]
    write_json(args.output / "review.json", result)
    manifest.update(status="completed", output_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.name != "analysis_manifest.json"})
    write_json(args.output / "analysis_manifest.json", manifest)
    print(json.dumps({k: v for k, v in result.items() if k != "all_prefix_failure_cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
