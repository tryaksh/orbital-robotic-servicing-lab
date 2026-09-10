"""Condense verified physics sensitivity, all-attempt cost and native force evidence."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import LAB_COMMIT, sha256, write_json  # noqa: E402
from scripts.run_experiment import git, snapshot_source  # noqa: E402
from scripts.verify_training_contract import verify_assets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    detail = run / "force_review"
    if output.exists() or detail.exists():
        parser.error("Preserve previous analysis and evidence")
    detail.mkdir()
    torch.set_num_threads(1)
    verified_path = run / "paired_verification.json"
    verified = json.loads(verified_path.read_text())
    if verified["status"] != "verified":
        raise ValueError("Native paired verification must pass before interpretation")
    registration_path = ROOT / "configs/learned_physics_validation_v3.json"
    registration = json.loads(registration_path.read_text())
    provenance = {"source_commit": git(ROOT, "rev-parse", "HEAD"),
        "source_dirty": bool(git(ROOT, "status", "--porcelain")),
        "source_hashes": snapshot_source(detail / "source.zip"), "command": [sys.executable, *sys.argv],
        "created_at_utc": datetime.now(UTC).isoformat(), "simulator_transitions": 0,
        "input_verification_sha256": sha256(verified_path)}
    write_json(detail / "analysis_manifest.json", provenance)
    summaries, resources, force_rows, hold_medians, final_manifests = {}, [], [], {}, []
    for row in registration["run_order"]:
        label, hz = row["label"], 120 * row["refinement"]
        path = run / label
        manifest = verify_assets(path)
        final_manifests.append(manifest)
        report = json.loads((path / "validation/report.json").read_text())
        data = torch.load(path / "validation/trajectory.pt", map_location="cpu", weights_only=True)
        saved = verified["runs"][label]
        summary = {"controller": report["policy_phase_label"], "mode": report["mode"],
            "whole_job": saved["whole_job"],
            "nominal_completions": sum(j["success"] for c, j in zip(report["cases"], report["jobs"], strict=True) if c["bin_id"] == "nominal"),
            "nominal_requests": 4,
            "actual_contact_stalls": sum(w["witnessed_failure_step"] is not None for w in report["recovery_witness"]),
            "original_withdrawal_recoveries_all_jobs": saved["post_stall"]["original_withdrawal_recoveries"]}
        if report["mode"] != "direct_learned":
            summary["fixed_prefix_endpoint"] = saved["post_stall"]
        summaries[label] = summary
        physics = data["physics"]
        if report["mode"] == "prefix_learned":
            hold_medians[hz] = physics[hz // 2:hz, :, 2].median(dim=0).values.tolist()
        for i, (case, job) in enumerate(zip(report["cases"], report["jobs"], strict=True)):
            if job["outcome"] != "force_abort":
                continue
            terminal = round(job["elapsed_s"] * hz)
            window = physics[max(0, terminal - hz // 10):terminal, i]
            force_rows.append({"run": label, "case_id": case["case_id"], "bin_id": case["bin_id"],
                "elapsed_s": job["elapsed_s"], "native_step": terminal,
                "phase": report["script_phases"][(terminal - 1) // (8 * row["refinement"])][i],
                "between_external_sensor_ticks": terminal % row["refinement"] != 0,
                "raw_wrist_at_abort_n": float(window[-1, 2]), "filtered_wrist_at_abort_n": float(window[-1, 3]),
                "fixture_at_abort_n": float(window[-1, 11]),
                "fixture_last_100ms_max_n": float(window[:, 11].max()),
                "wrist_last_100ms_mean_n": float(window[:, 2].mean()),
                "fixture_impulse_at_abort_n_s": float(data["native"]["contact_impulse"][terminal - 1, i, 0, 0].norm())})
        gpu = json.loads((path / "gpu_resources.json").read_text())
        resources.append({"run": label, "wall_s": manifest["elapsed_wall_s"],
            "rollout_wall_s": report["rollout_wall_s"], "peak_gpu_mib": gpu["peak_used_mib"],
            "minimum_free_ram_mib": min(s["free_ram_mib"] for s in gpu["samples"] if s["free_ram_mib"] is not None),
            "sustained_control_transitions_per_s_min_max": [min(x["control_transitions_per_s"] for x in report["throughput"]), max(x["control_transitions_per_s"] for x in report["throughput"])],
            "native_physics_env_steps": report["cost"]["rollout_physics_env_steps"],
            "active_control_transitions": report["cost"]["active_control_transitions"],
            "absorbing_control_transitions": report["cost"]["absorbing_control_transitions"],
            "charged_reference_transitions": report["cost"]["charged_reference_transitions"]})
    write_json(detail / "force_aborts.json", force_rows)
    force_summary = {}
    for label in summaries:
        rows = [r for r in force_rows if r["run"] == label]
        force_summary[label] = {"force_aborts": len(rows),
            "during_scripted_prefix": sum(r["elapsed_s"] <= 4 and r["phase"] != "learned_mean" for r in rows),
            "between_external_sensor_ticks": sum(r["between_external_sensor_ticks"] for r in rows),
            "with_fixture_contact_at_abort": sum(r["fixture_at_abort_n"] > .1 for r in rows),
            "filtered_force_at_abort_min_max_n": [min(r["filtered_wrist_at_abort_n"] for r in rows), max(r["filtered_wrist_at_abort_n"] for r in rows)]}
    hold_ratios = [b / a for a, b in zip(hold_medians[120], hold_medians[240], strict=True)]
    attempts = []
    for queue_name in ("learned_physics_v1_r01", "learned_physics_v2_r01", run.name):
        for path in sorted((ROOT / "artifacts/assembly" / queue_name).glob("*/manifest.json")):
            m = verify_assets(path.parent)
            r = json.loads((path.parent / "validation/report.json").read_text())
            attempts.append({"run_id": m["run_id"], "status": m["status"], "source_commit": m["source_commit_at_start"],
                "manifest_sha256": sha256(path), "requested_jobs": 28,
                "jobs_with_terminal_records": len(r.get("jobs", [])), "failed_backend_requests": 28 if r["status"] == "failed" else 0,
                "charged_reference_transitions": r["cost"]["charged_reference_transitions"],
                "charged_control_equivalent_transitions": r["cost"]["charged_control_equivalent_transitions"],
                "initialization_physics_env_steps": r["cost"]["initialization_physics_env_steps"],
                "rollout_physics_env_steps": r["cost"]["rollout_physics_env_steps"]})
    charged = sum(r["charged_reference_transitions"] for r in attempts)
    checks = {
        "all_final_sources_clean": all(not m["source_dirty_at_start"] for m in final_manifests),
        "same_final_prelaunch_commit": len({m["source_commit_at_start"] for m in final_manifests}) == 1,
        "same_exact_archived_sources": all(m["source_hashes"] == final_manifests[0]["source_hashes"] for m in final_manifests),
        "same_pinned_upstream": all(m["upstream_commit"] == LAB_COMMIT and m["upstream_source_sha256"] == final_manifests[0]["upstream_source_sha256"] for m in final_manifests),
        "same_installed_source": all(m["installed_source_sha256"] == registration["installed_source_sha256"] for m in final_manifests),
        "same_frozen_checkpoint": all(m["input_checkpoint"]["sha256"] == registration["checkpoint_sha256"] for m in final_manifests),
        "same_initialization_configuration": len({sha256(run / row["label"] / "validation/initialization_environment.yaml") for row in registration["run_order"]}) == 1,
        "failed_backend_evidence_unchanged": all(sha256(ROOT / p) == h for p, h in registration["prior_failed_run_sha256"].items()),
        "all_ten_attempts_charged": len(attempts) == 10 and charged == registration["total_session_expected_charged_reference_transitions"] == 153545.,
    }
    next_action = "Preregister and execute a bounded 120/240/480 Hz contact-impact diagnostic on fixed development cases with the frozen scripted prefix, 120 Hz external servo/sensors and native raw 20 N aborts. Compare native force peaks, impulses, contact durations and inter-sensor peaks to identify the source of timestep sensitivity. Keep controller settings, geometry, gravity and the failed competence gate unchanged; do not launch the training screen or open final tests."
    result = {"schema": 1, "status": "verified_material_physics_sensitivity" if all(checks.values()) else "review_check_failed",
        "research_result": False, "recorded_at_utc": datetime.now(UTC).isoformat(),
        "registration": {"path": registration_path.relative_to(ROOT).as_posix(), "sha256": sha256(registration_path)},
        "checkpoint_sha256": registration["checkpoint_sha256"], "final_prelaunch_commit": final_manifests[0]["source_commit_at_start"],
        "summary": summaries, "pairing": verified["pairing"],
        "timestep_sensitivity": {k: {a: b for a, b in v.items() if a != "cases"} for k, v in verified["sensitivity"].items() if k != "direct_learned"},
        "direct_completion_change": summaries["240_direct_learned"]["whole_job"]["completions"] - summaries["120_direct_learned"]["whole_job"]["completions"],
        "force_summary": force_summary,
        "free_space_hold_control": {"interval_s": [.5, 1.], "cases": 28,
            "fine_to_coarse_median_force_ratio_min_median_max": [min(hold_ratios), statistics.median(hold_ratios), max(hold_ratios)],
            "interpretation": "Free-space hold loads are comparable and native impulse conversion passes; this argues against a simple constant force-scale error. It does not prove dynamic force calibration or contact convergence."},
        "verification": {"native_and_paired_checks_passed": len(verified["checks"]) + sum(len(r["checks"]) for r in verified["runs"].values()),
            "provenance_checks": checks, "maximum_captured_initial_state_difference": max(max(v.values()) for v in verified["initial_maximum_absolute_differences"].values()),
            "paired_report": {"path": verified_path.relative_to(ROOT).as_posix(), "sha256": sha256(verified_path)},
            "force_review": {"path": (detail / "force_aborts.json").relative_to(ROOT).as_posix(), "sha256": sha256(detail / "force_aborts.json")},
            "analysis_manifest": {"path": (detail / "analysis_manifest.json").relative_to(ROOT).as_posix(), "sha256": sha256(detail / "analysis_manifest.json")}},
        "resources": resources, "all_attempts": attempts,
        "cost": {"final_six_run_reference_transitions": 114807., "prior_probe_reference_transitions": 38738.,
            "session_charged_reference_transitions": charged,
            "session_control_time_equivalent_transitions": sum(r["charged_control_equivalent_transitions"] for r in attempts),
            "cumulative_tracked_reference_transition_lower_bound": 24841670.25 + charged,
            "training_transitions": 0, "final_validation_requests": 168, "all_attempted_requests": 280,
            "prior_complete_probe_requests": 84, "failed_backend_requests": 28,
            "unit": "Eight executed native physics environment steps per reference transition. Initializations, absorbing slots and all failed attempts included; old cumulative accounting remains a lower bound."},
        "decision": verified["decision"],
        "superseded_interpretation": {"field": "direct_learned reports' post_stall_summary.prefix_script_completions",
            "correction": "The shared diagnostic helper uses a hypothetical four-second boundary in direct mode. Those early completions were produced by learned actions, not a script. This evidence reports direct whole jobs and actual witness incidence, and excludes that inapplicable field; saved reports remain unchanged.",
            "pre_outcome_scope_note": "artifacts/assembly/learned_physics_v2_r01/reporting_scope_note.json"},
        "scope_and_limitations": [*registration["scope_and_limitations"],
            "The complete validation is six fresh common-worker runs. Three prior complete coarse probes and one failed fine initialization are preserved and charged separately. Repeated cases and different controllers must not be pooled into a deployment reliability rate.",
            "Most additional prefix failures precede learned control. Conditional exposure changes from 17 to five jobs; on the same five cases, coarse learned/retry each complete five while fine learned/retry complete three/two.",
            "Short peaks and actor filtering do not excuse a native force abort. Lower mean time at 240 Hz includes early failures and is not a throughput improvement.",
            "No hardware calibration, contact convergence, adaptive-curriculum benefit or publication-worthiness is established. Training remains blocked by material physics sensitivity."],
        "next_action": next_action}
    write_json(output, result)
    print(json.dumps({"status": result["status"], "provenance_checks": checks, "cost": result["cost"]}), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
