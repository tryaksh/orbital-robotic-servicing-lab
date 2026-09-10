"""Export a standalone two-panel figure from verified contact and capacity artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "artifacts/assembly/research_cycle_20260910_r01/figures"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified(path, *, capacity=False):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = "verified" if capacity else "verified_diagnostic"
    if data.get("status") != expected:
        raise ValueError(f"Only independently verified completed artifacts may be plotted: {path}")
    checks = data.get("runs", []) if capacity else data.get("verification", [])
    if not checks or not all(row.get("checks") and all(row["checks"].values()) for row in checks):
        raise ValueError(f"The underlying verification checks must all pass: {path}")
    return data


def prefix_rows(paths, expected_hz):
    rows = {}
    for path in paths:
        data = read_verified(path)
        for run in data["runs"]:
            hz, summary, cases = run["physics_hz"], run["prefix_summary"], run["cases"]
            if hz in rows:
                raise ValueError(f"Duplicate native resolution {hz} Hz; do not pool duplicate prefixes")
            if summary["requests"] != 28 or len(cases) != 28 or summary["forbidden_events"] != 0:
                raise ValueError("Every plotted prefix must retain all 28 requests without forbidden state changes")
            force_aborts = sum(case["outcome"] == "force_abort" for case in cases)
            censored = sum(case["outcome"] == "probe_end" for case in cases)
            if force_aborts != summary["force_aborts"] or censored != summary["active_at_prefix_end"]:
                raise ValueError("Per-case outcomes and prefix summary disagree")
            if any(case["active_end_s"] > 4 + 1e-6 for case in cases):
                raise ValueError("This panel is restricted to four-second prefix diagnostics")
            rows[hz] = {"physics_hz": hz, "requests": 28, "force_aborts": force_aborts,
                        "censored_at_four_seconds": censored, "prefix_completions": summary["completions_before_prefix_end"],
                        "case_ids": [case["case_id"] for case in cases]}
    if sorted(rows) != expected_hz:
        raise ValueError(f"Require exactly native resolutions {expected_hz}, received {sorted(rows)}")
    return [rows[hz] for hz in expected_hz]


def collect_data(original_paths, servo_path, reference_path, capacity_path):
    original = prefix_rows(original_paths, [120, 240, 480, 960])
    faster_servo = prefix_rows([servo_path], [480, 960])
    reference = prefix_rows([reference_path], [480, 960])
    if read_verified(servo_path).get("verifier_version") != "servo_impact_review_v1":
        raise ValueError("Expected the verified registered 480 Hz servo cause test")
    if read_verified(reference_path).get("verifier_version") != "approach_slew_review_v1":
        raise ValueError("Expected the verified final shared 20 mm/s reference cause test")
    ordered_cases = original[0]["case_ids"]
    if len(set(ordered_cases)) != 28 or any(row["case_ids"] != ordered_cases for row in original + faster_servo + reference):
        raise ValueError("All controller conditions and timesteps must retain the same declared 28 cases")
    capacity = read_verified(capacity_path, capacity=True)
    runs = sorted(capacity["runs"], key=lambda row: row["specification"]["num_envs"])
    if [run["specification"]["num_envs"] for run in runs] != [1024, 2048]:
        raise ValueError("Expected the independently verified 1024/2048 capacity pair")
    capacity_rows = [{"num_envs": run["specification"]["num_envs"],
                      "throughput": run["throughput"]["startup_inclusive"],
                      "charged_transitions": run["measured_cost"]["charged_control_equivalent_transitions"],
                      "active_samples": run["measured_cost"]["active_control_transitions"],
                      "absorbing_fraction": run["absorbing_rollout_fraction"],
                      "optimizer_steps": run["optimizer_steps"], "resources": run["resources"]} for run in runs]
    return {"original_120hz_servo": original, "faster_480hz_servo": faster_servo,
            "servo_480hz_reference_20mm_s": reference, "capacity": capacity_rows,
            "contact_controller": "Frozen scripted first insertion attempt; no candidate policy supplies actions.",
            "contact_horizon_s": 4, "full_job_deadline_s": 30, "raw_wrist_abort_n": 20,
            "reference_gate_status": read_verified(reference_path)["decision"],
            "trace_selection": "No individual trajectory or best-case demonstration is selected.",
            "capacity_scope": "Unequal early-training work and active samples; no policy quality or curriculum comparison."}


def draw(data, stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, (contact, capacity) = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=0.068, right=0.98, top=0.83, bottom=0.35, wspace=0.23)
    fig.suptitle("Contact diagnostics and representative training capacity", x=0.068, y=0.968,
                 ha="left", fontsize=17, fontweight="semibold")
    fig.text(0.068, 0.915, "Franka + FORGE simulation  |  Development measurements  |  10 September 2026", color="#526070", fontsize=10)
    styles = [
        ("original_120hz_servo", "Original: 120 Hz servo", "#3569a8", "o", "-", (-18, 10)),
        ("faster_480hz_servo", "480 Hz servo", "#b3651f", "s", "--", (0, 24)),
        ("servo_480hz_reference_20mm_s", "480 Hz servo + 20 mm/s reference", "#247d69", "^", "-.", (18, -14)),
    ]
    for key, label, color, marker, linestyle, offset in styles:
        rows = data[key]
        x = [row["physics_hz"] for row in rows]
        y = [row["force_aborts"] for row in rows]
        contact.plot(x, y, label=label, color=color, marker=marker, markersize=7.5,
                     linewidth=2, linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.7)
        for hz, count in zip(x, y, strict=True):
            contact.annotate(str(count), (hz, count), xytext=offset, textcoords="offset points",
                             ha="center", va="center", fontsize=10, color=color, fontweight="semibold")
    contact.set_xscale("log", base=2)
    contact.set_xticks([120, 240, 480, 960], labels=["120", "240", "480", "960"])
    contact.set_xlim(103, 1120)
    contact.set_ylim(-1.5, 30.5)
    contact.set_yticks([0, 4, 8, 12, 16, 20, 24, 28])
    contact.set_xlabel("Native physics frequency (Hz)", labelpad=9)
    contact.set_ylabel("Force aborts among 28 requests")
    contact.set_title("A   Scripted insertion: first four seconds", loc="left", pad=14, fontweight="semibold")
    contact.grid(axis="y", color="#dde3ea", linewidth=0.7)
    contact.set_axisbelow(True)
    contact.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="white", fontsize=9,
                   borderpad=0.55, handlelength=2.6)
    rows = data["capacity"]
    positions = np.arange(2)
    width = 0.31
    metrics = ["active_samples_per_s", "charged_transitions_per_s"]
    for i, (row, color) in enumerate(zip(rows, ["#8c96a5", "#6d54a3"], strict=True)):
        values = [row["throughput"][key] for key in metrics]
        bars = capacity.bar(positions + (i - 0.5) * width, values, width=width, color=color,
                            label=f"{row['num_envs']:,} environments")
        capacity.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=4, fontsize=10)
    for i, key in enumerate(metrics):
        gain = rows[1]["throughput"][key] / rows[0]["throughput"][key] - 1
        highest = max(row["throughput"][key] for row in rows)
        capacity.annotate(f"+{gain:.1%}", (positions[i], highest), xytext=(0, 32), textcoords="offset points",
                          ha="center", color="#563c8c", fontsize=11, fontweight="semibold")
    capacity.set_xticks(positions, labels=["Active learning samples", "All charged transitions"])
    capacity.set_ylim(0, 2850)
    capacity.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    capacity.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    capacity.set_ylabel("Samples or charged transitions per second")
    capacity.set_title("B   Training capacity, including startup", loc="left", pad=14, fontweight="semibold")
    capacity.grid(axis="y", color="#dde3ea", linewidth=0.7)
    capacity.set_axisbelow(True)
    capacity.legend(loc="upper left", frameon=False, fontsize=9)
    stability = "passed" if data["reference_gate_status"]["finest_prefix_pair_pass"] else "failed"
    support = "passed" if data["reference_gate_status"]["active_prefix_support_floor_pass"] else "failed"
    fig.text(0.068, 0.16,
             "All 28 requests at every point; native raw wrist abort stays at 20 N.\n"
             "Survivors at 4 s are censored, not failed complete 30 s jobs.\n"
             "Servo and reference changes are shared controller conditions.\n"
             f"Final reference pair: stability {stability}; active-request support {support}.",
             fontsize=9.4, color="#374151", linespacing=1.5)
    fig.text(0.575, 0.18,
             "Timing includes initialization, PPO, diagnostics and checkpoint export.\n"
             "Two cohorts per size; twice the charged work at 2,048 environments.\n"
             "Unequal active samples: capacity only, not a learning-quality result.",
             fontsize=9.4, color="#374151", linespacing=1.5)
    fig.text(0.068, 0.055,
             "The contact panel uses scripted actions throughout. No candidate policy controlled these prefixes.\n"
             "The 20 mm/s limit applies to the Cartesian reference, not actual tool speed or a force ceiling.",
             fontsize=9.4, color="#374151", linespacing=1.5)
    metadata = {"Title": "Contact diagnostics and representative training capacity",
                "Author": "Assembly Recovery Lab", "Subject": "Four-second scripted prefixes and unequal-work capacity measurements"}
    fig.savefig(stem.with_suffix(".png"), dpi=200, metadata={"Description": metadata["Subject"]})
    fig.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-review", type=Path, nargs="+", required=True)
    parser.add_argument("--servo-review", type=Path, required=True)
    parser.add_argument("--reference-review", type=Path, required=True)
    parser.add_argument("--capacity-review", type=Path, default=ROOT / "evidence/training_capacity_v2.json")
    parser.add_argument("--output-stem", type=Path, default=FIGURES / "research_cycle_v1")
    parser.add_argument("--check-inputs-only", action="store_true")
    args = parser.parse_args()
    data = collect_data(args.original_review, args.servo_review, args.reference_review, args.capacity_review)
    inputs = args.original_review + [args.servo_review, args.reference_review, args.capacity_review]
    record = {"schema": 1, "figure_version": "research_cycle_v1", "status": "verified_inputs",
              "plot_source_sha256": digest(Path(__file__)), "inputs": [
                  {"path": str(path.resolve()), "sha256": digest(path)} for path in inputs], "plotted_data": data,
              "scope": "Descriptive complete-case development figure. Prefix censoring is distinct from full-job failure; capacity work is unequal."}
    if args.check_inputs_only:
        print(json.dumps(record, indent=2))
        return 0
    stem = args.output_stem.resolve()
    if not stem.is_relative_to(FIGURES.resolve()):
        parser.error("Figure artifacts must remain under the declared research-cycle figures directory")
    if any(stem.with_suffix(suffix).exists() for suffix in (".png", ".pdf", ".json")):
        parser.error("Preserve existing figure outputs; choose a new versioned output stem")
    stem.parent.mkdir(parents=True, exist_ok=True)
    draw(data, stem)
    record.update(status="exported", outputs=[{"path": str(stem.with_suffix(suffix)),
                  "bytes": stem.with_suffix(suffix).stat().st_size, "sha256": digest(stem.with_suffix(suffix))}
                  for suffix in (".png", ".pdf")])
    with stem.with_suffix(".json").open("x", encoding="utf8") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "outputs": record["outputs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
