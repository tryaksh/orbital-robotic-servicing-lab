"""Offline desired/applied/measured motion diagnosis for a saved development job."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.protocol import sha256  # noqa: E402


def diagnose(directory, index):
    report = json.loads((directory / "probe/report.json").read_text())
    controls = [json.loads(line) for line in (directory / "probe/control_samples.jsonl").read_text().splitlines()]
    bound = report["geometry"]["position_action_bound_m"]
    fixed = report["initial_fixed_pos"][index]
    noise = report["initial_state"]["init_fixed_pos_obs_noise"][index]
    threshold = report["initial_state"]["pos_threshold"][index]
    phases = {}
    for phase in ("insert", "withdraw", "realign", "search", "seat"):
        rows = [r for r in controls if r["phase"][index] == phase and r["active_before_step"][index]]
        if not rows:
            continue
        part_xy = [[r["part_pos"][index][a] - fixed[a] for a in range(2)] for r in rows]
        tip_xy = [[r["fingertip_pos"][index][a] - fixed[a] for a in range(2)] for r in rows]
        desired = [[r["commanded_action"][index][a] * bound + noise[a] for a in range(2)] for r in rows]
        applied = [[r["applied_action"][index][a] * bound + noise[a] for a in range(2)] for r in rows]
        errors = [math.dist(a, b) for a, b in zip(applied, tip_xy, strict=True)]
        phases[phase] = {
            "control_steps": len(rows), "time_range_s": [rows[0]["step"] / 15, rows[-1]["step"] / 15],
            "closest_part_base_xy_m": min(math.hypot(*x) for x in part_xy),
            "part_xy_range_m": [[min(x[a] for x in part_xy), max(x[a] for x in part_xy)] for a in range(2)],
            "desired_target_xy_range_m": [[min(x[a] for x in desired), max(x[a] for x in desired)] for a in range(2)],
            "ema_applied_target_xy_range_m": [[min(x[a] for x in applied), max(x[a] for x in applied)] for a in range(2)],
            "median_applied_target_tip_error_m": median(errors), "max_applied_target_tip_error_m": max(errors),
            "endpoint_xy_threshold_exceedances": sum(any(abs(a[k] - b[k]) > threshold[k] for k in range(2)) for a, b in zip(applied, tip_xy, strict=True)),
            "median_noisy_wrist_load_n": median(r["controller_state"][index]["noisy_load_mean_n"] for r in rows),
            "last_integral_xy_m": rows[-1]["controller_state"][index]["integral_xy_m"],
        }
    return {"schema": 1, "research_result": False, "source_run": directory.name, "job_index": index,
            "inputs": {name: sha256(directory / "probe" / name) for name in ("report.json", "control_samples.jsonl")},
            "geometry": report["geometry"], "fixture_bias_m": noise, "position_threshold_m": threshold,
            "phases": phases,
            "scope_and_limitations": ["Offline evaluator diagnosis; true part/fixture positions do not enter controller decisions.",
                                      "Applied targets reconstructed after EMA and before position-error clipping. Endpoint threshold checks do not prove every physics-step target was unclipped.",
                                      "This motion diagnosis cannot attribute failure to load, dead zone, or search coverage alone. Feasibility requires the separate matched contact/withdrawal/completion witness."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.run, args.job)
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
