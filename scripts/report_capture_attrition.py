#!/usr/bin/env python3
"""What is left once the destination bay holds the module, and what records it.

Note on the name: this was written as "capture attrition" and the phase
breakdown says that is only half right. Undelivered episodes split between
capture and extract, and which one dominates depends on the module: at nominal
it is 3 capture against 2 extract, at 120 x 16 mm it is 33 against 7, and at
140 x 26 mm it inverts to 1 against 16. A thinner module is hard to grasp; a
thicker one is hard to pull clear.

With the rack's retention pawls fitted, seating is solved at every configuration
measured so far: precision given delivery is 1.000 at nominal and at both module
cross-sections. The whole remaining failure budget is upstream -- episodes where
the module never reaches the destination at all.

Those episodes are **deterministic within a module geometry**: across every arm
run on 2026-09-05 that kept the nominal cross-section -- retention fitted and
absent, both channel reliefs -- exactly the same environments fail, with
residuals identical to the millimetre. Change the cross-section and a different
set fails, which is the configuration doing its job rather than noise.

And they are almost unrecorded. The trace carries a sampled time series for
transit, insert and settle, and **nothing for capture or extract**: an episode
that fails in either writes one phase-transition row and its terminal outcome.
That is the same shape as the blocker this branch opened with, where the cohort
whose outcomes varied had no handoff trace. The failure modes that now dominate
are the ones with the least instrumentation.

    python scripts/report_capture_attrition.py --report evidence/capture_attrition_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from handoff_qualification.records import CATASTROPHIC_RESIDUAL_M  # noqa: E402
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

SEEDS = (4070, 5070, 6070)
ARMS = {
    "nominal, retention absent": "artifacts/traced_nominal/seed{seed}/nominal.npz",
    "nominal, pawls": "artifacts/retention_nominal/seed{seed}/nominal.npz",
    "prescribed relief, pawls": "artifacts/prescription_retained/seed{seed}/nominal.npz",
    "section 120x16, pawls": "artifacts/section_retained/seed{seed}/section_120x16.npz",
    "section 140x26, pawls": "artifacts/section_retained/seed{seed}/section_140x26.npz",
}


def undelivered(path: Path) -> dict | None:
    if not path.exists():
        return None
    archive = np.load(path, allow_pickle=True)
    fields = [str(name) for name in archive["fields"]]
    rows = archive["rows"].astype(float)
    lateral = rows[:, fields.index("lateral_error_m")]
    missing = lateral >= CATASTROPHIC_RESIDUAL_M

    def column(name: str) -> np.ndarray:
        return rows[:, fields.index(name)] if name in fields else np.full(len(rows), np.nan)

    return {
        "episodes": len(rows),
        "never_delivered": int(missing.sum()),
        "environments": sorted(int(v) for v in np.where(missing)[0]),
        "timed_out_in_phase": sorted({int(v) for v in column("timed_out_in_phase")[missing]}),
        "control_steps": sorted({int(v) for v in column("control_steps")[missing]}),
        "timed_out_phase_counts": {
            ("capture", "seat", "extract", "transit", "insert", "done")[int(v)]: int(
                (column("timed_out_in_phase")[missing] == v).sum()
            )
            for v in sorted({int(x) for x in column("timed_out_in_phase")[missing] if 0 <= x < 6})
        },
        "grip_error_m": sorted(round(float(v), 5) for v in column("grip_error_m")[missing]),
        "delivered_grip_error_median_m": float(np.median(column("grip_error_m")[~missing])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    arms: dict[str, dict] = {}
    for label, template in ARMS.items():
        per_seed = {}
        for seed in SEEDS:
            result = undelivered(ROOT / template.format(seed=seed))
            if result is not None:
                per_seed[str(seed)] = result
        if per_seed:
            arms[label] = per_seed

    print(f"{'arm':>28} {'seed':>6} {'undelivered':>12} {'environments':>22} {'steps':>7}")
    for label, per_seed in arms.items():
        for seed, entry in per_seed.items():
            print(
                f"{label:>28} {seed:>6} {entry['never_delivered']:>7}/{entry['episodes']:<4} "
                f"{str(entry['environments']):>22} {str(entry['control_steps']):>7}"
            )

    print()
    print("which phase they time out in:")
    for label, per_seed in arms.items():
        pooled: dict[str, int] = {}
        for entry in per_seed.values():
            for phase, count in entry["timed_out_phase_counts"].items():
                pooled[phase] = pooled.get(phase, 0) + count
        print(f"   {label:>28}: {pooled}")

    # Determinism is only meaningful between arms that share module geometry:
    # changing the cross-section changes the grasp, so different environments
    # fail and that is the configuration doing its job, not noise.
    same_geometry = [
        label for label in arms if "section" not in label
    ]
    per_seed_sets: dict[str, list[set[int]]] = {}
    for label in same_geometry:
        for seed, entry in arms[label].items():
            per_seed_sets.setdefault(seed, []).append(set(entry["environments"]))
    identical = {
        seed: all(s == sets[0] for s in sets) for seed, sets in per_seed_sets.items()
    }
    print()
    print("across the arms that share the nominal module geometry, the same")
    print(f"environments fail every time, per seed: {identical}")
    print("   (arms with a changed cross-section are excluded: a different module")
    print("    changes the grasp, so different environments failing is the")
    print("    configuration working rather than a lack of determinism)")

    document = {
        "title": "What remains once the bay holds the module, and what records it",
        "evidence_type": "second_reading_of_existing_episodes",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "With the pawls fitted, precision given delivery is 1.000 everywhere "
            "measured. What is the remaining failure budget, and what evidence exists "
            "about it?"
        ),
        "source_revision": git_source_revision(ROOT),
        "seeds": list(SEEDS),
        "arms": arms,
        "same_environments_across_nominal_geometry_arms_per_seed": identical,
        "capture_phase_has_a_sampled_trace": False,
        "trace_blocks_that_exist": ["handoff", "transit", "insert", "settle"],
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "This is a second reading of existing episodes, not a new measurement.",
            "An episode is called undelivered when its terminal lateral error exceeds "
            "the 50 mm boundary fixed before any of these cohorts were read. On this "
            "workcell that distance is the bay separation, so such an episode did not "
            "mis-seat: it never left the source bay.",
            "The grip error quoted for an undelivered episode is a judgement-time "
            "value, describing where the tool ended up. It does not say why the "
            "approach failed, and nothing here should be read as a cause.",
            "Undelivered episodes time out in capture or in extract, and which "
            "dominates depends on the module cross-section. Reading them all as "
            "capture failures would be wrong.",
            "The capture phase writes no sampled trace, so an episode that fails there "
            "leaves one phase-transition row and its terminal outcome. Diagnosing these "
            "needs instrumentation that does not exist yet, not more analysis of what "
            "does.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
