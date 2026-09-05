#!/usr/bin/env python3
"""The rack prescription, read as the 2x2 it needed to be all along.

`evidence/rack_prescription_paired_n192.json` is the only published *refutation*
in this project's boundary set: rebuilding the destination channel at the design
library's prescribed clearance took the chain from 110/192 to 64/192, and the
conclusion drawn was that the library's seated-rest upper bound does not govern.

Both arms of that comparison ran with the destination bay's retention pawls
absent. The failure the prescription was meant to fix -- episodes that arrive,
seat, and miss the terminal gate -- is now known to be the module coasting after
release with nothing holding it, and fitting the pawls removes all 77 of them.
So the refutation was measured against a failure mode the prescription could not
have fixed and the fixture does.

Crossing the two factors separates what the prescription actually does from what
the missing fixture was doing:

    channel relief x retention fitted, 192 paired episodes a cell

The distribution shape matters more than the rate here, and is reported beside
it. At the shipped relief no delivered module comes to rest beyond 6 mm. At the
prescribed clearance a tail appears that did not exist before -- episodes 10 to
50 mm off centre -- and a pawl that engages after a bad seating locks the bad
seating in. A rate alone cannot tell "more near-misses" from "a new gross
failure mode", and those have different consequences for the interface.

    python scripts/report_prescription_factorial.py --report evidence/prescription_factorial_v1.json
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

from handoff_qualification import load_cohort  # noqa: E402
from handoff_qualification.records import (  # noqa: E402
    CATASTROPHIC_RESIDUAL_M,
    DEFAULT_LATERAL_CRITERION_M,
)
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

SEEDS = (4070, 5070, 6070)

#: The four cells, as (channel relief, retention) -> path template.
CELLS: dict[tuple[str, str], str] = {
    ("4.61 mm", "absent"): "artifacts/traced_nominal/seed{seed}/nominal.npz",
    ("0.00 mm", "absent"): "artifacts/relief0_seed{seed}/nominal.npz",
    ("4.61 mm", "pawls"): "artifacts/retention_nominal/seed{seed}/nominal.npz",
    ("0.00 mm", "pawls"): "artifacts/prescription_retained/seed{seed}/nominal.npz",
}

#: Residual bands, in metres. The boundary at 6 mm is where the shipped-relief
#: distribution ends, so anything past it is a mode that configuration does not
#: produce at all.
BANDS = (
    (0.0, DEFAULT_LATERAL_CRITERION_M, "passes"),
    (DEFAULT_LATERAL_CRITERION_M, 0.006, "near miss 2.5-6 mm"),
    (0.006, 0.010, "off 6-10 mm"),
    (0.010, CATASTROPHIC_RESIDUAL_M, "gross 10-50 mm"),
)


def load_cell(template: str, seeds: tuple[int, ...]) -> list | None:
    records = []
    for seed in seeds:
        path = ROOT / template.format(seed=seed)
        if not path.exists():
            return None
        records.extend(load_cohort(path))
    return records


def describe(records: list, criterion_m: float) -> dict:
    residual = np.array([r.residual_lateral_m for r in records])
    delivered = residual < CATASTROPHIC_RESIDUAL_M
    arrived = residual[delivered]
    bands = {}
    for low, high, name in BANDS:
        bands[name] = int(((arrived >= low) & (arrived < high)).sum())
    return {
        "episodes": len(records),
        "passes": int((residual < criterion_m).sum()),
        "delivered": int(delivered.sum()),
        "precision_given_delivery": float((arrived < criterion_m).mean()),
        "residual_median_m": float(np.median(arrived)),
        "residual_max_m": float(arrived.max()),
        "bands": bands,
        "never_delivered": int((~delivered).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--criterion_m", type=float, default=DEFAULT_LATERAL_CRITERION_M)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cells: dict[str, dict] = {}
    missing: list[str] = []
    for (relief, retention), template in CELLS.items():
        records = load_cell(template, SEEDS)
        key = f"relief {relief}, retention {retention}"
        if records is None:
            missing.append(key)
            continue
        cells[key] = {"relief": relief, "retention": retention, **describe(records, args.criterion_m)}

    header = f"{'channel relief':>15} {'retention':>10} {'pass':>10} {'deliv':>8} {'prec|del':>9} {'med mm':>8} {'max mm':>8}"
    print(header)
    for cell in cells.values():
        print(
            f"{cell['relief']:>15} {cell['retention']:>10} "
            f"{cell['passes']:>4}/{cell['episodes']:<5} {cell['delivered']:>8} "
            f"{cell['precision_given_delivery']:>9.4f} "
            f"{cell['residual_median_m'] * 1000:>8.3f} {cell['residual_max_m'] * 1000:>8.3f}"
        )
    for key in missing:
        print(f"{key:>36}   (not yet run)")

    print()
    print("Where the delivered modules come to rest:")
    band_names = [name for _, _, name in BANDS]
    print(f"{'cell':>36} " + " ".join(f"{n:>19}" for n in band_names))
    for key, cell in cells.items():
        row = " ".join(f"{cell['bands'][n]:>19}" for n in band_names)
        print(f"{key:>36} {row}")

    print()
    print(
        "A tail past 6 mm is a mode the shipped relief does not produce at all. "
        "Pawls engage after seating, so they preserve a bad seating as readily as "
        "a good one and cannot remove that tail."
    )

    document = {
        "title": "The rack prescription crossed with the retention fixture",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "The published refutation of the design library's prescribed clearance was "
            "measured with the destination bay's retention pawls absent, against a "
            "failure mode the fixture is now known to remove. What does the "
            "prescription do once the bay holds the module?"
        ),
        "criterion_m": args.criterion_m,
        "source_revision": git_source_revision(ROOT),
        "seeds": list(SEEDS),
        "cells": cells,
        "cells_not_yet_run": missing,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "One point of the sweep -- nominal -- at three seeds. The other boundary "
            "points are still retention-absent and are not covered by this.",
            "The historical relief-0 cohort ran at STEPS=6000 where the others use the "
            "default; the per-phase budgets are identical across all four cells and "
            "bind first, so the step cap is not the operative difference.",
            "The 50 mm boundary separating a delivered module from one that never left "
            "the source bay was fixed before these cells were read and is not tuned to "
            "them. Residuals between 10 and 50 mm are counted as delivered and are "
            "reported as their own band rather than folded into either end.",
            "Pawls engage after the unchanged insertion predicate fires. They preserve "
            "whatever seating they are given and are not a correction mechanism.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
