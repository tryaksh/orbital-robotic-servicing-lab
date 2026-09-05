#!/usr/bin/env python3
"""What tolerance does this controller and fixture actually hold, and to what?

The engineer-facing entry point. It takes cohorts of finished episodes and a
criterion, and answers three questions a counted success rate cannot:

1. **What is the pass rate at a criterion this cohort was not run at?** Read
   off the residual distribution the episodes already contain.
2. **What criterion would this configuration meet at a required rate?** The
   inverse of the same curve. This is the number a tolerance negotiation needs
   and the one no rate can supply.
3. **How far is a failing configuration from qualifying?** A cohort that scores
   0 of 16 has a success rate whose 95% interval runs from 0 to 0.21, which is
   compatible with "nearly works" and with "hopeless" alike. Its residuals
   separate those two immediately.

Question 3 is where this earns its place. `artifacts/campaign/secondtask` ran
the same frozen controller and the same bay geometry on the `install` workflow
-- the module arrives by a different path -- and scored 0 of 16. The rate says
only that it failed. The residuals say every episode arrived and seated, between
4.05 mm and 10.41 mm, against a 2.5 mm criterion: a factor of 2.6 on the median,
not a broken workflow.

What this does **not** do is predict an outcome before the episode runs. The
residual is measured at the end of the episode, so everything here is a
description of a configuration that has been exercised, not a gate. Building
the predictive version needs pre-handoff state, which needs `--handoff_trace`,
which the cohorts whose outcomes vary were not run with. See `docs/CHARTER.md`.

    python scripts/qualify_handoff.py --cohort artifacts/robustness192_section/nominal.npz
    python scripts/qualify_handoff.py --cohort <a.npz> --cohort <b.npz> --required_rate 0.95
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
from handoff_qualification.residual import empirical_pass_rate, wilson  # noqa: E402
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

#: Criteria to report the curve at, in metres.
CURVE_M = (0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075, 0.010, 0.015)


def criterion_for_rate(residuals: np.ndarray, required: float) -> float | None:
    """The smallest criterion at which the cohort would reach ``required``.

    Returns None when even admitting every arrived episode falls short, which
    happens when the catastrophic share alone exceeds the shortfall. That is a
    refusal, not a large number: no tolerance qualifies a module that is not in
    the bay.
    """

    arrived = np.sort(residuals[residuals < CATASTROPHIC_RESIDUAL_M])
    if len(arrived) == 0:
        return None
    needed = int(np.ceil(required * len(residuals)))
    if needed > len(arrived):
        return None
    return float(arrived[needed - 1])


def describe(path: Path, criterion_m: float, required: float) -> dict:
    records = load_cohort(path)
    residuals = np.array([r.residual_lateral_m for r in records])
    arrived = residuals < CATASTROPHIC_RESIDUAL_M
    rate, interval = empirical_pass_rate(records, criterion_m)
    needed = criterion_for_rate(residuals, required)
    curve = []
    for value in CURVE_M:
        point_rate, point_interval = empirical_pass_rate(records, value)
        curve.append(
            {
                "criterion_m": value,
                "pass_rate": point_rate,
                "wilson_95": [round(point_interval[0], 4), round(point_interval[1], 4)],
            }
        )
    # The decomposition the counted rate cannot make. Two configurations can
    # lose the same number of episodes for opposite reasons: one never delivers
    # the module, the other delivers it imprecisely. They want opposite fixes,
    # and a pooled rate reports them identically.
    delivered = int(arrived.sum())
    delivered_and_passed = int((residuals[arrived] < criterion_m).sum())
    precision = delivered_and_passed / delivered if delivered else None
    precision_interval = wilson(delivered_and_passed, delivered) if delivered else None

    return {
        "cohort": path.as_posix(),
        "episodes": len(records),
        "arrived": delivered,
        "never_arrived": int((~arrived).sum()),
        "delivery_rate": delivered / len(records) if len(records) else None,
        "precision_given_delivery": precision,
        "precision_wilson_95": (
            [round(precision_interval[0], 4), round(precision_interval[1], 4)]
            if precision_interval
            else None
        ),
        "criterion_m": criterion_m,
        "pass_rate": rate,
        "wilson_95": [round(interval[0], 4), round(interval[1], 4)],
        "residual_median_m": float(np.median(residuals[arrived])) if arrived.any() else None,
        "residual_p95_m": float(np.percentile(residuals[arrived], 95)) if arrived.any() else None,
        "residual_max_arrived_m": float(residuals[arrived].max()) if arrived.any() else None,
        "criterion_for_required_rate_m": needed,
        "required_rate": required,
        "margin_factor": (needed / criterion_m) if needed else None,
        "curve": curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    parser.add_argument("--criterion_m", type=float, default=DEFAULT_LATERAL_CRITERION_M)
    parser.add_argument("--required_rate", type=float, default=0.95)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    entries = [describe(path, args.criterion_m, args.required_rate) for path in args.cohort]

    for entry in entries:
        print(f"{entry['cohort']}")
        print(
            f"  {entry['episodes']} episodes, {entry['arrived']} arrived, "
            f"{entry['never_arrived']} never arrived"
        )
        print(
            f"  at {entry['criterion_m'] * 1000:.2f} mm: {entry['pass_rate']:.4f} "
            f"[{entry['wilson_95'][0]:.3f}, {entry['wilson_95'][1]:.3f}]"
        )
        if entry["precision_given_delivery"] is not None:
            print(
                f"  delivery {entry['delivery_rate']:.4f}  x  "
                f"precision-given-delivery {entry['precision_given_delivery']:.4f} "
                f"[{entry['precision_wilson_95'][0]:.3f}, {entry['precision_wilson_95'][1]:.3f}]"
            )
        if entry["residual_median_m"] is not None:
            print(
                f"  residual (arrived): median {entry['residual_median_m'] * 1000:.2f} mm, "
                f"p95 {entry['residual_p95_m'] * 1000:.2f} mm, "
                f"max {entry['residual_max_arrived_m'] * 1000:.2f} mm"
            )
        needed = entry["criterion_for_required_rate_m"]
        if needed is None:
            print(
                f"  for {entry['required_rate']:.0%}: no criterion qualifies this cohort -- "
                f"{entry['never_arrived']} episodes never reached the bay"
            )
        else:
            print(
                f"  for {entry['required_rate']:.0%}: criterion would have to be "
                f"{needed * 1000:.2f} mm ({entry['margin_factor']:.2f}x the current one)"
            )
        print()

    if len(entries) > 1:
        print("Between cohorts, at the shipped criterion:")
        print(f"  {'overall':>9} {'delivery':>9} {'precision':>10}  cohort")
        for entry in entries:
            precision = entry["precision_given_delivery"]
            print(
                f"  {entry['pass_rate']:>9.4f} {entry['delivery_rate']:>9.4f} "
                f"{precision if precision is not None else float('nan'):>10.4f}  "
                f"{Path(entry['cohort']).name}"
            )
        print(
            "  A configuration that loses episodes in the delivery column and one that "
            "loses them in the precision column want opposite fixes."
        )

    if args.report:
        document = {
            "title": "Handoff qualification read from the residual distribution",
            "evidence_type": "second_reading_of_existing_episodes",
            "generated_utc": datetime.now(UTC).isoformat(),
            "criterion_m": args.criterion_m,
            "required_rate": args.required_rate,
        "source_revision": git_source_revision(ROOT),
            "cohorts": entries,
            "scope_and_limitations": [
                "Simulation only. No result here was produced on real hardware.",
                "This is a second reading of existing episodes, not a new measurement.",
                "The residual is measured at the end of the episode. Nothing here "
                "predicts an outcome before the episode runs, and none of it is a "
                "runtime gate.",
                "A criterion read off this curve is qualified only over the residual "
                "range the cohort actually sampled; outside it the curve is flat "
                "because there is no data, not because the rate is.",
                "Success in these archives is exactly the terminal lateral error "
                "against INSERTION_LATERAL_TOLERANCE_M. That identity was verified on "
                "the 192-episode nominal cohort, where the threshold reproduces the "
                "recorded label for 192 of 192 episodes. It is a property of this "
                "criterion, not a general fact about assembly.",
            ],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
