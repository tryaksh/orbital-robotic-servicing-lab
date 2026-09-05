#!/usr/bin/env python3
"""Does the criterion curve hold up, within a configuration and across three?

Two questions, in the order they have to be asked.

**Within.** A criterion curve read from n episodes predicts the pass rate at a
criterion those episodes were not scored at. Is that prediction any good? Tested
by holding out: estimate the curve from a subsample, compare its value at a
criterion against the full cohort's measured rate there. If this fails, nothing
else matters -- the curve would be an artefact of the sample rather than a
property of the configuration.

**Across.** Three configurations of the same chain differ only in the module
cross-section. If their residual distributions are ordered the way the geometry
is, a curve measured on two says something about the third. If they are not, the
curve is a per-configuration measurement and must be sold as one -- which is
still useful, and much less than a method.

The second question is the one the charter's H3 turns on, and it is set up to
fail: three configurations is three points, and two of them being ordered
correctly by chance is a coin toss.

    python scripts/report_residual_transfer.py --report evidence/residual_transfer_v1.json
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

#: Criteria to test the held-out prediction at, in metres.
HELD_OUT_CRITERIA = (0.002, 0.003, 0.004)


def within_configuration(
    residuals: np.ndarray,
    criteria: tuple[float, ...],
    sizes: tuple[int, ...],
    repeats: int,
    seed: int,
) -> list[dict]:
    """Estimate the curve from a subsample; score it against the full cohort."""

    rng = np.random.default_rng(seed)
    rows = []
    for criterion in criteria:
        truth = float((residuals < criterion).mean())
        for size in sizes:
            if size >= len(residuals):
                continue
            errors = np.empty(repeats)
            covered = 0
            for index in range(repeats):
                draw = rng.choice(residuals, size=size, replace=False)
                estimate = float((draw < criterion).mean())
                errors[index] = estimate - truth
                # A binomial interval on the subsample, checked for coverage.
                spread = 1.959963985 * np.sqrt(max(estimate * (1 - estimate), 1e-9) / size)
                if estimate - spread <= truth <= estimate + spread:
                    covered += 1
            rows.append(
                {
                    "criterion_m": criterion,
                    "full_cohort_rate": truth,
                    "subsample_size": size,
                    "mean_absolute_error": float(np.abs(errors).mean()),
                    "bias": float(errors.mean()),
                    "coverage_95": covered / repeats,
                    "repeats": repeats,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sweep_dir", type=Path, default=ROOT / "artifacts/robustness192_section"
    )
    parser.add_argument("--points", nargs="+", default=["nominal", "section_120x16", "section_140x26"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cohorts: dict[str, np.ndarray] = {}
    for point in args.points:
        path = args.sweep_dir / f"{point}.npz"
        if not path.exists():
            print(f"missing {path}, skipping")
            continue
        records = load_cohort(path)
        cohorts[point] = np.array([r.residual_lateral_m for r in records])

    if not cohorts:
        print("no cohorts found")
        return 1

    print("Residual distribution by configuration (arrived episodes only):")
    print(f"{'configuration':>18} {'n':>5} {'arrived':>8} {'median mm':>10} {'p95 mm':>8} {'rate@2.5mm':>11}")
    descriptions = {}
    for name, residuals in cohorts.items():
        arrived = residuals[residuals < CATASTROPHIC_RESIDUAL_M]
        rate = float((residuals < DEFAULT_LATERAL_CRITERION_M).mean())
        descriptions[name] = {
            "episodes": len(residuals),
            "arrived": len(arrived),
            "median_m": float(np.median(arrived)),
            "p95_m": float(np.percentile(arrived, 95)),
            "rate_at_default_criterion": rate,
        }
        print(
            f"{name:>18} {len(residuals):>5} {len(arrived):>8} "
            f"{np.median(arrived) * 1000:>10.3f} {np.percentile(arrived, 95) * 1000:>8.3f} {rate:>11.4f}"
        )

    print()
    print("WITHIN a configuration: curve from a subsample against the full cohort")
    within = {}
    for name, residuals in cohorts.items():
        rows = within_configuration(
            residuals, HELD_OUT_CRITERIA, tuple(args.sizes), args.repeats, args.seed
        )
        within[name] = rows
    print(f"{'config':>18} {'crit mm':>8} {'n':>5} {'truth':>7} {'MAE':>7} {'bias':>8} {'cover':>7}")
    for name, rows in within.items():
        for row in rows:
            print(
                f"{name:>18} {row['criterion_m'] * 1000:>8.1f} {row['subsample_size']:>5} "
                f"{row['full_cohort_rate']:>7.3f} {row['mean_absolute_error']:>7.3f} "
                f"{row['bias']:>+8.4f} {row['coverage_95']:>7.3f}"
            )

    print()
    print("ACROSS configurations: is the ordering the geometry predicts the ordering observed?")
    ordered_by_median = sorted(descriptions, key=lambda k: descriptions[k]["median_m"])
    ordered_by_rate = sorted(descriptions, key=lambda k: -descriptions[k]["rate_at_default_criterion"])
    print(f"  by residual median (best first): {ordered_by_median}")
    print(f"  by pass rate       (best first): {ordered_by_rate}")
    consistent = ordered_by_median == ordered_by_rate
    print(f"  the two orderings agree: {consistent}")
    print(
        "  Three configurations is three points. Agreement here is necessary and "
        "nowhere near sufficient."
    )

    document = {
        "title": "Criterion-curve reliability within a configuration, and ordering across three",
        "evidence_type": "resampling_from_existing_episodes",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Is a pass rate read off the residual distribution at an unscored criterion "
            "reliable within a configuration, and does it order three configurations "
            "the way their geometry does?"
        ),
        "default_criterion_m": DEFAULT_LATERAL_CRITERION_M,
        "source_revision": git_source_revision(ROOT),
        "configurations": descriptions,
        "within_configuration": within,
        "across_configurations": {
            "ordered_by_residual_median": ordered_by_median,
            "ordered_by_pass_rate": ordered_by_rate,
            "orderings_agree": consistent,
        },
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "This is a second reading of existing episodes, not a new measurement.",
            "Within-configuration subsamples are drawn from the same 192 episodes they "
            "are scored against, so the reported error is the error of reading a curve "
            "from a subsample, not the error of a fresh campaign.",
            "Three configurations is three points. Consistent ordering is necessary for "
            "any transfer claim and comes nowhere near establishing one.",
            "The ordering compares two summaries of the same residuals, so agreement is "
            "close to arithmetic when no episode is catastrophic; it is reported because "
            "disagreement would be informative, not because agreement is a result.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
