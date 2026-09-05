#!/usr/bin/env python3
"""Does reading the residual beat counting successes, at the same episode cost?

This is the falsifying test for the correction-aware framing, and it is set up
so that it can fail. Both arms see **exactly the same episodes**. Nothing is
collected for one that is not collected for the other. The only difference is
what is read off them:

* the **counted** arm compares two configurations by their success rates, which
  is what every comparison in this repository has done so far;
* the **residual** arm compares the same episodes by their terminal lateral
  error, using a rank test that assumes no distribution.

If dichotomising at 2.5 mm costs nothing, the two curves lie on top of each
other and the framing buys nothing. The reason to expect otherwise is not novel
-- cutting a continuous measurement into a binary one is known to throw away
information -- and that is exactly why it belongs here as a baseline check
rather than as a contribution: the question is how much it costs *on this
system*, where the answer decides whether a campaign needs 24 episodes a cell
or 96.

Power is estimated by subsampling the two 192-episode cohorts without
replacement, so the "truth" is the full-cohort difference and every subsample is
a real draw from real episodes rather than a simulation of one.

    python scripts/compare_residual_estimators.py --report evidence/residual_vs_counted_power_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from handoff_qualification import load_cohort  # noqa: E402
from handoff_qualification.records import DEFAULT_LATERAL_CRITERION_M  # noqa: E402
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

ALPHA = 0.05
Z_ALPHA = 1.959963985


def two_proportion_p(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    """Two-sided normal-approximation test on two independent rates."""

    if n_a == 0 or n_b == 0:
        return 1.0
    pooled = (successes_a + successes_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        return 1.0
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if standard_error == 0:
        return 1.0
    z = (successes_a / n_a - successes_b / n_b) / standard_error
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided rank-sum test, normal approximation with tie correction."""

    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 1.0
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    # Average ranks within ties.
    values = combined[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    rank_sum_a = ranks[:n_a].sum()
    u_a = rank_sum_a - n_a * (n_a + 1) / 2
    mean_u = n_a * n_b / 2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = float((counts**3 - counts).sum())
    total = n_a + n_b
    variance = (n_a * n_b / 12) * ((total + 1) - tie_term / (total * (total - 1)))
    if variance <= 0:
        return 1.0
    z = (abs(u_a - mean_u) - 0.5) / math.sqrt(variance)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def power_curve(
    residuals_a: np.ndarray,
    residuals_b: np.ndarray,
    criterion_m: float,
    sizes: list[int],
    repeats: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size > min(len(residuals_a), len(residuals_b)):
            continue
        counted_hits = residual_hits = 0
        for _ in range(repeats):
            draw_a = rng.choice(residuals_a, size=size, replace=False)
            draw_b = rng.choice(residuals_b, size=size, replace=False)
            successes_a = int((draw_a < criterion_m).sum())
            successes_b = int((draw_b < criterion_m).sum())
            if two_proportion_p(successes_a, size, successes_b, size) < ALPHA:
                counted_hits += 1
            if mann_whitney_p(draw_a, draw_b) < ALPHA:
                residual_hits += 1
        rows.append(
            {
                "episodes_per_arm": size,
                "counted_power": counted_hits / repeats,
                "residual_power": residual_hits / repeats,
                "repeats": repeats,
            }
        )
    return rows


def _episodes_for_power(rows: list[dict], key: str, target: float) -> int | None:
    for row in rows:
        if row[key] >= target:
            return row["episodes_per_arm"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/robustness192_section/nominal.npz")
    parser.add_argument(
        "--treatment", type=Path, default=ROOT / "artifacts/robustness192_section/section_120x16.npz"
    )
    parser.add_argument("--criterion_m", type=float, default=DEFAULT_LATERAL_CRITERION_M)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48, 64, 96])
    parser.add_argument("--repeats", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    baseline = load_cohort(args.baseline)
    treatment = load_cohort(args.treatment)
    residuals_a = np.array([r.residual_lateral_m for r in baseline])
    residuals_b = np.array([r.residual_lateral_m for r in treatment])

    rate_a = float((residuals_a < args.criterion_m).mean())
    rate_b = float((residuals_b < args.criterion_m).mean())

    rows = power_curve(
        residuals_a, residuals_b, args.criterion_m, args.sizes, args.repeats, args.seed
    )

    print(f"baseline  {args.baseline.name}: {len(residuals_a)} episodes, rate {rate_a:.4f}")
    print(f"treatment {args.treatment.name}: {len(residuals_b)} episodes, rate {rate_b:.4f}")
    print(f"criterion {args.criterion_m * 1000:.3f} mm, alpha {ALPHA}, {args.repeats} subsamples a point")
    print()
    print(f"{'episodes/arm':>13} {'counted':>9} {'residual':>9} {'ratio':>7}")
    for row in rows:
        counted, residual = row["counted_power"], row["residual_power"]
        ratio = f"{residual / counted:.2f}x" if counted > 0 else "--"
        print(f"{row['episodes_per_arm']:>13} {counted:>9.3f} {residual:>9.3f} {ratio:>7}")

    counted_80 = _episodes_for_power(rows, "counted_power", 0.80)
    residual_80 = _episodes_for_power(rows, "residual_power", 0.80)
    print()
    print(f"episodes an arm for 80% power -- counted: {counted_80}, residual: {residual_80}")

    document = {
        "title": "Reading the residual against counting successes, at matched episode cost",
        "evidence_type": "resampling_from_existing_episodes",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Every comparison in this repository dichotomises a continuous terminal "
            "lateral error at 2.5 mm and compares rates. What does that cost in "
            "episodes, on this system?"
        ),
        "method": (
            "Subsample both 192-episode cohorts without replacement at matched size, "
            "test the same episodes two ways (two-proportion on the dichotomised "
            "outcome, Mann-Whitney on the residual), and count rejections at "
            "alpha = 0.05. Identical episodes reach both arms."
        ),
        "criterion_m": args.criterion_m,
        "source_revision": git_source_revision(ROOT),
        "cohorts": {
            "baseline": {"path": args.baseline.as_posix(), "episodes": len(residuals_a), "rate": rate_a},
            "treatment": {"path": args.treatment.as_posix(), "episodes": len(residuals_b), "rate": rate_b},
        },
        "alpha": ALPHA,
        "power": rows,
        "episodes_for_80_percent_power": {"counted": counted_80, "residual": residual_80},
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "This is a second reading of existing episodes, not a new measurement.",
            "Power is estimated against the full-cohort difference between these two "
            "specific configurations. It is not a general statement about every "
            "comparison this repository makes.",
            "Subsamples from a 192-episode pool are not independent of one another, so "
            "the curve is a fair comparison between the two readings and not an "
            "absolute power calculation for a fresh campaign.",
            "The rank test detects a shift in the residual distribution. A shift that "
            "stays entirely on one side of the criterion changes no success rate and "
            "is not, by itself, an engineering result.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
