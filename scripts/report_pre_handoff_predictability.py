#!/usr/bin/env python3
"""Does the state handed to the seating step predict what the seating step leaves?

This is the test the charter's H2 turns on, and the one that decides GO against
NARROW. It is written to fail cleanly.

Only **pre-handoff** columns are admissible: the handoff row recorded at
``to_phase == INSERT``, which is state that existed before the contact interval
began. The terminal pose columns in the episode archive are written at the
moment of judgement -- `_freeze` in `scripts/run_workflow_demo.py` -- and
`lateral_error_m` there *defines* the failure, so using it to predict that
failure would be reading the success predicate back. It is the target here and
never an input.

Three things are reported, in increasing strength of evidence:

1. **Univariate rank correlation** of every varying pre-handoff feature with the
   terminal residual, calibrated against the null. Reporting the largest of k
   correlations without saying what the largest of k *noise* correlations looks
   like at this sample size is how a table of small numbers becomes a finding.
2. **Held-out predictive score.** A ridge regression on the residual and a
   logistic model on the thresholded outcome, scored by grouped cross-validation
   against the only baselines that matter -- predicting the cohort mean, and
   predicting the cohort base rate.
3. **The decision**, against the margin the charter declared in advance: a
   predictor must beat the base rate by at least 0.10 absolute in Brier score.

Folds are grouped by seed. Episodes within one seed share a checkpoint, a scene
build and a reset stream, so splitting them at random would leak.

    python scripts/report_pre_handoff_predictability.py \\
        --cohort artifacts/traced_nominal/seed4070/nominal.npz \\
        --cohort artifacts/traced_nominal/seed5070/nominal.npz \\
        --cohort artifacts/traced_nominal/seed6070/nominal.npz \\
        --report evidence/pre_handoff_predictability_v1.json
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
from handoff_qualification.records import (  # noqa: E402
    CATASTROPHIC_RESIDUAL_M,
    DEFAULT_LATERAL_CRITERION_M,
)
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

#: The charter's declared margin: a predictor must beat the base rate by this
#: much, absolute, in held-out Brier score, or H2 is falsified.
REQUIRED_BRIER_GAIN = 0.10


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in np.atleast_1d(z)]))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator else 0.0


def null_max_correlation(features: int, samples: int, draws: int, seed: int) -> dict:
    """What the largest of `features` correlations looks like when none is real."""

    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    for index in range(draws):
        largest = 0.0
        for _ in range(features):
            largest = max(
                largest, abs(spearman(rng.normal(size=samples), rng.normal(size=samples)))
            )
        maxima[index] = largest
    return {
        "draws": draws,
        "median": float(np.median(maxima)),
        "percentile_95": float(np.percentile(maxima, 95)),
    }


def ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    centred = np.column_stack([np.ones(len(x)), x])
    identity = np.eye(centred.shape[1])
    identity[0, 0] = 0.0  # never penalise the intercept
    return np.linalg.solve(centred.T @ centred + penalty * identity, centred.T @ y)


def apply_ridge(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def grouped_scores(
    x: np.ndarray,
    residual: np.ndarray,
    passed: np.ndarray,
    groups: np.ndarray,
    criterion_m: float,
    penalty: float,
) -> dict:
    """Leave-one-group-out. Groups are seeds, which share a checkpoint."""

    unique = np.unique(groups)
    brier_model = []
    brier_base = []
    squared_model = []
    squared_base = []
    for held in unique:
        train = groups != held
        test = ~train
        if train.sum() < 5 or test.sum() == 0:
            continue
        centre = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale == 0] = 1.0
        fitted = ridge((x[train] - centre) / scale, residual[train], penalty)
        predicted_residual = apply_ridge(fitted, (x[test] - centre) / scale)
        # The thresholded prediction, as a probability via the training spread.
        spread = float(np.std(residual[train] - apply_ridge(fitted, (x[train] - centre) / scale)))
        spread = max(spread, 1e-9)
        z = (criterion_m - predicted_residual) / spread
        probability = np.clip(_normal_cdf(z), 1e-6, 1 - 1e-6)

        base_rate = float(passed[train].mean())
        base_residual = float(residual[train].mean())

        brier_model.append(float(np.mean((probability - passed[test]) ** 2)))
        brier_base.append(float(np.mean((base_rate - passed[test]) ** 2)))
        squared_model.append(float(np.mean((predicted_residual - residual[test]) ** 2)))
        squared_base.append(float(np.mean((base_residual - residual[test]) ** 2)))

    return {
        "folds": len(brier_model),
        "brier_model": float(np.mean(brier_model)) if brier_model else None,
        "brier_base_rate": float(np.mean(brier_base)) if brier_base else None,
        "brier_gain": (
            float(np.mean(brier_base) - np.mean(brier_model)) if brier_model else None
        ),
        "residual_mse_model": float(np.mean(squared_model)) if squared_model else None,
        "residual_mse_mean": float(np.mean(squared_base)) if squared_base else None,
    }


def incoming_spread(x: np.ndarray, names: list[str], residual: np.ndarray) -> dict:
    """How wide is the distribution the handoff actually delivers?

    A null predictive result has two possible causes and they call for opposite
    next steps. Either the contact interval genuinely destroys the information
    the incoming state carried, or the incoming state never varied enough to
    carry any -- in which case nothing has been tested and the experiment that
    would test it is to widen the handoff distribution on purpose.

    Reporting the spread of each incoming length beside the spread of the
    residual is the cheapest way to tell those apart.
    """

    lengths = [n for n in names if n.endswith("_m")]
    spreads = {}
    for name in lengths:
        column = x[:, names.index(name)]
        spreads[name] = float(np.percentile(column, 95) - np.percentile(column, 5))
    return {
        "incoming_p5_p95_spread_m": spreads,
        "residual_p5_p95_spread_m": float(
            np.percentile(residual, 95) - np.percentile(residual, 5)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    parser.add_argument("--criterion_m", type=float, default=DEFAULT_LATERAL_CRITERION_M)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--null_draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    records = []
    groups = []
    for index, path in enumerate(args.cohort):
        loaded = load_cohort(path)
        for record in loaded:
            if record.incoming:
                records.append(record)
                groups.append(index)

    if not records:
        print("no episode carries a pre-handoff row; was the cohort run with --handoff_trace?")
        return 1

    names = sorted(records[0].incoming)
    x_all = np.array([[r.incoming[n] for n in names] for r in records])
    y_all = np.array([r.residual_lateral_m for r in records])
    group_all = np.array(groups)

    arrived = y_all < CATASTROPHIC_RESIDUAL_M
    x = x_all[arrived]
    y = y_all[arrived]
    group = group_all[arrived]
    passed = (y < args.criterion_m).astype(float)

    varying = [i for i in range(x.shape[1]) if np.ptp(x[:, i]) > 0]
    x = x[:, varying]
    varying_names = [names[i] for i in varying]

    print(f"{len(records)} episodes carry a pre-handoff row; {int(arrived.sum())} arrived")
    print(f"{len(varying_names)} of {len(names)} pre-handoff features vary")
    print(f"base rate at {args.criterion_m * 1000:.2f} mm: {passed.mean():.4f}")
    print()

    correlations = sorted(
        ((abs(spearman(x[:, i], y)), spearman(x[:, i], y), n) for i, n in enumerate(varying_names)),
        reverse=True,
    )
    print("pre-handoff feature against terminal residual (arrived episodes):")
    for _magnitude, value, name in correlations[:10]:
        print(f"   {name:>42}  rho = {value:+.3f}")

    null = null_max_correlation(len(varying_names), len(y), args.null_draws, args.seed)
    observed_max = correlations[0][0] if correlations else 0.0
    print()
    print(
        f"under the null, the largest of {len(varying_names)} correlations at n={len(y)}: "
        f"median {null['median']:.3f}, 95th {null['percentile_95']:.3f}"
    )
    print(f"observed largest: {observed_max:.3f}")
    beats_noise = observed_max > null["percentile_95"]
    print(
        "  -> "
        + (
            "exceeds the 95th percentile of noise"
            if beats_noise
            else "does NOT exceed what noise produces; no univariate signal"
        )
    )

    spread = incoming_spread(x, varying_names, y)
    print()
    print("spread the handoff delivers, against the spread the seating step leaves (mm):")
    for name, value in sorted(spread["incoming_p5_p95_spread_m"].items(), key=lambda kv: -kv[1]):
        print(f"   {name:>42}  {value * 1000:>8.3f}")
    print(f"   {'TERMINAL lateral_error_m':>42}  {spread['residual_p5_p95_spread_m'] * 1000:>8.3f}")

    scores = grouped_scores(x, y, passed, group, args.criterion_m, args.penalty)
    print()
    print(f"leave-one-seed-out, {scores['folds']} folds:")
    if scores["brier_model"] is not None:
        print(f"  Brier, model      {scores['brier_model']:.4f}")
        print(f"  Brier, base rate  {scores['brier_base_rate']:.4f}")
        print(f"  gain              {scores['brier_gain']:+.4f}  (need +{REQUIRED_BRIER_GAIN:.2f})")
        print(f"  residual MSE, model {scores['residual_mse_model']:.3e}")
        print(f"  residual MSE, mean  {scores['residual_mse_mean']:.3e}")
    else:
        print("  not enough groups to score; pass more than one cohort")

    gain = scores["brier_gain"]
    verdict = (
        "insufficient_groups"
        if gain is None
        else ("supports_h2" if gain >= REQUIRED_BRIER_GAIN else "falsifies_h2")
    )
    print()
    print(f"VERDICT: {verdict}")

    document = {
        "title": "Whether pre-handoff state predicts the residual the seating step leaves",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "The pivot brief asks which incoming errors get corrected and which stay "
            "dangerous. This asks the measurable form: does the state handed to the "
            "seating step carry any information about the residual it leaves?"
        ),
        "cohorts": [p.as_posix() for p in args.cohort],
        "criterion_m": args.criterion_m,
        "source_revision": git_source_revision(ROOT),
        "episodes_with_pre_handoff_row": len(records),
        "arrived": int(arrived.sum()),
        "base_rate": float(passed.mean()),
        "features_varying": varying_names,
        "univariate": {
            "correlations": [
                {"feature": name, "spearman_rho": value} for _, value, name in correlations
            ],
            "observed_max_abs_rho": observed_max,
            "null_max_abs_rho": null,
            "exceeds_noise_95th": bool(beats_noise),
        },
        "delivered_distribution": spread,
        "held_out": scores,
        "required_brier_gain": REQUIRED_BRIER_GAIN,
        "verdict": verdict,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "Only pre-handoff columns are used as inputs. The terminal pose is the "
            "target and never an input, because it defines the outcome.",
            "Folds are grouped by seed; episodes within a seed share a checkpoint and "
            "a reset stream, so a random split would leak.",
            "One fixed checkpoint set. This says nothing about whether a differently "
            "trained controller would leave a predictable residual.",
            "A null univariate result does not prove no predictor exists. It bounds "
            "how strong a simple one can be at this sample size, which is the "
            "question the day-14 decision needs answered.",
            "A null result here is not by itself evidence that the fixture destroys "
            "incoming error. Read it beside delivered_distribution: if the handoff "
            "delivers a distribution far narrower than the residual it leaves, then "
            "the incoming variation was never wide enough to be predictive and the "
            "experiment that would settle it -- widening the handoff distribution on "
            "purpose -- has not been run.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
