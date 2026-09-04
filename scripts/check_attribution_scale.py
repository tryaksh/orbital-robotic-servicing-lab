#!/usr/bin/env python3
"""Is the channel interaction an artifact of the scale it was measured on?

The paper's perception contribution is an interaction: noising the pose channels
alone costs 8.33 points, the velocity channel alone 10.21, and both together
41.15, so the two cost 22.61 points more together than apart. That is stated as a
difference of proportions, and a difference of proportions is not scale-free. A
reviewer is entitled to ask whether the interaction is real or whether it is what
any two large effects look like when they are added on a bounded scale and run
out of room near zero.

This answers it by recomputing the same contrast on the log-odds scale, where
"no interaction" means the odds multiply. Both readings come from the same four
counts in ``extract_channel_attribution_v1.json``; nothing is re-run.

If the odds ratio's interval excludes 1, the interaction is not an artifact of
measuring probabilities, and the claim can be made in the paper without a
caveat about scale. If it includes 1, the honest statement is that the two
channels are multiplicative and the paper should say *that* instead, which is a
weaker but still publishable result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "evidence" / "extract_channel_attribution_v1.json"

#: Arm labels the report uses, in factorial order.
EXACT, POSE, VELOCITY, BOTH = "exact state", "pose only", "velocity only", "both"


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def interaction(counts: dict[str, tuple[int, int]], resamples: int, seed: int) -> dict:
    """Both readings of the same contrast, with parametric bootstrap intervals."""

    rng = np.random.default_rng(seed)
    draws = {name: rng.binomial(n, s / n, resamples) / n for name, (s, n) in counts.items()}
    probability = draws[EXACT] - draws[POSE] - draws[VELOCITY] + draws[BOTH]
    log_odds = _logit(draws[EXACT]) - _logit(draws[POSE]) - _logit(draws[VELOCITY]) + _logit(draws[BOTH])

    def interval(sample: np.ndarray) -> list[float]:
        low, high = np.percentile(sample, [2.5, 97.5])
        return [round(float(low), 4), round(float(high), 4)]

    odds_low, odds_high = np.exp(np.percentile(log_odds, [2.5, 97.5]))
    return {
        "rates": {name: round(s / n, 6) for name, (s, n) in counts.items()},
        "probability_scale": {
            "interaction_points": round(float(probability.mean()) * 100.0, 2),
            "bootstrap_95_points": [round(v * 100.0, 2) for v in interval(probability)],
        },
        "log_odds_scale": {
            "interaction": round(float(log_odds.mean()), 4),
            "bootstrap_95": interval(log_odds),
            "odds_ratio": round(float(np.exp(log_odds.mean())), 4),
            "odds_ratio_95": [round(float(odds_low), 4), round(float(odds_high), 4)],
        },
        "survives_the_scale_change": bool(odds_high < 1.0 or odds_low > 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attribution", type=Path, default=DEFAULT)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=4070)
    args = parser.parse_args()

    report = json.loads(args.attribution.read_text(encoding="utf-8"))
    counts = {arm["arm"]: (int(arm["successes"]), int(arm["episodes"])) for arm in report["arms"]}
    missing = {EXACT, POSE, VELOCITY, BOTH} - set(counts)
    if missing:
        raise SystemExit(f"{args.attribution} is missing arms: {sorted(missing)}")

    result = interaction(counts, args.resamples, args.seed)
    for name, rate in result["rates"].items():
        print(f"  {name:14s} {rate:.6f}")
    prob, odds = result["probability_scale"], result["log_odds_scale"]
    print()
    print(f"  probability scale  {prob['interaction_points']:+.2f} points  95% {prob['bootstrap_95_points']}")
    print(f"  log-odds scale     {odds['interaction']:+.4f}         95% {odds['bootstrap_95']}")
    print(f"  odds ratio         {odds['odds_ratio']:.3f}           95% {odds['odds_ratio_95']}")
    print()
    if result["survives_the_scale_change"]:
        print("  The interval excludes an odds ratio of 1, so the interaction is not an")
        print("  artifact of measuring on the probability scale.")
    else:
        print("  The odds ratio's interval includes 1. The channels are multiplicative and")
        print("  the paper must say that rather than claiming an interaction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
