#!/usr/bin/env python3
"""Compare two arms that were run on the same cohort, as the paired data they are.

Every A/B in this project is built the same way: the same seeds, the same
checkpoints, the same environments, one flag changed. That is a paired design,
and a paired design is not read with two independent Wilson intervals. Doing so
throws away the pairing and can leave a clean result looking like nothing --
rack retention flips five episodes from failure to success and none the other
way, which two overlapping Wilson intervals report as inconclusive.

The right reading is McNemar's exact test on the discordant pairs: of the
episodes whose outcome changed, how lopsided is the change? Five-for-nothing is
one-sided p = 0.031. It is the same twenty-four episodes either way, so the
question "did the flag help" does not need the two arms to be separated as if
they were different populations.

**This assumes the pairing is real** -- that episode *i* of one arm and episode
*i* of the other started from the same state. Here that follows from identical
seeds, identical checkpoints and deterministic resets, which is what every
certification in this repository means by "identical fixed cohorts". It is an
assumption the tool cannot check, and if two arms were run at different seeds
the paired reading is wrong and the unpaired one is right. Both are printed.

Unpaired Wilson intervals stay in the output because they answer a different and
also useful question: what is each arm's rate, on its own. Neither replaces the
other.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zero_g_blade_swap.evaluation import wilson_interval  # noqa: E402


def _successes(paths: list[Path]) -> np.ndarray:
    out = []
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        fields = [str(f) for f in archive["fields"]]
        rows = archive["rows"].astype(float)
        out.append(rows[:, fields.index("success")] > 0.5)
    return np.concatenate(out)


def mcnemar_exact(gained: int, lost: int) -> dict:
    """Exact binomial test on the discordant pairs."""

    n = gained + lost
    if n == 0:
        return {"discordant": 0, "one_sided_p": None, "two_sided_p": None}
    tail = sum(math.comb(n, k) for k in range(0, min(gained, lost) + 1)) / 2**n
    # Stored at full precision and rounded only when printed. Thirty-six
    # discordant pairs all in one direction give 2**-36; rounded to five decimal
    # places that prints 0.0, which claims the result is impossible rather than
    # very unlikely. Rounding belongs in the presentation, not in the data.
    return {
        "discordant": n,
        "one_sided_p": tail,
        "two_sided_p": min(1.0, 2 * tail),
    }


def compare(baseline: np.ndarray, treatment: np.ndarray) -> dict:
    if len(baseline) != len(treatment):
        raise SystemExit(
            f"arms have different episode counts ({len(baseline)} and {len(treatment)}); "
            "they cannot be paired"
        )
    gained = int(((~baseline) & treatment).sum())
    lost = int((baseline & (~treatment)).sum())
    b_low, b_high = wilson_interval(int(baseline.sum()), len(baseline))
    t_low, t_high = wilson_interval(int(treatment.sum()), len(treatment))
    return {
        "episodes": int(len(baseline)),
        "baseline": {
            "successes": int(baseline.sum()),
            "rate": round(float(baseline.mean()), 6),
            "wilson_95": [round(b_low, 4), round(b_high, 4)],
        },
        "treatment": {
            "successes": int(treatment.sum()),
            "rate": round(float(treatment.mean()), 6),
            "wilson_95": [round(t_low, 4), round(t_high, 4)],
        },
        "paired": {
            "gained": gained,
            "lost": lost,
            **mcnemar_exact(gained, lost),
        },
        "wilson_intervals_overlap": bool(t_low <= b_high and b_low <= t_high),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--treatment", type=Path, nargs="+", required=True)
    parser.add_argument("--label", default="treatment against baseline")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    result = compare(_successes(args.baseline), _successes(args.treatment))
    result["label"] = args.label

    base, treat, paired = result["baseline"], result["treatment"], result["paired"]
    print(f"{args.label}, {result['episodes']} paired episodes")
    print(f"  baseline  {base['successes']:3d}  {base['rate']:.4f}  Wilson {base['wilson_95']}")
    print(f"  treatment {treat['successes']:3d}  {treat['rate']:.4f}  Wilson {treat['wilson_95']}")
    print(f"  discordant: gained {paired['gained']}, lost {paired['lost']}")
    if paired["one_sided_p"] is not None:
        print(
            f"  McNemar exact: one-sided p = {paired['one_sided_p']:.3g}, "
            f"two-sided p = {paired['two_sided_p']:.3g}"
        )
    if result["wilson_intervals_overlap"] and paired["one_sided_p"] is not None and paired["one_sided_p"] < 0.05:
        print("  The unpaired intervals overlap and the paired test does not. The pairing is")
        print("  carrying the result, so report it as paired and say the cohorts are fixed.")

    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
