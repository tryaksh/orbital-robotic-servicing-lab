#!/usr/bin/env python3
"""Decompose the camera-driven chain into main effects and interactions.

Three changes take the camera-driven chain from 4/24 to 13/16 and no single one
of them moves it at all. "The combination works" is a story; which parts of the
combination carry it, and how much of the gain lives in the interaction rather
than in the terms, is a decomposition -- and a decomposition is what another
group can apply to a system that is not ours.

The design is a full 2x2x2 over

    N  extraction retrained on the estimator's certified error
    K  module velocity from the robot's encoders rather than differenced camera poses
    L  guarded advance admitting on the entry flare's catch rather than the estimator's noise bound

Effects are reported on two scales because a difference of proportions is not
scale-free and the cells here run from about 0.06 to about 0.85, which is most
of the range where that matters. On the log-odds scale "no interaction" means the
odds multiply; on the probability scale it means the points add. A conclusion
that holds on both is a conclusion; one that holds on only one is a statement
about the scale and must be reported as such.

**Twenty-four episodes a cell.** Main effects average four cells against four and
are estimable. The three-way term is a single contrast over eight cells of
twenty-four and its interval will be very wide. Reporting it with that interval
is honest; reporting it as a finding is not.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: cell label -> (N, K, L) as 0/1
CELLS: dict[str, tuple[int, int, int]] = {
    "base_000": (0, 0, 0),
    "noised_extract": (1, 0, 0),
    "kinematic_velocity": (0, 1, 0),
    "guard_00L": (0, 0, 1),
    "bothchannels_NK0": (1, 1, 0),
    "noisedguard_N0L": (1, 0, 1),
    "velguard_0KL": (0, 1, 1),
    "noised_extract_kinematic_leadin": (1, 1, 1),
}

FACTORS = ("N", "K", "L")


def _load_counts(label: str, search: list[Path]) -> tuple[int, int] | None:
    """(successes, episodes) for one cell, from whichever archive holds it."""

    successes = episodes = 0
    found = False
    for directory in search:
        for path in sorted(directory.glob(f"{label}_seed*.npz")):
            if "stale" in path.name:
                continue
            archive = np.load(path, allow_pickle=True)
            fields = [str(f) for f in archive["fields"]]
            rows = archive["rows"].astype(float)
            successes += int((rows[:, fields.index("success")] > 0.5).sum())
            episodes += int(rows.shape[0])
            found = True
        if found:
            break
    return (successes, episodes) if found else None


def _contrast(assignment: tuple[int, ...], terms: tuple[int, ...]) -> int:
    """+1/-1 sign for one cell in the contrast defined by `terms`."""

    sign = 1
    for level, active in zip(assignment, terms, strict=True):
        if active:
            sign *= 1 if level else -1
    return sign


def effects(counts: dict[str, tuple[int, int]], resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    labels = [c for c in CELLS if c in counts]
    if len(labels) != 8:
        missing = [c for c in CELLS if c not in counts]
        print(f"  {len(labels)} of 8 cells present; missing {missing}", file=sys.stderr)

    draws = {c: rng.binomial(counts[c][1], counts[c][0] / counts[c][1], resamples) / counts[c][1] for c in labels}
    # Keep bootstrap rates off the boundary so the logit is finite.
    eps = 1.0 / (2 * max(n for _, n in counts.values()))
    logit = {c: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps))) for c, v in draws.items()}

    out: dict = {"cells": {c: {"successes": counts[c][0], "episodes": counts[c][1],
                               "rate": round(counts[c][0] / counts[c][1], 4)} for c in labels}}
    out["effects"] = {}
    for order in (1, 2, 3):
        for terms in itertools.product((0, 1), repeat=3):
            if sum(terms) != order:
                continue
            name = "".join(f for f, t in zip(FACTORS, terms, strict=True) if t)
            n_cells = len(labels)
            prob = sum(_contrast(CELLS[c], terms) * draws[c] for c in labels) / (n_cells / 2)
            lodd = sum(_contrast(CELLS[c], terms) * logit[c] for c in labels) / (n_cells / 2)
            out["effects"][name] = {
                "order": order,
                "probability_points": round(float(prob.mean()) * 100, 2),
                "probability_95": [round(float(v) * 100, 2) for v in np.percentile(prob, [2.5, 97.5])],
                "log_odds": round(float(lodd.mean()), 3),
                "log_odds_95": [round(float(v), 3) for v in np.percentile(lodd, [2.5, 97.5])],
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=4070)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    search = [ROOT / "artifacts/campaign/factorial", ROOT / "artifacts/campaign/rgbdcohorts"]
    counts: dict[str, tuple[int, int]] = {}
    for label in CELLS:
        got = _load_counts(label, search)
        if got and got[1]:
            counts[label] = got

    if not counts:
        raise SystemExit("no cells found; has the factorial run?")

    result = effects(counts, args.resamples, args.seed)
    print("cells:")
    for label, cell in result["cells"].items():
        noised, kinematic, lead_in = CELLS[label]
        print(f"  N={noised} K={kinematic} L={lead_in}  {label:34s} {cell['successes']:3d}/{cell['episodes']:<3d} {cell['rate']:.4f}")
    print("\neffects (positive means the change helps):")
    for name, e in sorted(result["effects"].items(), key=lambda kv: (kv[1]["order"], kv[0])):
        print(f"  {name:4s} order {e['order']}  "
              f"{e['probability_points']:+7.2f} pts {str(e['probability_95']):>20s}   "
              f"log-odds {e['log_odds']:+6.3f} {str(e['log_odds_95']):>18s}")

    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
