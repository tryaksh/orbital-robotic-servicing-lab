#!/usr/bin/env python3
"""Does the quantity a criterion bounds still predict the failure it names?

Five of this project's closed-form criteria came back as mismatches against the
simulator, and the standing explanation was a rule stated in words: a bound
transfers when nothing in the process corrects the quantity it bounds, and fails
to transfer when something does. That rule was inferred from which criteria
happened to work. It predicted nothing, and it could not be wrong.

This makes it a measurement. For each criterion, take the quantity it bounds as
the episode hands it over, and ask how well that quantity ranks the episodes that
later fail in the specific mode the criterion predicts. The statistic is the area
under the ROC curve, which for this purpose reads directly: **the probability
that a failing episode entered with a larger deviation than a surviving one.**

* **AUC near 0.5** -- the hand-over value carries no information about the
  outcome. Something between hand-over and the seated plane has erased it, so a
  bound written on that quantity cannot govern the result and will not transfer,
  however correct its arithmetic.
* **AUC well above 0.5** -- the deviation survives to the outcome. Nothing
  corrects it, and a bound written on it should transfer.

The rule then has a falsifiable form. Take one design point, remove the geometry
that does the correcting, change nothing else, and the statistic must move from
the first case to the second. That experiment exists: the entry flares, deleted
at 6 mm of lateral clearance per side.

Retention is not a success rate and does not replace one. A criterion can retain
perfectly and still bound the wrong quantity -- the rail-indexing axis does
exactly that, where the surviving deviation is a settling velocity and the bound
was written on a position. Retention says whether a bound *can* govern, not
whether it is the right bound.

Mode definitions are imported from ``report_boundary_failure_modes.py`` rather
than restated, so the partition here cannot drift away from the one every
boundary report already uses.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

_spec = importlib.util.spec_from_file_location("_modes", ROOT / "scripts" / "report_boundary_failure_modes.py")
assert _spec is not None and _spec.loader is not None
modes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(modes)

#: ``criterion: (hand-over quantity, unit scale, unit, the mode it predicts)``.
#: Each pairing is the criterion's own statement: the grip criterion bounds how
#: far a pad may slide off the pin and predicts losing the module during the
#: pull; the entry criterion bounds the attitude the transit hands over at and
#: predicts a jam.
CRITERIA: dict[str, tuple[str, float, str, str]] = {
    "grip": ("grip_error_m", 1000.0, "mm", "lost_before_delivery"),
    "entry": ("grip_attitude_rad", 1000.0, "mrad", "jammed_in_the_bay"),
}

#: Bootstrap resamples for the interval. The cohorts are 64 to 192 episodes, so
#: the interval is wide and saying so is the point.
RESAMPLES = 2000

#: Below this many events the statistic is not reported as a measurement. Two
#: events give an AUC of 1.000 with a bootstrap interval of [1.0, 1.0], which
#: looks like certainty and is an artifact of there being nothing to resample.
MINIMUM_EVENTS = 10


def auc(values: np.ndarray, positive: np.ndarray) -> float:
    """P(a positive episode ranks above a negative one), ties counted as half."""

    pos, neg = values[positive], values[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average ranks over ties so a quantity that is constant scores exactly 0.5.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap(values: np.ndarray, positive: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    scores = []
    for _ in range(resamples):
        pick = rng.integers(0, n, n)
        sampled = positive[pick]
        if sampled.all() or not sampled.any():
            continue
        scores.append(auc(values[pick], sampled))
    if not scores:
        return float("nan"), float("nan")
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def retention_for_point(path: Path, seed: int) -> dict:
    archive = np.load(path, allow_pickle=True)
    rows = archive["rows"].astype(float)
    fields = [str(f) for f in archive["fields"]]
    partition = modes.decompose(rows, fields)

    out: dict = {"episodes": int(rows.shape[0]), "criteria": {}}
    for criterion, (column, scale, unit, mode) in CRITERIA.items():
        if column not in fields:
            continue
        values = rows[:, fields.index(column)] * scale
        mask = np.asarray(partition["masks"][mode], dtype=bool) if "masks" in partition else None
        if mask is None:
            mask = _mode_mask(rows, fields, mode)
        events = int(mask.sum())
        entry: dict = {
            "hand_over_quantity": column,
            "unit": unit,
            "predicted_mode": mode,
            "events": events,
            "median_at_hand_over": round(float(np.median(values)), 4),
        }
        if events == 0:
            # Not the same as "the process erased the signal". The failure this
            # criterion predicts did not happen at all, so there is nothing for
            # the hand-over value to rank. Where a corrector was present and the
            # mode is empty, that is the corrector working, and it is a stronger
            # statement than a weakened correlation -- but it is a different one.
            entry["verdict"] = "mode_never_occurred"
        elif events < MINIMUM_EVENTS:
            entry["verdict"] = "too_few_events"
            entry["retention_auc"] = round(auc(values, mask), 4)
        else:
            score = auc(values, mask)
            low, high = bootstrap(values, mask, RESAMPLES, seed)
            entry["retention_auc"] = round(score, 4)
            entry["bootstrap_95"] = [round(low, 4), round(high, 4)]
            entry["verdict"] = "retained" if low > 0.5 else "erased_by_the_process"
        out["criteria"][criterion] = entry
    return out


def _mode_mask(rows: np.ndarray, fields: list[str], mode: str) -> np.ndarray:
    """The same partition ``decompose`` uses, as a boolean mask."""

    column = {name: rows[:, fields.index(name)] for name in fields}
    success = column["success"] > 0.5
    timed_out = column["timed_out_in_phase"].astype(int)
    reached = column["reached_phase"].astype(int)
    lost_before_delivery = np.isin(timed_out, modes.DELIVERY_PHASES)
    lost_in_transit = timed_out == modes.TRANSIT_PHASE
    jammed = (~success) & ~(lost_before_delivery | lost_in_transit) & (reached <= modes.INSERT_PHASE)
    return {
        "lost_before_delivery": lost_before_delivery,
        "lost_in_transit": lost_in_transit,
        "jammed_in_the_bay": jammed,
    }[mode]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--point", action="append", required=True, dest="points")
    parser.add_argument("--seed", type=int, default=4070)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    result: dict = {
        "title": "Does the quantity each criterion bounds still predict the failure it names?",
        "evidence_type": "criterion_retention_under_closed_loop_correction",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_revision": git_source_revision(ROOT),
        "statistic": (
            "Area under the ROC curve of the hand-over quantity against the failure mode the "
            "criterion predicts. 0.5 means the hand-over value carries no information about the "
            "outcome, so something between hand-over and the seated plane corrected it and a bound "
            "written on that quantity cannot govern the result. Above 0.5 means the deviation "
            "survives to the outcome."
        ),
        "what_this_does_not_say": (
            "Retention is not correctness. A criterion can retain and still bound the wrong "
            "quantity: the rail-indexing axis retains a settling velocity while its bound is "
            "written on a position. Retention says whether a bound can govern, not whether it is "
            "the right bound. It is also not a success rate and must never be quoted as one."
        ),
        "arms": {},
    }

    for label, directory in args.arm:
        arm: dict = {}
        for point in args.points:
            path = Path(directory) / f"{point}.npz"
            if not path.exists():
                continue
            arm[point] = retention_for_point(path, args.seed)
        result["arms"][label] = {"directory": directory, "points": arm}

    for label, arm in result["arms"].items():
        print(f"\n{label}  ({arm['directory']})")
        for point, record in arm["points"].items():
            print(f"  {point}  n={record['episodes']}")
            for criterion, entry in record["criteria"].items():
                head = f"    {criterion:6s} -> {entry['predicted_mode']:20s} events={entry['events']:3d}  "
                if entry["verdict"] == "mode_never_occurred":
                    print(head + "the failure this criterion predicts did not occur")
                elif entry["verdict"] == "too_few_events":
                    print(head + f"AUC {entry['retention_auc']:.3f}  too few events to report")
                else:
                    low, high = entry["bootstrap_95"]
                    print(head + f"AUC {entry['retention_auc']:.3f} [{low}, {high}]  {entry['verdict']}")

    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
