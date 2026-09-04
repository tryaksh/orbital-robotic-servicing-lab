#!/usr/bin/env python3
"""Rank episodes by a recorded quantity against the failure mode a criterion names.

**Read the limitation before the output.** This was written to test whether a
deviation *at hand-over* governs the outcome, and it cannot answer that. The
episode archives contain no hand-over value: ``_freeze`` in
``run_workflow_demo.py`` stores each row at the moment of judgement, because a
completed workflow idles afterwards and the state at the timeout is not the state
that was achieved. Every quantity here therefore describes the state the outcome
was decided in. What comes out is a **concurrent association**, not a prediction,
and the first version of this script claimed otherwise; see
``evidence/RETRACTED.md``.

Two consequences, both load-bearing:

* **Quantities in the success predicate are circular.** ``SEATED_CONDITIONS``
  includes ``linear_velocity`` and ``angular_velocity``, so an episode fails
  partly because its velocity is high. Ranking failures by velocity returns an
  AUC of 1.000 and discovers nothing. Those columns are flagged rather than
  removed, because seeing the circularity is more useful than hiding it.
* **The rest are still worth reading, as signatures.** The grip criterion bounds
  how far a pad may slide off the pin; ``grip_error_m`` is not part of the
  success predicate, so an elevated value on the episodes that fail in the mode
  the criterion names is real information about mechanism. It is the same
  statement the grip signature in ``report_boundary_failure_modes.py`` makes,
  with a rank statistic and an interval instead of a difference of means.

To get the predictive version, the episode row has to carry the hand-over state
as its own column and the boundary arms have to be re-run. That is a change to
``run_workflow_demo.py``, not an analysis of archives that exist.

Mode definitions are imported from ``report_boundary_failure_modes.py`` rather
than restated, so the partition cannot drift from the one every boundary report
uses.
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

#: ``criterion: (recorded quantity, unit scale, unit, the mode it predicts)``.
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

#: Quantities an episode records that could plausibly govern an outcome. The
#: scan ranks these when a criterion's own quantity fails to explain the failure
#: it predicts, which is how a *missing* criterion announces itself.
#: Columns that appear in the success predicate. Ranking failures by one of
#: these reads the predicate back and cannot be evidence of anything.
CIRCULAR_WITH_SUCCESS = frozenset({"blade_linear_velocity_mps", "blade_angular_velocity_radps"})

SCANNABLE: dict[str, tuple[float, str]] = {
    "grip_error_m": (1000.0, "mm"),
    "grip_attitude_rad": (1000.0, "mrad"),
    "tool_to_handle_error_m": (1000.0, "mm"),
    "tool_to_handle_orientation_rad": (1000.0, "mrad"),
    "blade_linear_velocity_mps": (1000.0, "mm/s"),
    "blade_angular_velocity_radps": (1000.0, "mrad/s"),
    "blade_centre_x_m": (1000.0, "mm"),
    "perceived_error_mean_m": (1000.0, "mm"),
    "cycle_time_s": (1.0, "s"),
}

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
            "quantity_at_judgement": column,
            "unit": unit,
            "predicted_mode": mode,
            "events": events,
            "median_at_judgement": round(float(np.median(values)), 4),
        }
        if events == 0:
            # Not the same as "the process erased the signal". The failure this
            # criterion predicts did not happen at all, so there is nothing for
            # the recorded value to rank. Where a corrector was present and the
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
            entry["verdict"] = "elevated_on_the_failures" if low > 0.5 else "not_associated"
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


def scan_point(path: Path, mode: str, seed: int) -> dict:
    """Rank every recorded quantity by how well it explains one failure mode.

    Used when a criterion's own quantity does not explain the failure it
    predicts. Two answers are possible and they mean opposite things: nothing
    explains it, or something the criterion set does not bound explains it. The
    second is a missing criterion, and it names the quantity to bound.

    **This is a scan and it must be read as one.** Ranking nine quantities and
    reporting the best inflates any single interval. A result here is a
    hypothesis unless the quantity was named before it was measured.
    """

    archive = np.load(path, allow_pickle=True)
    rows = archive["rows"].astype(float)
    fields = [str(f) for f in archive["fields"]]
    mask = _mode_mask(rows, fields, mode)
    events = int(mask.sum())

    out: dict = {"mode": mode, "events": events, "episodes": int(rows.shape[0]), "quantities": {}}
    if events < MINIMUM_EVENTS:
        out["verdict"] = "too_few_events"
        return out
    for column, (scale, unit) in SCANNABLE.items():
        if column not in fields:
            continue
        values = rows[:, fields.index(column)] * scale
        score = auc(values, mask)
        low, high = bootstrap(values, mask, RESAMPLES, seed)
        circular = column in CIRCULAR_WITH_SUCCESS
        out["quantities"][column] = {
            "unit": unit,
            "retention_auc": round(score, 4),
            "bootstrap_95": [round(low, 4), round(high, 4)],
            "median": round(float(np.median(values)), 4),
            "explains": bool(low > 0.5) and not circular,
            "circular_with_success_predicate": circular,
        }
    out["caveat"] = (
        "A scan over nine quantities. Reporting the best inflates any single interval, so a "
        "result here is a hypothesis unless the quantity was named before it was measured."
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--point", action="append", required=True, dest="points")
    parser.add_argument("--seed", type=int, default=4070)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--scan_mode",
        help=(
            "Also rank every recorded quantity against this failure mode, for points where the "
            "criterion's own quantity does not explain it. This is how a missing criterion is "
            "found: something the criterion set does not bound explains the failure."
        ),
    )
    args = parser.parse_args()

    result: dict = {
        "title": "Does the quantity each criterion bounds still predict the failure it names?",
        "evidence_type": "criterion_retention_under_closed_loop_correction",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_revision": git_source_revision(ROOT),
        "statistic": (
            "Area under the ROC curve of the quantity as recorded at judgement, against the failure "
            "mode the criterion predicts. This is a concurrent association and not a prediction: the "
            "archives carry no hand-over value, so a high score means the quantity is elevated on the "
            "episodes that fail, not that it caused them to."
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
            if args.scan_mode:
                arm[point]["scan"] = scan_point(path, args.scan_mode, args.seed)
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
            scan = record.get("scan")
            if scan and scan.get("quantities"):
                ranked = sorted(
                    scan["quantities"].items(), key=lambda kv: kv[1]["retention_auc"], reverse=True
                )
                print(f"    scan of {scan['mode']} ({scan['events']} events), best first:")
                for name, q in ranked[:4]:
                    lo, hi = q["bootstrap_95"]
                    mark = (
                        "  <- CIRCULAR: in the success predicate"
                        if q["circular_with_success_predicate"]
                        else ("  <- associated with the failure" if q["explains"] else "")
                    )
                    print(f"      {name:30s} AUC {q['retention_auc']:.3f} [{lo}, {hi}]{mark}")

    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
