#!/usr/bin/env python3
"""Score three predictors against configurations none of them had seen.

The scoring rule was fixed in `docs/PREREGISTRATION_jam_prediction.md` before
any of the cohorts existed, and the three predictors' claims are in
`configs/prediction_claims.json` so that scoring is mechanical rather than
assembled by hand once the answer is visible.

Three tests, in the order the pre-registration puts them:

* **S1, phase identification** -- which phase's failure rate actually rose most
  against the reference, and did the predictor name it? This is the headline.
  An engineer deciding where to spend effort is better served by the right phase
  with a wrong number than a close number on the wrong phase.
* **S2, direction** -- per phase the predictor spoke about, did the rate move the
  way it said? "Unchanged" counts as correct when the observed rate falls inside
  the reference's Wilson 95% interval, so a predictor is not punished for noise.
* **S3, rate error** -- absolute difference between predicted and observed, for
  predictors that give a rate. Reported and not thresholded: with three seeds
  the interval is wide and a small error is not a result.

Predictors are scored only on what they claim. The clearance rule produces a
direction and no rate, so it is scored on direction alone -- neither penalised
for silence on phases it has no model of, nor credited for them.

    python scripts/score_prediction_test.py --report evidence/prediction_scorecard_v1.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from handoff_qualification.records import CATASTROPHIC_RESIDUAL_M  # noqa: E402
from handoff_qualification.residual import wilson  # noqa: E402
from scripts.predict_configuration_change import failing_phase  # noqa: E402
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

PHASE_NAMES = ["capture", "seat", "extract", "transit", "insert", "done"]
SCORED_PHASES = ("capture", "seat", "extract", "transit", "insert", "terminal_gate")


def observe(pattern: str) -> dict | None:
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        return None
    counts: dict[str, int] = {}
    episodes = 0
    residuals: list[float] = []
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        fields = [str(n) for n in archive["fields"]]
        rows = archive["rows"].astype(float)
        column = {name: index for index, name in enumerate(fields)}
        episodes += len(rows)
        residuals.extend(rows[:, column["lateral_error_m"]].tolist())
        for row in rows:
            phase = failing_phase(row, column, PHASE_NAMES)
            if phase:
                counts[phase] = counts.get(phase, 0) + 1
    residual = np.array(residuals)
    delivered = residual < CATASTROPHIC_RESIDUAL_M
    return {
        "cohorts": len(paths),
        "episodes": episodes,
        "phase_counts": counts,
        "phase_rates": {p: counts.get(p, 0) / episodes for p in SCORED_PHASES},
        "delivery_rate": float(delivered.mean()),
        "jam_rate_among_delivered": float(
            ((residual >= 0.010) & (residual < CATASTROPHIC_RESIDUAL_M)).sum() / delivered.sum()
        )
        if delivered.any()
        else 0.0,
    }


def biggest_riser(observed: dict, reference: dict) -> str | None:
    """Which phase actually rose most. None when nothing rose."""

    rises = {
        phase: observed["phase_rates"].get(phase, 0.0) - reference["phase_rates"].get(phase, 0.0)
        for phase in SCORED_PHASES
    }
    phase, delta = max(rises.items(), key=lambda kv: kv[1])
    return phase if delta > 0 else None


def unchanged_ok(observed_rate: float, reference: dict, phase: str) -> bool:
    """Is an 'unchanged' call defensible against the reference's own interval?"""

    count = reference["phase_counts"].get(phase, 0)
    low, high = wilson(count, reference["episodes"])
    return low <= observed_rate <= high


def score(name: str, claim: dict, observed: dict, reference: dict) -> dict:
    produces = claim.get("produces", [])
    risen = biggest_riser(observed, reference)
    result: dict = {"predictor": name, "produces": produces}

    if "phase" in produces:
        result["s1_named_phase"] = claim["phase"]
        result["s1_actual_phase"] = risen
        result["s1_correct"] = (risen is not None and claim["phase"] == risen) or (
            risen is None and claim.get("direction") == "unchanged"
        )
    else:
        result["s1_correct"] = None

    phase = claim["phase"]
    observed_rate = observed["phase_rates"].get(phase, 0.0)
    reference_rate = reference["phase_rates"].get(phase, 0.0)
    if claim["direction"] == "rises":
        result["s2_correct"] = observed_rate > reference_rate and not unchanged_ok(
            observed_rate, reference, phase
        )
    elif claim["direction"] == "falls":
        result["s2_correct"] = observed_rate < reference_rate and not unchanged_ok(
            observed_rate, reference, phase
        )
    else:
        result["s2_correct"] = unchanged_ok(observed_rate, reference, phase)
    result["s2_observed_rate"] = observed_rate
    result["s2_reference_rate"] = reference_rate

    if claim.get("rate") is not None:
        result["s3_predicted"] = claim["rate"]
        result["s3_observed"] = observed_rate
        result["s3_absolute_error"] = abs(claim["rate"] - observed_rate)
    elif claim.get("rate_among_delivered") is not None:
        floor = claim["rate_among_delivered"]
        seen = observed["jam_rate_among_delivered"]
        result["s3_predicted"] = f">= {floor}"
        result["s3_observed"] = seen
        result["s3_absolute_error"] = max(0.0, floor - seen)
    elif claim.get("delivery_rate") is not None:
        result["s3_predicted"] = claim["delivery_rate"]
        result["s3_observed"] = observed["delivery_rate"]
        result["s3_absolute_error"] = abs(claim["delivery_rate"] - observed["delivery_rate"])
    else:
        result["s3_absolute_error"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--claims", type=Path, default=ROOT / "configs/prediction_claims.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.claims.read_text(encoding="utf-8"))
    reference = observe(spec["reference"]["cohort"])
    if reference is None:
        print("reference cohort missing")
        return 1
    print(f"reference: {reference['episodes']} episodes, {spec['reference']['note']}")
    print(f"   phase rates: { {k: round(v, 4) for k, v in reference['phase_rates'].items() if v} }")
    print()

    scored: dict[str, dict] = {}
    for label, entry in spec["configurations"].items():
        observed = observe(entry["cohort"])
        if observed is None:
            print(f"{label}: not yet run, skipping")
            continue
        print(f"=== {label} ===")
        print(
            f"   {observed['episodes']} episodes, delivery {observed['delivery_rate']:.4f}, "
            f"phase rates { {k: round(v, 4) for k, v in observed['phase_rates'].items() if v} or 'no failures'}"
        )
        risen = biggest_riser(observed, reference)
        print(f"   phase that actually rose most: {risen or 'none - nothing rose'}")
        rows = {
            name: score(name, claim, observed, reference)
            for name, claim in entry["predictors"].items()
        }
        print(f"   {'predictor':>30} {'S1 phase':>10} {'S2 dir':>8} {'S3 err':>10}")
        for name, row in rows.items():
            s1 = "-" if row["s1_correct"] is None else ("yes" if row["s1_correct"] else "NO")
            s2 = "yes" if row["s2_correct"] else "NO"
            s3 = "-" if row["s3_absolute_error"] is None else f"{row['s3_absolute_error']:.4f}"
            print(f"   {name:>30} {s1:>10} {s2:>8} {s3:>10}")
        print()
        scored[label] = {"observed": observed, "phase_that_rose": risen, "predictors": rows}

    document = {
        "title": "Three predictors scored against configurations none had seen",
        "evidence_type": "prospective_prediction_scored",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": git_source_revision(ROOT),
        "scoring_rule": "docs/PREREGISTRATION_jam_prediction.md, third addendum",
        "claims_file": args.claims.as_posix(),
        "reference": reference,
        "configurations": scored,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "The predictions and the scoring rule were both committed before any "
            "episode of either configuration existed; git history is the record.",
            "Three seeds and 192 episodes a configuration. A phase whose reference "
            "rate is a handful of events has a wide interval and a rate error of a "
            "few percent is not a result.",
            "Two configurations is two independent transfer cases however many "
            "episodes each contains.",
            "The clearance rule is scored on direction alone because that is all it "
            "produces. It is neither penalised for silence on phases it cannot "
            "represent nor credited for them.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
