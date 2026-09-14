"""Predict which phase a proposed geometry change will break, before running it.

The engineer-facing question this answers: *I want to change a part or a
fixture. Will my existing controller cope, where will it fail, and what should I
measure?*

The procedure is deliberately not clever. It reads what already happened at
configurations that varied some dimension, turns that into a sensitivity per
phase, and projects a proposed change through it. Everything it knows comes from
cohorts; nothing comes from knowledge of a particular workcell. That is the
point — an engineer with their own cohorts should get their own answer.

**It refuses more often than it answers, on purpose.** Two refusals matter:

* *unvaried* — no cohort in the corpus moved the dimension being proposed, so
  there is no evidence to project. Naming the experiment that would supply it is
  more useful than a number.
* *confounded* — cohorts moved the dimension only together with another, so the
  phase response cannot be attributed between them. This is the common case in
  practice, because campaigns vary a part, not a dimension.

The rack this was built on confounds width with thickness: its two section
cohorts are 120 x 16 and 140 x 26, both narrower-and-thinner or wider-and-
thicker. So the tool says "confounded" and names the two configurations that
would separate them, which is exactly the experiment that was then run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from handoff_qualification.records import CATASTROPHIC_RESIDUAL_M, EpisodeRecord

#: Phases a workflow can fail in. Supplied by the caller's records; these are
#: the names this repository's driver uses and nothing here depends on them.
DEFAULT_PHASES = ("capture", "seat", "extract", "transit", "insert")

#: A dimension must move by at least this fraction of its reference value for a
#: cohort to count as having varied it. Below that the cohort is treated as
#: holding the dimension fixed, so floating-point noise in a config does not
#: manufacture a sensitivity.
VARIED_FRACTION = 0.01


@dataclass(frozen=True)
class Cohort:
    """One configuration that was actually run."""

    label: str
    #: Named dimensions and their values, in consistent units.
    dimensions: dict[str, float]
    records: list[EpisodeRecord]
    #: Phase each undelivered episode timed out in, parallel to `records`.
    timed_out_phase: list[str | None] = field(default_factory=list)

    def delivered(self) -> np.ndarray:
        return np.array(
            [r.residual_lateral_m < CATASTROPHIC_RESIDUAL_M for r in self.records]
        )

    def delivery_rate(self) -> float:
        return float(self.delivered().mean()) if self.records else 0.0

    def precision_given_delivery(self, criterion_m: float) -> float:
        delivered = self.delivered()
        if not delivered.any():
            return 0.0
        residual = np.array([r.residual_lateral_m for r in self.records])[delivered]
        return float((residual < criterion_m).mean())

    def phase_failure_rate(self, phase: str) -> float:
        if not self.records:
            return 0.0
        return sum(1 for p in self.timed_out_phase if p == phase) / len(self.records)


@dataclass(frozen=True)
class Prediction:
    """What the procedure says about one phase, or why it will not say."""

    phase: str
    #: "rises", "falls", "unchanged", or None when refused.
    direction: str | None
    #: Predicted failure rate for this phase, and the span it is bounded to.
    rate: float | None
    rate_interval: tuple[float, float] | None
    reference_rate: float
    #: Set when the procedure declines. The other fields are then None.
    refused: str | None
    #: Dimensions the projection leaned on, and any it could not separate.
    driven_by: tuple[str, ...] = ()
    confounded_with: tuple[str, ...] = ()

    @property
    def answered(self) -> bool:
        return self.refused is None


def _varied(cohort: Cohort, reference: Cohort, dimension: str) -> bool:
    base = reference.dimensions.get(dimension)
    other = cohort.dimensions.get(dimension)
    if base is None or other is None or base == 0.0:
        return False
    return abs(other - base) / abs(base) >= VARIED_FRACTION


def _confounds(
    cohort: Cohort, reference: Cohort, dimension: str, dimensions: list[str]
) -> tuple[str, ...]:
    return tuple(
        other
        for other in dimensions
        if other != dimension and _varied(cohort, reference, other)
    )


def sensitivity_corpus(
    reference: Cohort, cohorts: list[Cohort], dimensions: list[str]
) -> dict[str, dict]:
    """Which dimensions the corpus can speak to, and which it confounds."""

    report: dict[str, dict] = {}
    for dimension in dimensions:
        varying = [c for c in cohorts if _varied(c, reference, dimension)]
        confounds: set[str] = set()
        clean = []
        for cohort in varying:
            with_others = _confounds(cohort, reference, dimension, dimensions)
            if with_others:
                confounds.update(with_others)
            else:
                clean.append(cohort.label)
        report[dimension] = {
            "cohorts_varying_it": [c.label for c in varying],
            "cohorts_varying_it_alone": clean,
            "confounded_with": sorted(confounds),
            "usable": bool(clean),
        }
    return report


def predict(
    reference: Cohort,
    cohorts: list[Cohort],
    change: dict[str, float],
    criterion_m: float,
    phases: tuple[str, ...] = DEFAULT_PHASES,
) -> dict[str, Prediction]:
    """Project a proposed change through what the corpus already measured."""

    dimensions = sorted(set(reference.dimensions) | {d for c in cohorts for d in c.dimensions})
    corpus = sensitivity_corpus(reference, cohorts, dimensions)
    moved = [
        d
        for d, value in change.items()
        if reference.dimensions.get(d) is not None
        and abs(value - reference.dimensions[d]) / max(abs(reference.dimensions[d]), 1e-12)
        >= VARIED_FRACTION
    ]

    predictions: dict[str, Prediction] = {}
    for phase in phases:
        reference_rate = reference.phase_failure_rate(phase)

        # Which moved dimensions does the corpus have any evidence about?
        informative = [d for d in moved if corpus.get(d, {}).get("cohorts_varying_it")]
        if not informative:
            predictions[phase] = Prediction(
                phase=phase,
                direction=None,
                rate=None,
                rate_interval=None,
                reference_rate=reference_rate,
                refused=(
                    "no cohort in the corpus varied "
                    + (", ".join(moved) if moved else "any proposed dimension")
                ),
            )
            continue

        clean = [d for d in informative if corpus[d]["usable"]]
        if not clean:
            predictions[phase] = Prediction(
                phase=phase,
                direction=None,
                rate=None,
                rate_interval=None,
                reference_rate=reference_rate,
                refused=(
                    f"{', '.join(informative)} was only ever varied together with "
                    f"{', '.join(sorted({x for d in informative for x in corpus[d]['confounded_with']}))}; "
                    "the phase response cannot be attributed between them"
                ),
                confounded_with=tuple(
                    sorted({x for d in informative for x in corpus[d]["confounded_with"]})
                ),
            )
            continue

        # Project linearly along each cleanly varied dimension and take the
        # widest span the corpus supports as the interval. A linear projection
        # from two points is the least the evidence allows, not a model of the
        # physics, and the interval says so by being wide.
        estimates = []
        for dimension in clean:
            base = reference.dimensions[dimension]
            delta = change[dimension] - base
            for cohort in cohorts:
                if not _varied(cohort, reference, dimension):
                    continue
                if _confounds(cohort, reference, dimension, dimensions):
                    continue
                span = cohort.dimensions[dimension] - base
                if span == 0:
                    continue
                slope = (cohort.phase_failure_rate(phase) - reference_rate) / span
                estimates.append(reference_rate + slope * delta)
        if not estimates:
            predictions[phase] = Prediction(
                phase=phase,
                direction=None,
                rate=None,
                rate_interval=None,
                reference_rate=reference_rate,
                refused="the corpus varies the dimension but not in a usable direction",
            )
            continue

        rate = float(np.median(estimates))
        low, high = float(min(estimates)), float(max(estimates))
        rate = min(max(rate, 0.0), 1.0)
        interval = (max(0.0, low), min(1.0, high))
        if math.isclose(rate, reference_rate, abs_tol=0.01):
            direction = "unchanged"
        else:
            direction = "rises" if rate > reference_rate else "falls"
        predictions[phase] = Prediction(
            phase=phase,
            direction=direction,
            rate=rate,
            rate_interval=interval,
            reference_rate=reference_rate,
            refused=None,
            driven_by=tuple(clean),
        )
    return predictions
