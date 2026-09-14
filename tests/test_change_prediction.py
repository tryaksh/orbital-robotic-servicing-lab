"""Contracts for the configuration-change procedure.

The procedure's value is as much in what it declines to say as in what it says,
so most of these test refusals. A tool that answers every question is not
useful here: a corpus that varied a part rather than a dimension cannot support
a per-dimension answer, and saying so is the difference between a method and a
guess.

CPU only, on synthetic cohorts. Nothing here reads `artifacts/`, which is
gitignored and absent in CI.
"""

from __future__ import annotations

import pytest

from handoff_qualification.change_prediction import (
    Cohort,
    predict,
    sensitivity_corpus,
)
from handoff_qualification.records import EpisodeRecord

DIMENSIONS = ["width", "thickness", "lateral_clearance"]


def cohort(label, width, thickness, clearance, episodes=100, capture=0, extract=0, insert=0):
    records: list[EpisodeRecord] = []
    phases: list[str | None] = []
    for index in range(episodes):
        if index < capture:
            residual, phase = 0.5, "capture"
        elif index < capture + extract:
            residual, phase = 0.5, "extract"
        elif index < capture + extract + insert:
            residual, phase = 0.032, "insert"
        else:
            residual, phase = 0.001, None
        records.append(
            EpisodeRecord(
                cohort=label,
                env=index,
                success=residual < 0.0025,
                residual_lateral_m=residual,
                residual_orientation_rad=0.0,
                reached_handoff=phase is None,
            )
        )
        phases.append(phase)
    return Cohort(
        label=label,
        dimensions={"width": width, "thickness": thickness, "lateral_clearance": clearance},
        records=records,
        timed_out_phase=phases,
    )


def reference():
    return cohort("nominal", 0.130, 0.020, 0.011065, capture=2, extract=1)


def confounded_corpus():
    """A corpus that varied *parts*, which is what campaigns actually do."""

    return [
        cohort("120x16", 0.120, 0.016, 0.016065, capture=17, extract=4),
        cohort("140x26", 0.140, 0.026, 0.006065, capture=1, extract=8),
        cohort("relief0", 0.130, 0.020, 0.006452, capture=2, extract=1, insert=6),
    ]


def test_a_dimension_varied_only_alongside_others_is_reported_confounded():
    report = sensitivity_corpus(reference(), confounded_corpus(), DIMENSIONS)
    assert report["thickness"]["usable"] is False
    assert "width" in report["thickness"]["confounded_with"]
    # Clearance did move on its own, in the relief cohort.
    assert report["lateral_clearance"]["usable"] is True
    assert report["lateral_clearance"]["cohorts_varying_it_alone"] == ["relief0"]


def test_the_procedure_refuses_a_change_its_corpus_confounds():
    """The case the rack's own corpus lands on, and the reason the tool exists."""

    change = {"width": 0.130, "thickness": 0.026, "lateral_clearance": 0.011065}
    predictions = predict(reference(), confounded_corpus(), change, 0.0025)
    assert all(not p.answered for p in predictions.values())
    reason = predictions["extract"].refused
    assert "thickness" in reason and "cannot be attributed" in reason
    assert "width" in predictions["extract"].confounded_with


def test_it_answers_where_a_dimension_moved_alone():
    change = {"width": 0.130, "thickness": 0.020, "lateral_clearance": 0.004}
    predictions = predict(reference(), confounded_corpus(), change, 0.0025)
    assert predictions["insert"].answered
    assert predictions["insert"].direction == "rises"
    assert predictions["insert"].driven_by == ("lateral_clearance",)
    # A phase the change has no evidence of touching must not drift.
    assert predictions["transit"].direction == "unchanged"


def test_it_refuses_a_dimension_no_cohort_ever_varied():
    thin = [cohort("relief0", 0.130, 0.020, 0.006452, capture=2, extract=1, insert=6)]
    change = {"width": 0.180, "thickness": 0.020, "lateral_clearance": 0.011065}
    predictions = predict(reference(), thin, change, 0.0025)
    assert not predictions["capture"].answered
    assert "no cohort" in predictions["capture"].refused


def test_the_predicted_rate_stays_a_rate():
    """A linear projection can leave [0, 1]; what it reports may not."""

    change = {"width": 0.130, "thickness": 0.020, "lateral_clearance": 0.0001}
    predictions = predict(reference(), confounded_corpus(), change, 0.0025)
    insert = predictions["insert"]
    assert insert.answered
    assert 0.0 <= insert.rate <= 1.0
    assert 0.0 <= insert.rate_interval[0] <= insert.rate_interval[1] <= 1.0


def test_a_change_that_moves_nothing_predicts_nothing_moving():
    unchanged = dict(reference().dimensions)
    predictions = predict(reference(), confounded_corpus(), unchanged, 0.0025)
    for prediction in predictions.values():
        assert not prediction.answered
        assert "any proposed dimension" in prediction.refused


def test_delivery_and_precision_are_available_per_cohort():
    reference_cohort = reference()
    assert reference_cohort.delivery_rate() == pytest.approx(0.97)
    assert reference_cohort.precision_given_delivery(0.0025) == pytest.approx(1.0)
    jammed = confounded_corpus()[2]
    # Jams are delivered but out of tolerance: they cost precision, not delivery.
    assert jammed.delivery_rate() == pytest.approx(0.97)
    assert jammed.precision_given_delivery(0.0025) < 1.0
