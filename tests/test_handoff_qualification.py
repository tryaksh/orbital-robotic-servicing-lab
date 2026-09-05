"""Contracts for the residual reading of a handoff.

These are CPU tests on synthetic arrays and on two committed evidence files.
They do not need a simulator, and deliberately do not read episode archives
under `artifacts/`, which is gitignored and absent in CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from handoff_qualification.records import CATASTROPHIC_RESIDUAL_M, EpisodeRecord
from handoff_qualification.residual import (
    MAXIMUM_EXTRAPOLATION,
    MINIMUM_ARRIVED,
    empirical_pass_rate,
    fit_residual_model,
    qualify,
    wilson,
)


def records(residuals, criterion=0.0025):
    return [
        EpisodeRecord(
            cohort="synthetic",
            env=index,
            success=value < criterion,
            residual_lateral_m=float(value),
            residual_orientation_rad=0.0,
            reached_handoff=True,
        )
        for index, value in enumerate(residuals)
    ]


def test_empirical_rate_is_the_counted_rate_at_the_run_criterion():
    """The identity that makes the reframing safe rather than clever.

    If these ever disagree, the success label is not a threshold on the
    residual and everything downstream needs rechecking.
    """

    values = [0.001, 0.002, 0.0024, 0.0026, 0.003, 0.01]
    sample = records(values)
    rate, _ = empirical_pass_rate(sample, 0.0025)
    assert rate == sum(r.success for r in sample) / len(sample)
    assert rate == pytest.approx(3 / 6)


def test_criterion_transfer_is_monotone():
    sample = records([0.001, 0.002, 0.003, 0.004, 0.005])
    rates = [empirical_pass_rate(sample, c)[0] for c in (0.0015, 0.0025, 0.0035, 0.0045, 0.0055)]
    assert rates == sorted(rates)
    assert rates[0] == pytest.approx(0.2)
    assert rates[-1] == pytest.approx(1.0)


def test_catastrophic_episodes_are_separated_not_pooled():
    """216 mm is not a mis-seat; it is a module that never left the source bay."""

    sample = records([0.002] * 10 + [0.22, 1.2])
    model = fit_residual_model([r.residual_lateral_m for r in sample])
    assert model.catastrophic == 2
    assert model.arrived == 10
    assert model.catastrophic_rate == pytest.approx(2 / 12)
    # A catastrophic episode can never pass, at any criterion.
    assert model.pass_rate(CATASTROPHIC_RESIDUAL_M * 10) <= model.arrived / model.episodes


def test_qualify_refuses_a_thin_cohort():
    result = qualify(records([0.002] * (MINIMUM_ARRIVED - 1)), 0.0025, bootstrap=50)
    assert not result.supported
    assert "arrived" in result.insufficient_evidence
    # The counted rate is still reported: a refusal to model is not a refusal
    # to answer what was actually observed.
    assert result.counted_rate == pytest.approx(1.0)


def test_qualify_refuses_extrapolation_beyond_the_sampled_range():
    rng = np.random.default_rng(0)
    sample = records(rng.uniform(0.002, 0.004, size=60))
    far = qualify(sample, 0.00001, bootstrap=50)
    assert not far.supported
    assert "extrapolation" in far.insufficient_evidence
    near = qualify(sample, 0.003, bootstrap=50)
    assert near.supported


def test_extrapolation_threshold_is_relative_to_the_sampled_span():
    """A wide cohort may answer where a narrow one must refuse."""

    rng = np.random.default_rng(1)
    narrow = records(rng.uniform(0.00200, 0.00210, size=60))
    wide = records(rng.uniform(0.00100, 0.01000, size=60))
    assert not qualify(narrow, 0.0015, bootstrap=50).supported
    assert qualify(wide, 0.0015, bootstrap=50).supported
    assert 0 < MAXIMUM_EXTRAPOLATION < 1


def test_wilson_matches_the_projects_evaluation_module():
    from zero_g_blade_swap.evaluation import wilson_interval

    for successes, trials in ((0, 16), (110, 192), (32, 48), (16, 16)):
        mine = wilson(successes, trials)
        theirs = wilson_interval(successes, trials)
        assert mine[0] == pytest.approx(theirs[0], abs=1e-9)
        assert mine[1] == pytest.approx(theirs[1], abs=1e-9)


def test_a_zero_rate_cohort_still_yields_a_distance_to_qualification():
    """The case the tool exists for: 0 of 16 is not a measurement of anything."""

    from scripts.qualify_handoff import criterion_for_rate

    residuals = np.array([0.004, 0.005, 0.006, 0.0065, 0.007, 0.008, 0.009, 0.0104])
    rate, interval = empirical_pass_rate(records(residuals), 0.0025)
    assert rate == 0.0
    assert interval[1] > 0.2  # the rate alone is compatible with almost anything
    needed = criterion_for_rate(residuals, 0.95)
    assert needed == pytest.approx(0.0104)
    assert needed / 0.0025 == pytest.approx(4.16, abs=0.02)


def test_no_criterion_qualifies_a_cohort_that_did_not_arrive():
    from scripts.qualify_handoff import criterion_for_rate

    residuals = np.array([0.002] * 10 + [0.5] * 10)
    assert criterion_for_rate(residuals, 0.95) is None
    assert criterion_for_rate(residuals, 0.50) == pytest.approx(0.002)


def test_lognormal_fit_is_reported_with_its_distance():
    """The fit is kept, and so is the evidence that it is a poor one here.

    On the 192-episode nominal cohort the lognormal returned 0.650 against a
    counted 0.573 with an interval that excluded the truth. Nothing in the
    package is allowed to present a fit without its KS distance.
    """

    rng = np.random.default_rng(2)
    sample = records(np.exp(rng.normal(math.log(0.002), 0.4, size=80)))
    result = qualify(sample, 0.0025, bootstrap=200)
    assert result.fit_distance is not None
    assert 0.0 <= result.fit_distance <= 1.0
    if result.supported:
        assert result.modelled_interval[0] <= result.modelled_rate <= result.modelled_interval[1]
