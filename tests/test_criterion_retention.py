"""Unit tests for the criterion-retention statistic.

The claim this statistic supports is strong -- that a closed-form bound governs
the closed-loop outcome only when the process leaves the bounded quantity alone
-- so the statistic itself has to be beyond argument on the cases where the
answer is known by construction. Perfect separation must read 1.0, an inverted
relationship 0.0, and a quantity carrying no information exactly 0.5 even when
every value is identical, which is the case a naive rank implementation gets
wrong.

The guards matter as much as the arithmetic. Two events give an AUC of 1.000
with a bootstrap interval of [1.0, 1.0]; reported without a floor that reads as
certainty and is an artifact of having nothing to resample.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("measure_criterion_retention", ROOT / "scripts/measure_criterion_retention.py")
assert _spec is not None and _spec.loader is not None
retention = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retention)


def test_perfect_separation_reads_one() -> None:
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    positive = np.array([False, False, False, True, True, True])
    assert retention.auc(values, positive) == pytest.approx(1.0)


def test_inverted_separation_reads_zero() -> None:
    values = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])
    positive = np.array([False, False, False, True, True, True])
    assert retention.auc(values, positive) == pytest.approx(0.0)


def test_a_constant_quantity_reads_exactly_one_half() -> None:
    """A quantity the process has flattened carries no information, and must say so.

    Without tie-averaging this returns 1.0 or 0.0 depending on argsort order,
    which would turn an erased signal into the strongest possible evidence.
    """

    values = np.full(20, 7.0)
    positive = np.array([True] * 8 + [False] * 12)
    assert retention.auc(values, positive) == pytest.approx(0.5)


def test_a_symmetric_arrangement_reads_exactly_one_half() -> None:
    """Positives spread symmetrically around the negatives carry no ranking information.

    Written as an exact case rather than an approximate one. An interleave looks
    symmetric and is not -- taking every other rank offsets the positives by one
    place and lands at 0.45 -- and a test with a loose tolerance around the wrong
    number would pass while hiding a real bias.
    """

    values = np.array([0.0, 1.0, 2.0, 3.0])
    positive = np.array([True, False, False, True])
    assert retention.auc(values, positive) == pytest.approx(0.5)


def test_no_events_is_not_the_same_as_no_signal() -> None:
    values = np.arange(10, dtype=float)
    positive = np.zeros(10, dtype=bool)
    assert np.isnan(retention.auc(values, positive))


def test_the_event_floor_is_high_enough_to_kill_the_two_event_artifact() -> None:
    assert retention.MINIMUM_EVENTS > 2


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.0, 1.0, 80), rng.normal(1.5, 1.0, 40)])
    positive = np.array([False] * 80 + [True] * 40)
    point = retention.auc(values, positive)
    low, high = retention.bootstrap(values, positive, 500, seed=0)
    assert low < point < high
    assert low > 0.5, "a real 1.5-sigma separation should be called retained"
