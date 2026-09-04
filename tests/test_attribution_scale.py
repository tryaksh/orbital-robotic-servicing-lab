"""Unit tests for the scale check on the channel interaction.

The check exists to answer one reviewer question -- whether an interaction stated
as a difference of proportions survives being restated as an odds ratio -- so the
test that matters is that it says *no* when the answer is no. A check that only
ever confirms the paper is not a check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_attribution_scale", ROOT / "scripts/check_attribution_scale.py")
assert _spec is not None and _spec.loader is not None
scale = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scale)


def _counts(exact: float, pose: float, velocity: float, both: float, n: int = 1536) -> dict:
    return {
        scale.EXACT: (round(exact * n), n),
        scale.POSE: (round(pose * n), n),
        scale.VELOCITY: (round(velocity * n), n),
        scale.BOTH: (round(both * n), n),
    }


def test_purely_multiplicative_channels_report_no_interaction() -> None:
    """Odds that simply multiply must not be reported as an interaction.

    Constructed so the odds ratio is exactly 1: each channel divides the odds by
    a fixed factor, and both together divide by their product. On the
    probability scale this still looks like a large super-additive effect, which
    is precisely the artifact the check exists to catch.
    """

    def probability(odds: float) -> float:
        return odds / (1.0 + odds)

    base = 9.6667
    counts = _counts(
        probability(base),
        probability(base / 2.2),
        probability(base / 2.6),
        probability(base / (2.2 * 2.6)),
    )
    result = scale.interaction(counts, resamples=4000, seed=0)
    assert result["probability_scale"]["interaction_points"] < -5, "should look super-additive on probabilities"
    assert not result["survives_the_scale_change"]
    low, high = result["log_odds_scale"]["odds_ratio_95"]
    assert low < 1.0 < high


def test_the_measured_channels_survive_the_scale_change() -> None:
    """The published counts, which are the case the paper actually makes."""

    counts = {
        scale.EXACT: (1392, 1536),
        scale.POSE: (1264, 1536),
        scale.VELOCITY: (1236, 1537),
        scale.BOTH: (760, 1536),
    }
    result = scale.interaction(counts, resamples=4000, seed=0)
    assert result["survives_the_scale_change"]
    assert result["log_odds_scale"]["odds_ratio"] == pytest.approx(0.50, abs=0.05)
    assert result["log_odds_scale"]["odds_ratio_95"][1] < 1.0
