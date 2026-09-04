"""Unit tests for the paired A/B comparison.

The tool exists because reading a paired design with two independent Wilson
intervals throws away the pairing, so the cases that matter are the ones where
the two readings disagree: a lopsided flip that the unpaired intervals call
inconclusive, and a large but balanced flip that no reading should call a win.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_paired_arms", ROOT / "scripts/compare_paired_arms.py")
assert _spec is not None and _spec.loader is not None
paired = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paired)


def _arms(baseline: list[int], treatment: list[int]):
    return np.array(baseline, dtype=bool), np.array(treatment, dtype=bool)


def test_five_for_nothing_matches_the_hand_computation() -> None:
    """The rack-retention case: five gained, none lost, one-sided p = 2**-5."""

    result = paired.mcnemar_exact(gained=5, lost=0)
    assert result["discordant"] == 5
    assert result["one_sided_p"] == pytest.approx(0.03125)
    assert result["two_sided_p"] == pytest.approx(0.0625)


def test_a_balanced_flip_is_not_a_result() -> None:
    """Ten changed episodes, five each way, is exactly no evidence."""

    result = paired.mcnemar_exact(gained=5, lost=5)
    assert result["two_sided_p"] == pytest.approx(1.0)


def test_no_discordant_pairs_yields_no_test() -> None:
    result = paired.mcnemar_exact(gained=0, lost=0)
    assert result["one_sided_p"] is None


def test_the_paired_reading_can_beat_the_unpaired_one() -> None:
    """The case the tool was written for.

    Twelve episodes: the treatment fixes five failures and breaks none. The
    unpaired intervals overlap heavily; the paired test does not.
    """

    baseline, treatment = _arms(
        [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    )
    result = paired.compare(baseline, treatment)
    assert result["paired"]["gained"] == 5
    assert result["paired"]["lost"] == 0
    assert result["wilson_intervals_overlap"]
    assert result["paired"]["one_sided_p"] < 0.05


def test_arms_of_different_length_are_refused() -> None:
    baseline, treatment = _arms([1, 0, 1], [1, 0])
    with pytest.raises(SystemExit, match="cannot be paired"):
        paired.compare(baseline, treatment)
