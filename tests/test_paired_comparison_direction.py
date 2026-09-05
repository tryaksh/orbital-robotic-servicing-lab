"""The paired test must know which arm won.

`compare_paired_arms.mcnemar_exact` reported a field called `one_sided_p` that
was computed from the *smaller* discordant tail, so it carried no direction at
all: fourteen gained and none lost, and none gained and fourteen lost, both
returned 6.1e-05. One published report ran the wrong way under it --
`evidence/rack_prescription_paired_n192.json`, 28 gained against 74 lost,
printed with one_sided_p = 2.95e-06 -- and a reader skimming for a small p
would have read a decisive refutation as a decisive confirmation.

These tests are the brief's required regression: an improvement alternative
must give a small p only when the treatment actually improved.
"""

from __future__ import annotations

import math

from scripts.compare_paired_arms import mcnemar_exact


def test_direction_is_not_symmetric():
    won = mcnemar_exact(gained=14, lost=0)
    lost = mcnemar_exact(gained=0, lost=14)
    assert won["improvement_p"] < 1e-4
    assert lost["improvement_p"] == 1.0
    assert lost["deterioration_p"] < 1e-4
    assert won["deterioration_p"] == 1.0
    # The defect: these two were once equal.
    assert won["improvement_p"] != lost["improvement_p"]


def test_two_sided_is_symmetric_and_unchanged():
    won = mcnemar_exact(gained=14, lost=0)
    lost = mcnemar_exact(gained=0, lost=14)
    assert won["two_sided_p"] == lost["two_sided_p"]
    assert math.isclose(won["two_sided_p"], 2 * 2**-14)


def test_the_rack_prescription_is_a_loss():
    """The published counts, read the right way round."""

    result = mcnemar_exact(gained=28, lost=74)
    assert result["direction"] == "lost"
    assert result["improvement_p"] > 0.999
    assert result["deterioration_p"] < 1e-5
    # Two-sided still says the arms differ, which was never in doubt.
    assert result["two_sided_p"] < 1e-5


def test_rack_retention_five_for_nothing():
    result = mcnemar_exact(gained=5, lost=0)
    assert result["direction"] == "gained"
    assert math.isclose(result["improvement_p"], 2**-5)
    assert math.isclose(result["two_sided_p"], 2 * 2**-5)


def test_no_discordant_pairs():
    result = mcnemar_exact(gained=0, lost=0)
    assert result["discordant"] == 0
    assert result["improvement_p"] is None
    assert result["two_sided_p"] is None


def test_tied_discordant_pairs_are_not_a_direction():
    result = mcnemar_exact(gained=7, lost=7)
    assert result["direction"] == "tied"
    assert result["two_sided_p"] == 1.0
