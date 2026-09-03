"""The p95-to-sigma inversion has to be right, because nothing downstream checks it.

A training run noised from a wrong sigma still trains, still certifies, and still
publishes a number; the error would only ever show up as a policy that does not
transfer, months later, with no way to attribute it. So the derivation is checked
two ways here: against the closed-form chi-3 distribution, and against a Monte
Carlo draw of the quantity the certificate actually measures -- the norm of a
three-component residual.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from zero_g_blade_swap.estimator_noise import (
    CERTIFIED_QUANTILE,
    EstimatorNoiseModel,
    chi3_cdf,
    chi3_quantile,
    sigma_from_norm_quantile,
)


def test_chi3_cdf_is_a_distribution_function() -> None:
    assert chi3_cdf(0.0) == 0.0
    assert chi3_cdf(-1.0) == 0.0
    assert chi3_cdf(50.0) == pytest.approx(1.0, abs=1.0e-12)
    previous = 0.0
    for step in range(1, 200):
        value = chi3_cdf(step * 0.05)
        assert value >= previous
        previous = value


def test_chi3_quantile_inverts_the_cdf() -> None:
    for probability in (0.05, 0.5, 0.95, 0.99):
        assert chi3_cdf(chi3_quantile(probability)) == pytest.approx(probability, abs=1.0e-9)


def test_the_p95_chi3_quantile_matches_the_published_value() -> None:
    # Tabulated: the 95th percentile of a chi variate with three degrees of
    # freedom is 2.7955. If this constant ever moves, every sigma derived from a
    # certificate moves with it, so it is asserted rather than assumed.
    assert chi3_quantile(0.95) == pytest.approx(2.7955, abs=5.0e-4)


def test_sigma_reproduces_the_certified_quantile_by_monte_carlo() -> None:
    """Draw the residual the certificate measures and recover its p95."""

    certified_p95_mm = 1.9098
    sigma = sigma_from_norm_quantile(certified_p95_mm)
    generator = random.Random(70)
    norms = []
    for _ in range(200_000):
        x = generator.gauss(0.0, sigma)
        y = generator.gauss(0.0, sigma)
        z = generator.gauss(0.0, sigma)
        norms.append(math.sqrt(x * x + y * y + z * z))
    norms.sort()
    realized_p95 = norms[int(CERTIFIED_QUANTILE * len(norms))]
    assert realized_p95 == pytest.approx(certified_p95_mm, rel=0.01)


def test_sigma_rejects_a_negative_quantile() -> None:
    with pytest.raises(ValueError):
        sigma_from_norm_quantile(-1.0)


def test_the_model_reads_a_real_certification(tmp_path) -> None:
    report = tmp_path / "certification.json"
    report.write_text(
        json.dumps(
            {
                "detection_rate": 0.9189453125,
                "position_error_mm": {"p95": 1.9098028824951063},
                "orientation_error_rad": {"p95": 0.02005764650042329},
            }
        ),
        encoding="utf-8",
    )
    model = EstimatorNoiseModel.from_certification(report)
    assert model.position_sigma_m == pytest.approx(1.9098028824951063 / 2.7955 / 1000.0, rel=1.0e-3)
    assert model.orientation_sigma_rad == pytest.approx(0.02005764650042329 / 2.7955, rel=1.0e-3)
    described = model.describe()
    assert described["certified_position_p95_mm"] == pytest.approx(1.9098028824951063)
    assert "chi3" in str(described["derivation"])


def test_the_model_refuses_a_report_that_is_not_a_perception_certificate(tmp_path) -> None:
    report = tmp_path / "not_a_certification.json"
    report.write_text(json.dumps({"overall": {"success_rate": 0.9167}}), encoding="utf-8")
    with pytest.raises(ValueError):
        EstimatorNoiseModel.from_certification(report)


def test_the_model_refuses_an_impossible_detection_rate(tmp_path) -> None:
    report = tmp_path / "impossible.json"
    report.write_text(
        json.dumps(
            {
                "detection_rate": 1.5,
                "position_error_mm": {"p95": 1.9},
                "orientation_error_rad": {"p95": 0.02},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        EstimatorNoiseModel.from_certification(report)


def test_the_shipped_certification_derives_a_usable_model() -> None:
    """The certificate the training tasks are noised from must stay readable."""

    model = EstimatorNoiseModel.from_certification("evidence/fiducial_rgbd_flush_v4_seed285_gripper_clear.json")
    # Sub-millimetre per axis, and a few milliradians. Quoted as bounds rather
    # than as equalities so a re-certified estimator does not fail this test for
    # improving; a model outside these is a different estimator and should.
    assert 0.0 < model.position_sigma_m < 0.005
    assert 0.0 < model.orientation_sigma_rad < 0.050
    assert 0.85 < model.detection_rate <= 1.0
