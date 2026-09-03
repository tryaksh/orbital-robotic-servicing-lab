"""The deployed estimator's error, as a distribution a training task can sample.

The cost of perception in this project is measured, not guessed: the same chain,
on the same task, with the same seeds, checkpoints, guard and observation terms,
scores 83.33% when the module pose comes from the simulator and 16.67% when it
comes from the cameras. Only one term is substituted between those two arms, so
the 67-point step is the estimator. It is also not a perception defect -- the
estimator's own error on healthy episodes is about 2 mm, and it is certified on
held-out frames against unchanged gates.

What it is, is the expected price of an untrained transfer. Capture, extraction
and the guard were trained on simulator state and are deployed against an
estimator whose error was never in their training distribution. The literature's
fix is to put it there rather than to improve the estimator ("From Imitation to
Refinement: Residual RL for Precise Assembly", arXiv 2407.16677, reports a
state-privileged teacher at 98% distilling to a vision student at 73%).

Rendering during skill training is not affordable -- the skills need thousands
of epochs and the vision task runs at a fraction of the state task's throughput
-- so the training task samples the estimator's *statistics* instead of its
images. This module holds the part of that model which needs no simulator, so it
can be derived, checked and tested on the CPU:

* the certified error is the **norm** of a three-component residual, both for
  position and for the rotation vector (`certify_fiducial_perception.py` takes
  `np.linalg.norm` of each). For an isotropic zero-mean Gaussian residual, that
  norm is ``sigma`` times a chi variate with three degrees of freedom, so a
  certified p95 inverts to a per-axis sigma exactly rather than by choice;
* the inversion is done here, from the chi-3 quantile, and
  ``describe`` records both the certified input and the derived sigma so a
  report can state which certificate a training run was noised from.

Nothing here tunes a number to make anything pass. The certified p95 is an
input; sigma is its consequence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# The certification quotes p95, so this is the quantile that has to be inverted.
# It is a named constant because it appears in the derivation and in the test
# that checks the derivation, and those two must not drift apart.
CERTIFIED_QUANTILE = 0.95


def chi3_cdf(x: float) -> float:
    """Return P(X <= x) for a chi variate with three degrees of freedom.

    Closed form, so the inversion below needs no SciPy: the regularized lower
    incomplete gamma P(3/2, x^2/2) reduces to
    ``erf(x/sqrt(2)) - sqrt(2/pi) * x * exp(-x^2/2)``.
    """

    if x <= 0.0:
        return 0.0
    return math.erf(x / math.sqrt(2.0)) - math.sqrt(2.0 / math.pi) * x * math.exp(-0.5 * x * x)


def chi3_quantile(probability: float, tolerance: float = 1.0e-12) -> float:
    """Invert :func:`chi3_cdf` by bisection on a bracket that always contains it."""

    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1); got {probability}")
    low, high = 0.0, 1.0
    while chi3_cdf(high) < probability:
        high *= 2.0
        if high > 1.0e6:  # unreachable for any probability < 1, kept as a guard
            raise RuntimeError("chi3 quantile failed to bracket the requested probability")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if chi3_cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def sigma_from_norm_quantile(quantile_value: float, probability: float = CERTIFIED_QUANTILE) -> float:
    """Return the per-axis sigma whose three-component norm has this quantile.

    This is the whole derivation. A certificate reports the p95 of a norm; a
    sampler needs the sigma of one axis; for an isotropic Gaussian residual the
    two differ by exactly the chi-3 quantile and by nothing that is chosen.
    """

    if quantile_value < 0.0:
        raise ValueError(f"a quantile of an error norm cannot be negative; got {quantile_value}")
    return quantile_value / chi3_quantile(probability)


@dataclass(frozen=True)
class EstimatorNoiseModel:
    """What one perception certificate says a training task should sample.

    ``detection_rate`` is carried because a missed frame is not a large error --
    the deployed estimator fails closed and holds its last pose, which is a
    *stale* observation rather than a noisy one, and staleness is the part of the
    estimator a finite-difference velocity channel is least able to absorb.
    """

    position_p95_mm: float
    orientation_p95_rad: float
    detection_rate: float
    source: str

    @property
    def position_sigma_m(self) -> float:
        return sigma_from_norm_quantile(self.position_p95_mm) / 1000.0

    @property
    def orientation_sigma_rad(self) -> float:
        return sigma_from_norm_quantile(self.orientation_p95_rad)

    @classmethod
    def from_certification(cls, path: str | Path) -> EstimatorNoiseModel:
        """Read a `certify_fiducial_perception.py` report, and nothing else.

        Reading the certificate rather than restating its numbers is deliberate:
        it means a training run cannot be noised from figures that no longer
        match any evidence file, and the report can name the certificate.
        """

        report = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            position = float(report["position_error_mm"]["p95"])
            orientation = float(report["orientation_error_rad"]["p95"])
            detection = float(report["detection_rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} is not a fiducial perception certification: {error}") from error
        if not 0.0 < detection <= 1.0:
            raise ValueError(f"{path} reports a detection rate outside (0, 1]: {detection}")
        return cls(
            position_p95_mm=position,
            orientation_p95_rad=orientation,
            detection_rate=detection,
            source=str(path),
        )

    def describe(self) -> dict[str, float | str]:
        """Return the certified inputs beside the derived sigmas, for a report."""

        return {
            "source_certification": self.source,
            "certified_position_p95_mm": self.position_p95_mm,
            "certified_orientation_p95_rad": self.orientation_p95_rad,
            "certified_detection_rate": self.detection_rate,
            "derived_position_sigma_m": self.position_sigma_m,
            "derived_orientation_sigma_rad": self.orientation_sigma_rad,
            "derivation": (
                "per-axis sigma = certified p95 of the error norm / chi3 quantile at 0.95 "
                f"({chi3_quantile(CERTIFIED_QUANTILE):.6f}); the certificate takes the norm of a "
                "three-component residual, so the inversion is exact for an isotropic residual"
            ),
        }


__all__ = [
    "CERTIFIED_QUANTILE",
    "EstimatorNoiseModel",
    "chi3_cdf",
    "chi3_quantile",
    "sigma_from_norm_quantile",
]
