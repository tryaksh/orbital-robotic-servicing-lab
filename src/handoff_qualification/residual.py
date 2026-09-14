"""Estimate a pass rate by modelling the residual, and say when you cannot.

The counted rate and the modelled rate answer the same question at the
criterion the cohort was run at, and only the modelled one answers it anywhere
else. That is the whole trade: a parametric form buys extrapolation and some
variance, and costs bias when the form is wrong. Both terms are reported.

The form is lognormal on the arrived population. It is not chosen for elegance
-- a residual is a non-negative quantity produced by a product of small
misalignments, which is the situation lognormality describes -- but it is
checked rather than assumed, by ``ResidualModel.goodness``, and ``qualify``
refuses when the criterion sits outside what the cohort actually sampled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from handoff_qualification.records import CATASTROPHIC_RESIDUAL_M, EpisodeRecord

#: Below this many arrived episodes, a two-parameter fit is not evidence.
MINIMUM_ARRIVED = 12
#: A criterion this far outside the sampled range is extrapolation, not
#: interpolation, and is refused. Expressed as a fraction of the sampled
#: log-range, so it scales with how wide the cohort actually was.
MAXIMUM_EXTRAPOLATION = 0.25


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilson(successes: int, trials: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval, matching ``zero_g_blade_swap.evaluation``."""

    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class ResidualModel:
    """A lognormal on the arrived residuals, plus a catastrophic mass."""

    log_mean: float
    log_sd: float
    arrived: int
    catastrophic: int
    sample: np.ndarray = field(repr=False)

    @property
    def episodes(self) -> int:
        return self.arrived + self.catastrophic

    @property
    def catastrophic_rate(self) -> float:
        return self.catastrophic / self.episodes if self.episodes else 0.0

    def pass_rate(self, criterion_m: float) -> float:
        """P(residual < criterion). Catastrophic episodes never pass."""

        if criterion_m <= 0 or self.arrived == 0:
            return 0.0
        arrived_share = self.arrived / self.episodes
        z = (math.log(criterion_m) - self.log_mean) / self.log_sd
        return arrived_share * _normal_cdf(z)

    def goodness(self) -> float:
        """Kolmogorov-Smirnov distance between the fit and the sample.

        Reported rather than turned into a p-value: at 192 episodes a KS test
        rejects a form that is nonetheless a better summary than a raw count,
        and the decision this feeds is an engineering one about extrapolation
        distance, not a hypothesis test.
        """

        if self.arrived < 2:
            return 1.0
        ordered = np.sort(self.sample)
        model = np.array(
            [_normal_cdf((math.log(x) - self.log_mean) / self.log_sd) for x in ordered]
        )
        empirical_high = np.arange(1, len(ordered) + 1) / len(ordered)
        empirical_low = np.arange(0, len(ordered)) / len(ordered)
        return float(
            max(
                np.abs(model - empirical_high).max(),
                np.abs(model - empirical_low).max(),
            )
        )


@dataclass(frozen=True)
class ResidualQualification:
    """What the tool returns. Reads as a refusal when it is one."""

    criterion_m: float
    counted_rate: float
    counted_interval: tuple[float, float]
    modelled_rate: float | None
    modelled_interval: tuple[float, float] | None
    episodes: int
    arrived: int
    catastrophic: int
    fit_distance: float | None
    insufficient_evidence: str | None
    tested_range_m: tuple[float, float] | None

    @property
    def supported(self) -> bool:
        return self.insufficient_evidence is None

    def summary(self) -> str:
        head = (
            f"{self.episodes} episodes, {self.arrived} arrived, "
            f"{self.catastrophic} never arrived; criterion {self.criterion_m * 1000:.3f} mm"
        )
        counted = (
            f"  counted   {self.counted_rate:.4f}  "
            f"[{self.counted_interval[0]:.4f}, {self.counted_interval[1]:.4f}]"
        )
        if not self.supported:
            return f"{head}\n{counted}\n  modelled  refused: {self.insufficient_evidence}"
        assert self.modelled_rate is not None and self.modelled_interval is not None
        modelled = (
            f"  modelled  {self.modelled_rate:.4f}  "
            f"[{self.modelled_interval[0]:.4f}, {self.modelled_interval[1]:.4f}]"
            f"  (KS {self.fit_distance:.3f})"
        )
        return f"{head}\n{counted}\n{modelled}"


def fit_residual_model(residuals: Sequence[float]) -> ResidualModel:
    values = np.asarray(list(residuals), dtype=float)
    arrived = values[(values > 0) & (values < CATASTROPHIC_RESIDUAL_M)]
    catastrophic = int((values >= CATASTROPHIC_RESIDUAL_M).sum())
    if len(arrived) < 2:
        return ResidualModel(0.0, 1.0, len(arrived), catastrophic, arrived)
    logs = np.log(arrived)
    return ResidualModel(
        log_mean=float(logs.mean()),
        log_sd=float(logs.std(ddof=1)) or 1e-9,
        arrived=len(arrived),
        catastrophic=catastrophic,
        sample=arrived,
    )


def qualify(
    records: Sequence[EpisodeRecord],
    criterion_m: float,
    bootstrap: int = 2000,
    seed: int = 0,
) -> ResidualQualification:
    """Qualify a cohort at a criterion, with an explicit refusal path."""

    residuals = [r.residual_lateral_m for r in records]
    successes = sum(1 for r in records if r.residual_lateral_m < criterion_m)
    episodes = len(records)
    counted = successes / episodes if episodes else 0.0
    counted_interval = wilson(successes, episodes)

    model = fit_residual_model(residuals)
    tested = (
        (float(model.sample.min()), float(model.sample.max())) if model.arrived else None
    )

    refusal: str | None = None
    if model.arrived < MINIMUM_ARRIVED:
        refusal = (
            f"only {model.arrived} episodes arrived; a two-parameter residual fit "
            f"needs at least {MINIMUM_ARRIVED}"
        )
    elif tested is not None:
        span = math.log(tested[1]) - math.log(tested[0])
        if span <= 0:
            refusal = "every arrived residual is identical; there is no distribution to fit"
        else:
            below = (math.log(tested[0]) - math.log(criterion_m)) / span
            above = (math.log(criterion_m) - math.log(tested[1])) / span
            reach = max(below, above)
            if reach > MAXIMUM_EXTRAPOLATION:
                where = "below" if below > above else "above"
                refusal = (
                    f"criterion is {reach:.0%} of the sampled log-range {where} anything "
                    f"observed ({tested[0] * 1000:.3f}-{tested[1] * 1000:.3f} mm); "
                    f"that is extrapolation, not qualification"
                )

    modelled = modelled_interval = None
    if refusal is None:
        modelled = model.pass_rate(criterion_m)
        rng = np.random.default_rng(seed)
        values = np.asarray(residuals, dtype=float)
        draws = np.empty(bootstrap)
        for index in range(bootstrap):
            resample = rng.choice(values, size=len(values), replace=True)
            draws[index] = fit_residual_model(resample).pass_rate(criterion_m)
        modelled_interval = (
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        )

    return ResidualQualification(
        criterion_m=criterion_m,
        counted_rate=counted,
        counted_interval=counted_interval,
        modelled_rate=modelled,
        modelled_interval=modelled_interval,
        episodes=episodes,
        arrived=model.arrived,
        catastrophic=model.catastrophic,
        fit_distance=model.goodness() if model.arrived >= 2 else None,
        insufficient_evidence=refusal,
        tested_range_m=tested,
    )


def empirical_pass_rate(
    records: Sequence[EpisodeRecord],
    criterion_m: float,
) -> tuple[float, tuple[float, float]]:
    """The rate at any criterion, counted rather than modelled.

    At the criterion a cohort was run at this is identically the reported
    success rate -- verified on the nominal cohort, 192 of 192 episodes agree --
    so it buys no variance there and none is claimed. What it buys is every
    *other* criterion from the same episodes, which a counted rate cannot give
    at any sample size.

    This is the estimator to use. The lognormal in ``ResidualModel`` was fitted
    to the same cohort and returned 0.650 against a true 0.573 with a bootstrap
    interval that excluded the truth: the parametric form is wrong here, and the
    honest tool reports the empirical curve and keeps the fit for the one job
    it is needed for, which is reaching past the sampled range.
    """

    residuals = np.asarray([r.residual_lateral_m for r in records], dtype=float)
    successes = int((residuals < criterion_m).sum())
    return successes / len(residuals), wilson(successes, len(residuals))


def criterion_curve(
    records: Sequence[EpisodeRecord],
    criteria_m: Sequence[float],
) -> list[tuple[float, float, tuple[float, float]]]:
    """The pass rate across a range of criteria, from one cohort."""

    return [(c, *empirical_pass_rate(records, c)) for c in criteria_m]
