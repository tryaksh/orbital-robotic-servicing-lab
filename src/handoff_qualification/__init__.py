"""Qualify a contact-rich handoff by the residual it leaves, not the rate it passes.

The premise this package is built on was measured, not assumed. In 192 episodes
of the certified relocation chain, the module jammed in the bay **zero** times
and every episode reached the final phase. What separated the 110 successes from
the 82 failures was one continuous quantity -- the lateral error the module came
to rest at -- cut by one constant, `INSERTION_LATERAL_TOLERANCE_M = 0.0025`.
Seventy-seven of the 82 failures landed between 2.51 mm and 5.73 mm; the
successes ran up to 2.451 mm. The distribution is continuous across the line.

That changes what a "handoff gate" can usefully be. A binary success rate throws
away where in the band an episode landed, and where it landed is the entire
signal. So this package estimates the *residual distribution* and reports the
pass rate as an integral of it, which buys three things a counted rate cannot:

1. the rate at a criterion the cohort was not run at, so a tolerance decision
   does not need a new campaign;
2. a variance reduction at fixed episode count, when the parametric form holds
   -- tested, not assumed, by `scripts/compare_residual_estimators.py`;
3. an explicit separation of the near-miss regime from the catastrophic one.
   Five of those 82 failures were at 216 mm to 1213 mm: the module never arrived.
   Pooling them into one rate hides two different engineering problems.

It refuses to answer when the evidence does not support one. See
`ResidualQualification.insufficient_evidence`.

Nothing here imports a simulator. It reads episode archives and returns numbers.
"""

from handoff_qualification.records import EpisodeRecord, load_cohort
from handoff_qualification.residual import (
    ResidualModel,
    ResidualQualification,
    fit_residual_model,
    qualify,
)

__all__ = [
    "EpisodeRecord",
    "load_cohort",
    "ResidualModel",
    "ResidualQualification",
    "fit_residual_model",
    "qualify",
]

from handoff_qualification.change_prediction import (  # noqa: E402
    Cohort,
    Prediction,
    predict,
    sensitivity_corpus,
)

__all__ += ["Cohort", "Prediction", "predict", "sensitivity_corpus"]
