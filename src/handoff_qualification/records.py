"""Read a cohort archive into records this package can qualify.

Two archives describe one cohort and they join on the environment index:

* ``<cohort>.npz`` -- one row per episode, written at the moment of judgement.
  Its pose columns are terminal, so they may be used to describe an outcome and
  never to predict one. `_freeze` in `scripts/run_workflow_demo.py` is where
  they are taken.
* ``<cohort>_trace.npz`` -- optional, written only when the driver ran with
  ``--handoff_trace``. Its ``handoff`` block holds one row per phase change, so
  the row with ``to_phase == INSERT`` is the state the seating step was handed:
  information that existed *before* the contact interval, which is the only
  kind admissible as a predictor.

Most cohorts in this repository have the first and not the second, and that
asymmetry is the measurement gap this package exists to close: the cohorts whose
outcomes vary were run without traces, and the cohorts with traces sit well
inside the acceptable region where nothing varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Phase codes, mirroring `scripts/run_workflow_demo.py`.
CAPTURE, SEAT, EXTRACT, TRANSIT, INSERT, DONE = range(6)
PHASE_NAMES = ("capture", "seat", "extract", "transit", "insert", "done")

#: The terminal criterion the outcome is cut by, from
#: `src/zero_g_blade_swap/tasks/blade_swap/mdp/insertion.py`. Imported as a
#: default only; every entry point takes it as an argument, because qualifying
#: at a criterion the cohort was not run at is the point.
DEFAULT_LATERAL_CRITERION_M = 0.0025


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode, with its moments kept apart by construction."""

    cohort: str
    env: int
    #: Outcome, independently evaluated.
    success: bool
    #: Terminal residual. Judgement-time: describes, never predicts.
    residual_lateral_m: float
    residual_orientation_rad: float
    #: Did the episode reach the contact interval at all? An episode that did
    #: not is an absorbing failure and is kept, not dropped.
    reached_handoff: bool
    #: Pre-handoff state, when a trace exists. Empty otherwise.
    incoming: dict[str, float] = field(default_factory=dict)

    @property
    def arrived(self) -> bool:
        """Did the module physically arrive at the bay?

        Separates the near-miss regime from the catastrophic one. A residual of
        216 mm is not a tolerance failure; it is the bay separation, and it
        means the module never left the source. See `docs/NOW.md` on the gravity
        sweep, where exactly this reading was corrected once already.
        """

        return self.residual_lateral_m < CATASTROPHIC_RESIDUAL_M


#: Above this the module is not mis-seated, it is elsewhere. Set an order of
#: magnitude above the widest near-miss ever recorded here (5.73 mm) and far
#: below the smallest catastrophic one (216 mm), so the gap it sits in is two
#: orders wide and no observation has ever landed in it.
CATASTROPHIC_RESIDUAL_M = 0.05


def _fields(archive, key: str) -> list[str]:
    return [str(name) for name in archive[key]]


def load_cohort(
    outcome_path: Path | str,
    trace_path: Path | str | None = None,
    cohort: str | None = None,
) -> list[EpisodeRecord]:
    """Load one cohort. ``trace_path`` defaults to the sibling ``_trace.npz``."""

    outcome_path = Path(outcome_path)
    if trace_path is None:
        candidate = outcome_path.with_name(outcome_path.stem + "_trace.npz")
        trace_path = candidate if candidate.exists() else None

    archive = np.load(outcome_path, allow_pickle=True)
    names = _fields(archive, "fields")
    rows = archive["rows"].astype(float)

    def column(name: str) -> np.ndarray:
        return rows[:, names.index(name)]

    success = column("success") > 0.5
    lateral = column("lateral_error_m")
    orientation = column("orientation_error_rad")

    incoming_by_env: dict[int, dict[str, float]] = {}
    reached = np.zeros(len(rows), dtype=bool)
    if trace_path is not None and Path(trace_path).exists():
        trace = np.load(trace_path, allow_pickle=True)
        handoff_names = _fields(trace, "handoff_fields")
        handoff = trace["handoff"]
        if handoff.size:
            into_insert = handoff[handoff[:, handoff_names.index("to_phase")] == INSERT]
            skip = {"step", "env", "from_phase", "to_phase"}
            keep = [n for n in handoff_names if n not in skip]
            for row in into_insert:
                env = int(row[handoff_names.index("env")])
                if env >= len(rows):
                    continue
                reached[env] = True
                incoming_by_env[env] = {n: float(row[handoff_names.index(n)]) for n in keep}
    else:
        # Without a trace, reaching the contact interval can only be inferred
        # from the terminal phase, which is weaker and is labelled as such.
        reached = column("reached_phase") >= INSERT

    label = cohort or outcome_path.stem
    return [
        EpisodeRecord(
            cohort=label,
            env=env,
            success=bool(success[env]),
            residual_lateral_m=float(lateral[env]),
            residual_orientation_rad=float(orientation[env]),
            reached_handoff=bool(reached[env]),
            incoming=incoming_by_env.get(env, {}),
        )
        for env in range(len(rows))
    ]
