"""Reset-safe terminal episode metrics and held-out evaluation statistics.

Isaac Lab resets a terminated environment *inside* ``ManagerBasedRLEnv.step``,
before control returns to the caller.  Reading blade pose or velocity error
after ``step`` therefore reports the next episode's starting state for exactly
the environments that just finished, which silently corrupts terminal-error
evidence.  :class:`TerminalMetricsMixin` captures every finished episode before
that reset runs, and the functions below turn the captured rows into a report.

Nothing here imports Isaac Sim, so the ordering guarantee and the statistics
are unit-testable on an ordinary Python installation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

# One row per completed episode, in this exact column order.
TERMINAL_METRIC_FIELDS = (
    "success",
    "termination_reason",
    "curriculum_stage",
    "control_steps",
    "cycle_time_s",
    "axial_error_m",
    "lateral_error_m",
    "orientation_error_rad",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
    "tool_to_handle_error_m",
    "tool_to_handle_orientation_rad",
)

# Columns a task may add on top of the core row.  Recording them is optional,
# so reports must align runs by column *name* rather than by position.
BLADE_MASS_FIELD = "blade_mass_kg"

# Highest priority first.  A physics blow-up or a mount failure in the same
# control step as a geometric success must never be reported as a success.
TERMINATION_PRIORITY = (
    "non_finite",
    "mount_unstable",
    "insertion_failed",
    "insertion_success",
    "time_out",
)
UNCATEGORIZED_TERMINATION = "uncategorized"
TERMINATION_REASONS = (*TERMINATION_PRIORITY, UNCATEGORIZED_TERMINATION)
INSTABILITY_TERMINATIONS = ("non_finite", "mount_unstable")
SUCCESS_TERMINATION = "insertion_success"

# Distributions worth reporting; ``success`` and ``termination_reason`` are
# categorical and are summarized as counts instead.
DEFAULT_METRIC_FIELDS = (
    "axial_error_m",
    "lateral_error_m",
    "orientation_error_rad",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
    "tool_to_handle_error_m",
    "tool_to_handle_orientation_rad",
    "control_steps",
    "cycle_time_s",
)

CATEGORICAL_FIELDS = ("success", "termination_reason", "curriculum_stage")
NORMAL_95_PERCENT_Z = 1.959963984540054
BUCKET_LABELS = ("low", "mid", "high")


def _to_numpy(value: Any) -> NDArray[np.float64]:
    """Convert a torch tensor, sequence, or array to a float64 NumPy array."""

    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    for attribute in ("detach", "cpu"):
        method = getattr(value, attribute, None)
        if callable(method):
            value = method()
    try:
        return np.asarray(value, dtype=np.float64)
    except (RuntimeError, TypeError, ValueError):
        # ``torch.inference_mode`` tensors refuse some zero-copy conversions.
        return np.asarray(value.tolist(), dtype=np.float64)


class TerminalMetricsMixin:
    """Snapshot finished episodes before Isaac Lab's automatic reset runs.

    Mix this in *before* ``ManagerBasedRLEnv`` so that ``_reset_idx`` is
    intercepted while the scene still holds the terminal physics state.  The
    hook is ``None`` unless a caller enables it, so training keeps its original
    behaviour, observation/action dimensions, rewards, and checkpoints.
    """

    _terminal_metrics_hook = None

    def enable_terminal_metrics(self, hook) -> None:
        """Call ``hook(env, env_ids)`` for finished episodes before each reset."""

        if not callable(hook):
            raise TypeError("terminal metrics hook must be callable")
        self._terminal_metrics_hook = hook

    def disable_terminal_metrics(self) -> None:
        self._terminal_metrics_hook = None

    def _finished_episode_ids(self, env_ids):
        """Drop environments that never stepped, such as the initial reset."""

        return env_ids[self.episode_length_buf[env_ids] > 0]

    def _reset_idx(self, env_ids):
        hook = self._terminal_metrics_hook
        if hook is not None:
            finished = self._finished_episode_ids(env_ids)
            if len(finished) > 0:
                hook(self, finished)
        super()._reset_idx(env_ids)


class TerminalEpisodeRecorder:
    """Append-only store of one metric row per completed episode."""

    def __init__(self, fields: Sequence[str] = TERMINAL_METRIC_FIELDS) -> None:
        self._fields = tuple(fields)
        if not self._fields:
            raise ValueError("at least one metric field is required")
        self._blocks: list[NDArray[np.float64]] = []
        self._rows: NDArray[np.float64] | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        return self._fields

    def record(self, rows: Any) -> int:
        """Store a ``(episodes, len(fields))`` block and return its row count."""

        block = _to_numpy(rows)
        if block.ndim != 2 or block.shape[1] != len(self._fields):
            raise ValueError(f"expected a (n, {len(self._fields)}) block, got shape {block.shape}")
        if block.shape[0] > 0:
            self._blocks.append(block)
            self._rows = None
        return int(block.shape[0])

    @property
    def rows(self) -> NDArray[np.float64]:
        if self._rows is None:
            self._rows = (
                np.concatenate(self._blocks, axis=0)
                if self._blocks
                else np.empty((0, len(self._fields)), dtype=np.float64)
            )
        return self._rows

    def column(self, name: str) -> NDArray[np.float64]:
        return self.rows[:, self._fields.index(name)]

    def __len__(self) -> int:
        return int(self.rows.shape[0])


def wilson_interval(
    successes: int,
    trials: int,
    z: float = NORMAL_95_PERCENT_Z,
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial success rate.

    The Wilson interval is used instead of the normal approximation because a
    100% held-out result has zero observed variance; the normal interval would
    collapse to a point and overstate the evidence.
    """

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    if z <= 0.0:
        raise ValueError("z must be positive")
    if trials == 0:
        return (0.0, 1.0)
    rate = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (rate + z_squared / (2.0 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(rate * (1.0 - rate) / trials + z_squared / (4.0 * trials * trials))
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_distribution(values: Any) -> dict[str, Any]:
    """Return count, mean, p50, p95, max, min, and the non-finite count."""

    sample = _to_numpy(values).reshape(-1)
    finite = sample[np.isfinite(sample)]
    non_finite = int(sample.size - finite.size)
    if finite.size == 0:
        return {"count": int(sample.size), "non_finite": non_finite}
    return {
        "count": int(sample.size),
        "non_finite": non_finite,
        "mean": float(finite.mean()),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(finite.max()),
        "min": float(finite.min()),
    }


def group_rows(
    rows: Any,
    name: str,
    fields: Sequence[str] = TERMINAL_METRIC_FIELDS,
) -> dict[int, NDArray[np.float64]]:
    """Split recorded rows by the integer value of one categorical column."""

    table = _to_numpy(rows)
    if table.ndim != 2 or table.shape[1] != len(fields):
        raise ValueError(f"expected a (n, {len(fields)}) table, got shape {table.shape}")
    column = table[:, tuple(fields).index(name)]
    return {int(value): table[column == value] for value in sorted(np.unique(column))}


def summarize_terminal_episodes(
    rows: Any,
    fields: Sequence[str] = TERMINAL_METRIC_FIELDS,
    metric_fields: Sequence[str] = DEFAULT_METRIC_FIELDS,
    include_successful_metrics: bool = True,
) -> dict[str, Any]:
    """Aggregate recorded episodes into success, failure, and error statistics."""

    names = tuple(fields)
    table = _to_numpy(rows)
    if table.ndim != 2 or table.shape[1] != len(names):
        raise ValueError(f"expected a (n, {len(names)}) table, got shape {table.shape}")

    success = table[:, names.index("success")] > 0.5
    reason_ids = table[:, names.index("termination_reason")]
    episodes = int(table.shape[0])
    successes = int(success.sum())
    low, high = wilson_interval(successes, episodes)

    reasons: dict[str, int] = {}
    for index, reason in enumerate(TERMINATION_REASONS):
        count = int((reason_ids == index).sum())
        if count:
            reasons[reason] = count
    instability = sum(reasons.get(name, 0) for name in INSTABILITY_TERMINATIONS)

    metric_columns = [name for name in metric_fields if name in names]
    # Check finiteness over physical and categorical columns only.  Optional
    # columns are NaN for runs that did not record them, which is missing data
    # rather than a physics instability.
    checked = [names.index(name) for name in (*metric_columns, *CATEGORICAL_FIELDS) if name in names]
    report: dict[str, Any] = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "success_rate_wilson_95": {"low": low, "high": high},
        "termination_reasons": reasons,
        "instability_terminations": instability,
        "non_finite_metric_episodes": int((~np.isfinite(table[:, checked])).any(axis=1).sum()),
        "terminal_metrics": {name: summarize_distribution(table[:, names.index(name)]) for name in metric_columns},
    }
    if include_successful_metrics:
        successful = table[success]
        report["successful_episode_metrics"] = {
            name: summarize_distribution(successful[:, names.index(name)]) for name in metric_columns
        }
    return report


def align_rows(rows: Any, source_fields: Sequence[str], target_fields: Sequence[str]) -> NDArray[np.float64]:
    """Reorder a run's columns into ``target_fields``, filling gaps with NaN.

    Runs recorded at different robustness levels carry different optional
    columns.  Aligning by name lets one report pool them without silently
    mixing, for example, blade mass into the cycle-time column.
    """

    table = _to_numpy(rows)
    source = tuple(source_fields)
    if table.ndim != 2 or table.shape[1] != len(source):
        raise ValueError(f"expected a (n, {len(source)}) table, got shape {table.shape}")
    aligned = np.full((table.shape[0], len(target_fields)), np.nan, dtype=np.float64)
    for index, name in enumerate(target_fields):
        if name in source:
            aligned[:, index] = table[:, source.index(name)]
    return aligned


def bucket_success_rates(
    rows: Any,
    name: str,
    low: float,
    high: float,
    fields: Sequence[str] = TERMINAL_METRIC_FIELDS,
    labels: Sequence[str] = BUCKET_LABELS,
) -> dict[str, Any]:
    """Split success by equal bands of a continuous randomized parameter."""

    names = tuple(fields)
    table = _to_numpy(rows)
    if high <= low:
        raise ValueError("high must exceed low")
    values = table[:, names.index(name)]
    success = table[:, names.index("success")] > 0.5
    observed = np.isfinite(values)
    # Substitute a placeholder for unrecorded values so the cast stays valid;
    # ``observed`` excludes them from every bucket below.
    normalized = np.clip((np.where(observed, values, low) - low) / (high - low), 0.0, 1.0 - 1.0e-9)
    indices = np.floor(len(labels) * normalized).astype(int)

    report: dict[str, Any] = {"range": [low, high], "episodes_without_value": int((~observed).sum())}
    rates: list[float] = []
    for index, label in enumerate(labels):
        selected = observed & (indices == index)
        episodes = int(selected.sum())
        successes = int((success & selected).sum())
        rate = successes / episodes if episodes else None
        interval_low, interval_high = wilson_interval(successes, episodes)
        report[label] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": rate,
            "success_rate_wilson_95": {"low": interval_low, "high": interval_high},
        }
        if rate is not None:
            rates.append(rate)
    report["minimum_observed_success_rate"] = min(rates) if rates else None
    report["observed_value_range"] = (
        [float(values[observed].min()), float(values[observed].max())] if observed.any() else None
    )
    return report


def concatenate_rows(blocks: Iterable[Any], fields: Sequence[str] = TERMINAL_METRIC_FIELDS) -> NDArray[np.float64]:
    """Join per-run episode tables into one table for pooled statistics."""

    width = len(fields)
    tables = []
    for block in blocks:
        table = _to_numpy(block)
        if table.ndim != 2 or table.shape[1] != width:
            raise ValueError(f"expected a (n, {width}) table, got shape {table.shape}")
        tables.append(table)
    return np.concatenate(tables, axis=0) if tables else np.empty((0, width), dtype=np.float64)


def round_floats(value: Any, digits: int = 6) -> Any:
    """Recursively round floats so reports stay small and readable."""

    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, digits) for item in value]
    if isinstance(value, float):
        return value if not math.isfinite(value) else round(value, digits)
    return value


__all__ = [
    "BLADE_MASS_FIELD",
    "BUCKET_LABELS",
    "CATEGORICAL_FIELDS",
    "DEFAULT_METRIC_FIELDS",
    "INSTABILITY_TERMINATIONS",
    "NORMAL_95_PERCENT_Z",
    "SUCCESS_TERMINATION",
    "TERMINAL_METRIC_FIELDS",
    "TERMINATION_PRIORITY",
    "TERMINATION_REASONS",
    "UNCATEGORIZED_TERMINATION",
    "TerminalEpisodeRecorder",
    "TerminalMetricsMixin",
    "align_rows",
    "bucket_success_rates",
    "concatenate_rows",
    "group_rows",
    "round_floats",
    "summarize_distribution",
    "summarize_terminal_episodes",
    "wilson_interval",
]
