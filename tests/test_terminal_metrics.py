"""Terminal-metric capture ordering and held-out aggregation statistics.

The ordering tests reproduce Isaac Lab's control flow: ``step`` computes
terminations and then resets the finished environments *before* returning.  A
fake environment reuses the production
:class:`~zero_g_blade_swap.evaluation.TerminalMetricsMixin`, so a regression in
the mixin's method resolution order or its pre-reset call site fails here
without needing Isaac Sim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from zero_g_blade_swap.evaluation import (
    BLADE_MASS_FIELD,
    RESET_STATION_FIELD,
    TERMINAL_METRIC_FIELDS,
    TERMINATION_REASONS,
    TerminalEpisodeRecorder,
    TerminalMetricsMixin,
    align_rows,
    bucket_success_rates,
    concatenate_rows,
    group_rows,
    round_floats,
    summarize_distribution,
    summarize_terminal_episodes,
    wilson_interval,
)

RESET_SENTINEL = -999.0


class _FakeBaseEnv:
    """Minimal stand-in for ``ManagerBasedRLEnv`` reset/step ordering."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int64)
        # One observable scalar per environment; reset overwrites it.
        self.axial_error = np.zeros(num_envs, dtype=np.float64)
        self.reset_calls: list[list[int]] = []

    def _reset_idx(self, env_ids) -> None:
        self.reset_calls.append([int(value) for value in np.atleast_1d(env_ids)])
        self.axial_error[env_ids] = RESET_SENTINEL
        self.episode_length_buf[env_ids] = 0


class _FakeEnv(TerminalMetricsMixin, _FakeBaseEnv):
    """Fake environment with the production pre-reset hook mixed in."""

    def reset(self) -> None:
        self._reset_idx(np.arange(self.num_envs))
        self.axial_error[:] = 0.0

    def step(self, error: np.ndarray, dones: np.ndarray) -> None:
        self.episode_length_buf += 1
        self.axial_error[:] = error
        finished = np.nonzero(dones)[0]
        if finished.size:
            self._reset_idx(finished)


def _hook(records: list[tuple[int, float, int]]):
    def record(env, env_ids) -> None:
        for env_id in np.atleast_1d(env_ids):
            index = int(env_id)
            records.append((index, float(env.axial_error[index]), int(env.episode_length_buf[index])))

    return record


def test_terminal_values_are_captured_before_the_automatic_reset() -> None:
    records: list[tuple[int, float, int]] = []
    env = _FakeEnv(num_envs=3)
    env.enable_terminal_metrics(_hook(records))
    env.reset()

    env.step(np.array([0.5, 0.6, 0.7]), np.array([False, False, False]))
    env.step(np.array([0.011, 0.9, 0.012]), np.array([True, False, True]))

    assert records == [(0, 0.011, 2), (2, 0.012, 2)]
    # The live buffers are already the next episode's, which is exactly the
    # corruption this capture path exists to avoid.
    assert env.axial_error[0] == RESET_SENTINEL
    assert env.axial_error[2] == RESET_SENTINEL
    assert env.episode_length_buf.tolist() == [0, 2, 0]


def test_initial_reset_records_nothing_and_hook_can_be_disabled() -> None:
    records: list[tuple[int, float, int]] = []
    env = _FakeEnv(num_envs=2)
    env.enable_terminal_metrics(_hook(records))

    env.reset()
    assert records == []
    assert env.reset_calls == [[0, 1]]

    env.disable_terminal_metrics()
    env.step(np.array([0.02, 0.03]), np.array([True, True]))
    assert records == []
    assert env.reset_calls[-1] == [0, 1]


def test_hook_must_be_callable() -> None:
    env = _FakeEnv(num_envs=1)
    with pytest.raises(TypeError):
        env.enable_terminal_metrics(object())


def test_recorder_preserves_every_episode_and_field_order() -> None:
    recorder = TerminalEpisodeRecorder()
    assert recorder.fields == TERMINAL_METRIC_FIELDS
    width = len(TERMINAL_METRIC_FIELDS)

    assert len(recorder) == 0
    assert recorder.rows.shape == (0, width)
    assert recorder.record(np.zeros((0, width))) == 0
    assert recorder.record(np.arange(2 * width, dtype=np.float64).reshape(2, width)) == 2
    assert recorder.record(np.full((3, width), 7.0)) == 3

    assert len(recorder) == 5
    assert recorder.rows.shape == (5, width)
    assert recorder.column("success").tolist() == [0.0, float(width), 7.0, 7.0, 7.0]
    with pytest.raises(ValueError):
        recorder.record(np.zeros((1, width + 1)))


def test_wilson_interval_matches_published_values() -> None:
    # Wilson (1927) score interval; the reference values below are the closed
    # form evaluated independently for a 95% two-sided interval.
    low, high = wilson_interval(0, 0)
    assert (low, high) == (0.0, 1.0)

    low, high = wilson_interval(3028, 3028)
    assert high == 1.0
    assert low == pytest.approx(0.99873, abs=1e-5)

    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.40383, abs=1e-5)
    assert high == pytest.approx(0.59617, abs=1e-5)

    low, high = wilson_interval(95, 100)
    assert low == pytest.approx(0.88825, abs=1e-5)
    assert high == pytest.approx(0.97846, abs=1e-5)

    # A perfect small sample must not claim a tight bound.
    assert wilson_interval(10, 10)[0] < wilson_interval(1000, 1000)[0]

    for bad in ((-1, 10), (11, 10), (5, -1)):
        with pytest.raises(ValueError):
            wilson_interval(*bad)


def test_summarize_distribution_reports_percentiles_and_non_finite() -> None:
    values = np.arange(1.0, 101.0)
    summary = summarize_distribution(values)
    assert summary["count"] == 100
    assert summary["non_finite"] == 0
    assert summary["mean"] == pytest.approx(50.5)
    assert summary["p50"] == pytest.approx(50.5)
    assert summary["p95"] == pytest.approx(np.percentile(values, 95.0))
    assert summary["max"] == 100.0
    assert summary["min"] == 1.0

    contaminated = summarize_distribution([1.0, 2.0, math.nan, math.inf])
    assert contaminated["count"] == 4
    assert contaminated["non_finite"] == 2
    assert contaminated["max"] == 2.0

    assert summarize_distribution([]) == {"count": 0, "non_finite": 0}


def _row(**overrides: float) -> list[float]:
    values = dict.fromkeys(TERMINAL_METRIC_FIELDS, 0.0)
    values.update(overrides)
    return [values[name] for name in TERMINAL_METRIC_FIELDS]


def test_summarize_terminal_episodes_counts_reasons_and_success_rate() -> None:
    success_id = float(TERMINATION_REASONS.index("insertion_success"))
    timeout_id = float(TERMINATION_REASONS.index("time_out"))
    non_finite_id = float(TERMINATION_REASONS.index("non_finite"))

    rows = np.array(
        [
            _row(success=1.0, termination_reason=success_id, curriculum_stage=0.0, axial_error_m=0.001,
                 cycle_time_s=1.0),
            _row(success=1.0, termination_reason=success_id, curriculum_stage=0.0, axial_error_m=0.002,
                 cycle_time_s=2.0),
            _row(success=1.0, termination_reason=success_id, curriculum_stage=1.0, axial_error_m=0.003,
                 cycle_time_s=3.0),
            _row(success=0.0, termination_reason=timeout_id, curriculum_stage=1.0, axial_error_m=0.400,
                 cycle_time_s=12.0),
            _row(success=0.0, termination_reason=non_finite_id, curriculum_stage=2.0, axial_error_m=math.nan,
                 cycle_time_s=4.0),
        ]
    )

    report = summarize_terminal_episodes(rows)
    assert report["episodes"] == 5
    assert report["successes"] == 3
    assert report["success_rate"] == pytest.approx(0.6)
    assert report["termination_reasons"] == {"non_finite": 1, "insertion_success": 3, "time_out": 1}
    assert report["instability_terminations"] == 1
    assert report["non_finite_metric_episodes"] == 1
    assert report["success_rate_wilson_95"]["low"] < 0.6 < report["success_rate_wilson_95"]["high"]

    axial = report["terminal_metrics"]["axial_error_m"]
    assert axial["count"] == 5 and axial["non_finite"] == 1
    assert axial["max"] == pytest.approx(0.400)
    # Successful episodes must be summarized separately from failures.
    assert report["successful_episode_metrics"]["axial_error_m"]["max"] == pytest.approx(0.003)
    assert report["successful_episode_metrics"]["cycle_time_s"]["mean"] == pytest.approx(2.0)

    stages = group_rows(rows, "curriculum_stage")
    assert sorted(stages) == [0, 1, 2]
    assert summarize_terminal_episodes(stages[0])["success_rate"] == pytest.approx(1.0)
    assert summarize_terminal_episodes(stages[1])["success_rate"] == pytest.approx(0.5)
    assert summarize_terminal_episodes(stages[2])["success_rate"] == pytest.approx(0.0)


def test_pooled_statistics_equal_single_pass_statistics() -> None:
    generator = np.random.default_rng(1060)
    blocks = []
    for _ in range(3):
        block = np.zeros((40, len(TERMINAL_METRIC_FIELDS)))
        block[:, TERMINAL_METRIC_FIELDS.index("axial_error_m")] = generator.random(40)
        block[:, TERMINAL_METRIC_FIELDS.index("success")] = (generator.random(40) > 0.1).astype(float)
        blocks.append(block)

    pooled = concatenate_rows(blocks)
    assert pooled.shape == (120, len(TERMINAL_METRIC_FIELDS))
    expected = np.concatenate([block[:, TERMINAL_METRIC_FIELDS.index("axial_error_m")] for block in blocks])
    summary = summarize_terminal_episodes(pooled)["terminal_metrics"]["axial_error_m"]
    assert summary["p95"] == pytest.approx(np.percentile(expected, 95.0))
    assert summary["mean"] == pytest.approx(expected.mean())

    successes = int(sum(block[:, TERMINAL_METRIC_FIELDS.index("success")].sum() for block in blocks))
    assert summarize_terminal_episodes(pooled)["successes"] == successes

    with pytest.raises(ValueError):
        concatenate_rows([np.zeros((2, 3))])


def test_align_rows_maps_optional_columns_by_name() -> None:
    source = (*TERMINAL_METRIC_FIELDS, BLADE_MASS_FIELD)
    rich = np.array([_row(success=1.0, axial_error_m=0.004) + [12.5]])
    plain = np.array([_row(success=1.0, axial_error_m=0.009)])

    # A run that recorded mass keeps it; one that did not gets NaN, never a
    # silently shifted column.
    kept = align_rows(rich, source, source)
    assert kept[0, source.index(BLADE_MASS_FIELD)] == 12.5
    assert kept[0, source.index("axial_error_m")] == pytest.approx(0.004)

    widened = align_rows(plain, TERMINAL_METRIC_FIELDS, source)
    assert widened.shape == (1, len(source))
    assert math.isnan(widened[0, source.index(BLADE_MASS_FIELD)])
    assert widened[0, source.index("axial_error_m")] == pytest.approx(0.009)

    narrowed = align_rows(rich, source, TERMINAL_METRIC_FIELDS)
    assert narrowed.shape == (1, len(TERMINAL_METRIC_FIELDS))
    assert narrowed[0, TERMINAL_METRIC_FIELDS.index("axial_error_m")] == pytest.approx(0.004)

    with pytest.raises(ValueError):
        align_rows(plain, source, source)


def test_reset_station_is_an_optional_named_category() -> None:
    source = (*TERMINAL_METRIC_FIELDS, RESET_STATION_FIELD)
    rows = np.array(
        [
            _row(success=1.0) + [0.0],
            _row(success=0.0) + [0.0],
            _row(success=1.0) + [8.0],
        ]
    )

    stations = group_rows(rows, RESET_STATION_FIELD, source)
    assert sorted(stations) == [0, 8]
    assert summarize_terminal_episodes(stations[0], source)["success_rate"] == pytest.approx(0.5)
    assert summarize_terminal_episodes(stations[8], source)["success_rate"] == pytest.approx(1.0)


def test_missing_optional_column_is_not_counted_as_instability() -> None:
    source = (*TERMINAL_METRIC_FIELDS, BLADE_MASS_FIELD)
    rows = align_rows(np.array([_row(success=1.0)]), TERMINAL_METRIC_FIELDS, source)

    report = summarize_terminal_episodes(rows, source)
    assert report["non_finite_metric_episodes"] == 0
    assert report["successes"] == 1


def test_bucket_success_rates_splits_the_randomized_range() -> None:
    source = (*TERMINAL_METRIC_FIELDS, BLADE_MASS_FIELD)
    rows = np.array(
        [
            _row(success=1.0) + [5.0],    # low band
            _row(success=1.0) + [7.9],    # low band
            _row(success=1.0) + [9.0],    # mid band
            _row(success=0.0) + [11.0],   # mid band
            _row(success=1.0) + [15.0],   # high band, clamped inside
        ]
    )

    report = bucket_success_rates(rows, BLADE_MASS_FIELD, 5.0, 15.0, source)
    assert report["low"]["episodes"] == 2 and report["low"]["success_rate"] == pytest.approx(1.0)
    assert report["mid"]["episodes"] == 2 and report["mid"]["success_rate"] == pytest.approx(0.5)
    assert report["high"]["episodes"] == 1 and report["high"]["success_rate"] == pytest.approx(1.0)
    assert report["minimum_observed_success_rate"] == pytest.approx(0.5)
    assert report["observed_value_range"] == [5.0, 15.0]
    assert report["episodes_without_value"] == 0

    # Rows from a run that never recorded mass must not be counted anywhere.
    widened = align_rows(np.array([_row(success=0.0)]), TERMINAL_METRIC_FIELDS, source)
    mixed = bucket_success_rates(np.concatenate([rows, widened]), BLADE_MASS_FIELD, 5.0, 15.0, source)
    assert mixed["episodes_without_value"] == 1
    assert sum(mixed[label]["episodes"] for label in ("low", "mid", "high")) == 5

    with pytest.raises(ValueError):
        bucket_success_rates(rows, BLADE_MASS_FIELD, 15.0, 5.0, source)


def test_round_floats_keeps_structure_and_non_finite_values() -> None:
    rounded = round_floats({"a": 1.23456789, "b": [2.3456789, math.inf], "c": 7, "d": "text"}, digits=3)
    assert rounded == {"a": 1.235, "b": [2.346, math.inf], "c": 7, "d": "text"}
