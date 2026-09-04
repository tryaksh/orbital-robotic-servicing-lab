"""Unit tests for pooling a sweep point across seeds.

Pooling is arithmetic, so the risk is not that it computes the wrong mean. The
risks are that it pools archives that are not comparable, that it silently pools
one directory and calls the result three seeds, or that it leaves a reader unable
to tell which rows in a merged table were measured at which sample size. Each of
those is a refusal or a recorded field, and each is tested here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pool_sweep_points", ROOT / "scripts/pool_sweep_points.py")
assert _spec is not None and _spec.loader is not None
pool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pool)

FIELDS = ["success", "axial_error_m", "lateral_error_m", "orientation_error_rad"]


def _archive(directory: Path, tag: str, successes: int, episodes: int, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = np.zeros((episodes, len(FIELDS)), dtype=np.float32)
    rows[:successes, 0] = 1.0
    rows[:, 1] = 0.001
    np.savez(
        directory / f"{tag}.npz",
        rows=rows,
        fields=np.array(FIELDS),
        metadata=np.array(json.dumps({"seed": seed})),
    )


def test_pooling_sums_episodes_and_successes(tmp_path: Path) -> None:
    dirs = []
    for index, (successes, seed) in enumerate([(10, 4070), (20, 5070), (30, 6070)]):
        directory = tmp_path / f"seed{index}"
        _archive(directory, "nominal", successes, 64, seed)
        dirs.append(directory)

    entry = pool.pool_point(dirs, "nominal")

    assert entry["episodes"] == 192
    assert entry["successes"] == 60
    assert entry["success_rate"] == pytest.approx(60 / 192, abs=1e-6)
    assert entry["pooled_from"]["seeds"] == [4070, 5070, 6070]
    assert entry["pooled_from"]["episodes_each"] == [64, 64, 64]


def test_pooling_narrows_the_interval(tmp_path: Path) -> None:
    """The whole point: the same rate at three times the n separates where one does not."""

    single = tmp_path / "one"
    _archive(single, "nominal", 22, 64, 4070)
    one = pool.pool_point([single, single], "nominal")

    dirs = []
    for index in range(3):
        directory = tmp_path / f"seed{index}"
        _archive(directory, "nominal", 22, 64, 4070 + 1000 * index)
        dirs.append(directory)
    three = pool.pool_point(dirs, "nominal")

    assert three["success_rate"] == pytest.approx(one["success_rate"], abs=1e-6)
    width_of = lambda e: e["wilson_95"]["high"] - e["wilson_95"]["low"]  # noqa: E731
    assert width_of(three) < width_of(one)


def test_a_missing_seed_is_refused_rather_than_pooled_short(tmp_path: Path) -> None:
    present = tmp_path / "present"
    _archive(present, "nominal", 10, 64, 4070)
    absent = tmp_path / "absent"
    absent.mkdir()

    with pytest.raises(SystemExit, match="does not exist"):
        pool.pool_point([present, absent], "nominal")


def test_archives_with_different_columns_are_refused(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _archive(first, "nominal", 10, 64, 4070)
    second = tmp_path / "second"
    second.mkdir()
    np.savez(
        second / "nominal.npz",
        rows=np.zeros((64, 2), dtype=np.float32),
        fields=np.array(["success", "something_else"]),
        metadata=np.array(json.dumps({"seed": 5070})),
    )

    with pytest.raises(SystemExit, match="different columns"):
        pool.pool_point([first, second], "nominal")
