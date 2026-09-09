import json
from collections import Counter
from pathlib import Path

import pytest

from assembly_recovery.faults import assert_training_case
from assembly_recovery.training_cases import support_grid_cases, uniform_training_cases

STUDY = json.loads((Path(__file__).resolve().parents[1] / "configs/study.json").read_text())


@pytest.mark.parametrize("size", [32, 64, 128])
def test_uniform_sampler_matches_nominal_and_bin_budget_and_is_replayable(size):
    first = uniform_training_cases(STUDY, 170, 0, size)
    assert first == uniform_training_cases(STUDY, 170, 0, size)
    assert first != uniform_training_cases(STUDY, 170, 1, size)
    bins = Counter(c["bin_id"] for c in first)
    assert bins.pop("nominal") == size // 4
    assert set(bins.values()) == {size // 8}
    for case in first:
        assert_training_case(STUDY, 170, case)


@pytest.mark.parametrize("seed", [10070, 20070, -1])
def test_training_sampler_rejects_development_and_final_seeds(seed):
    with pytest.raises(ValueError, match="Training requires"):
        uniform_training_cases(STUDY, seed, 0, 64)


def test_development_grids_cover_predeclared_boundaries_and_keep_tests_closed():
    for grid in ("low", "interior", "high"):
        cases = support_grid_cases(STUDY, 10071, grid)
        assert len(cases) == 32
        assert all(c["split"] == "development" for c in cases)
        assert sum(c["family"] == "nominal" for c in cases) == 8
    with pytest.raises(ValueError):
        support_grid_cases(STUDY, 20070, "high")
