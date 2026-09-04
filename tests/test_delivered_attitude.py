"""The delivered attitude is the number everything else is derived from.

`DELIVERED_ATTITUDE_RAD` sets the clearance window's lower bound, the derived
guide offset, the boundary decision and the published interface regime. It was
carried for ten days with a docstring citing a report field that has never
existed, and nothing checked that because nothing could: a constant's stated
provenance is prose.

It is not prose any more. `scripts/measure_delivered_attitude.py` reads the
quantity out of recorded traces, and these tests hold the two things that make
its answer meaningful -- that the constant is still what the workcell is derived
from, and that the regime boundaries the report prints are the library's rather
than the script's.

Deliberately reads no trace. `artifacts/` is gitignored, so a test that opened
one would pass here and fail collection in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def measure():
    script = ROOT / "scripts" / "measure_delivered_attitude.py"
    spec = importlib.util.spec_from_file_location("measure_delivered_attitude_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_delivered_attitude_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_nothing_writes_the_field_the_constant_used_to_cite() -> None:
    """`handoff_attitude_rad` may not appear as a field a payload carries.

    The constant's docstring named it as its own provenance for ten days and no
    run has ever written it. Prose may still mention the name -- two docstrings
    now do, to say it does not exist -- so this looks for the *quoted* form, which
    is what a dict key or a trace field tuple would be.

    If it starts writing, that is good news and this test should be replaced by
    one asserting the field is populated: the delivered attitude could then be
    read straight out of a report instead of out of a trace.
    """

    offenders = []
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            body = path.read_text(encoding="utf-8", errors="ignore")
            if '"handoff_attitude_rad"' in body or "'handoff_attitude_rad'" in body:
                offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, (
        "handoff_attitude_rad is written as a field again; the constant can be measured "
        "directly now and this test should say so instead: " + ", ".join(offenders)
    )


def test_the_constant_the_workcell_is_derived_from_has_not_moved(measure) -> None:
    """A tripwire, not a preference.

    Moving `DELIVERED_ATTITUDE_RAD` re-derives `GUIDE_CENTER_OFFSET_Y`, the
    clearance window, the section grid and the boundary decision. It is a
    workcell rebuild. If this test fails, the report the script writes is stale
    and so is every number computed from the old value.
    """

    script = ROOT / "scripts" / "check_workcell_geometry.py"
    spec = importlib.util.spec_from_file_location("workcell_geometry_for_attitude", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["workcell_geometry_for_attitude"] = module
    spec.loader.exec_module(module)
    assert module.DELIVERED_ATTITUDE_RAD == measure.ASSERTED_DELIVERED_ATTITUDE_RAD


def test_the_regimes_the_report_prints_are_the_librarys(measure) -> None:
    """The three readings that make the report a decision rather than a table.

    Below the passive-alignment limit a plain channel is a solution; above it a
    controller has to centre the module; above `2c/stroke` the bay must also
    square it during the stroke. The measured median and the asserted constant
    fall on opposite sides of *both* boundaries, which is the whole finding.
    """

    assert measure.implied_requirement(0.00819)["interface_regime"] == "passive"
    assert measure.implied_requirement(0.01450)["interface_regime"] == "active_centring"
    assert measure.implied_requirement(0.02607)["interface_regime"] == "active_centring"
    assert measure.implied_requirement(0.046)["interface_regime"] == "active_centring_and_correction"


def test_the_implied_clearance_is_the_same_arithmetic_as_the_window(measure) -> None:
    for attitude in (0.00819, 0.0145, 0.046):
        window = measure.implied_requirement(attitude)["lateral_clearance_window_m"]
        assert window["lower_bound_m"] == pytest.approx(0.5 * attitude * measure.MODULE_LENGTH_M, abs=1.0e-12)


def test_summarize_reports_the_spread_and_not_just_the_middle(measure) -> None:
    values = [0.001 * value for value in range(1, 101)]
    stats = measure.summarize(values)
    assert stats["count"] == 100
    assert stats["median_rad"] == pytest.approx(0.0505, abs=1.0e-9)
    assert stats["max_rad"] == pytest.approx(0.100, abs=1.0e-9)
    assert stats["min_rad"] == pytest.approx(0.001, abs=1.0e-9)
    assert stats["p95_rad"] > stats["median_rad"]


def _trace(commit: str, dirty: bool, seed: int, task: str = "T") -> dict:
    return {
        "trace": f"seed{seed}",
        "seed": seed,
        "task": task,
        "source_revision": {"available": True, "commit": commit, "dirty": dirty},
        "checkpoint_sha256": "x",
        "environments_traced": 2,
        "environments_that_handed_over": 2,
        "attitudes_rad": [0.008, 0.009],
    }


def test_pooling_refuses_what_the_aggregator_refuses(measure) -> None:
    """The same three refusals, for the same reason.

    A number pooled across two commits, or off a dirty tree, cannot be
    reproduced from the source it claims. This script writes evidence, so it
    holds the line `aggregate_evaluation.py` holds.
    """

    ok = [_trace("a" * 40, False, seed) for seed in (4070, 5070)]
    assert measure.build_report(ok)["measurement"]["pooled"]["count"] == 4

    with pytest.raises(ValueError, match="different source commits"):
        measure.build_report([_trace("a" * 40, False, 4070), _trace("b" * 40, False, 5070)])
    with pytest.raises(ValueError, match="clean source revisions"):
        measure.build_report([_trace("a" * 40, False, 4070), _trace("a" * 40, True, 5070)])
    with pytest.raises(ValueError, match="different tasks"):
        measure.build_report(
            [_trace("a" * 40, False, 4070), _trace("a" * 40, False, 5070, task="other")]
        )
