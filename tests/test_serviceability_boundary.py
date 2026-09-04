"""The serviceability-boundary comparison is conservative and fail closed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serviceability_boundary_for_test", ROOT / "scripts" / "validate_serviceability_boundary.py"
)
assert SPEC is not None and SPEC.loader is not None
BOUNDARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARY)


def _point(rate: float, low: float, high: float, episodes: int = 16) -> dict:
    return {
        "success_rate": rate,
        "episodes": episodes,
        "wilson_95": {"low": low, "high": high},
    }


def test_classification_requires_wilson_separation_for_predicted_loss() -> None:
    nominal = _point(0.94, 0.72, 0.99)
    separated = BOUNDARY.classify_simulation_point(
        label="outside", analytically_feasible=False, point=_point(0.0, 0.0, 0.19), nominal=nominal
    )
    overlapping = BOUNDARY.classify_simulation_point(
        label="outside", analytically_feasible=False, point=_point(0.75, 0.51, 0.90), nominal=nominal
    )

    assert separated["simulation"]["statistically_separated_loss"] is True
    assert separated["comparison"] == "supports_boundary"
    assert overlapping["simulation"]["statistically_separated_loss"] is False
    assert overlapping["comparison"] == "does_not_support_boundary"


def test_feasible_point_fails_when_simulation_has_a_separated_loss() -> None:
    result = BOUNDARY.classify_simulation_point(
        label="inside",
        analytically_feasible=True,
        point=_point(0.06, 0.01, 0.28),
        nominal=_point(0.94, 0.72, 0.99),
    )
    assert result["comparison"] == "does_not_support_boundary"


def test_loader_rejects_required_input_that_is_not_canonical(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in BOUNDARY.REQUIRED_INPUTS:
        (evidence / name).write_text("{}", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"canonical": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        BOUNDARY.load_inputs(evidence, manifest)


def test_current_geometry_recomputes_the_exact_section_predictions() -> None:
    current = BOUNDARY.recompute_geometry()
    small = BOUNDARY._section(current, 0.120, 0.016)
    nominal = BOUNDARY._section(current, 0.130, 0.020)
    large = BOUNDARY._section(current, 0.140, 0.026)

    assert small["lead_ins_admit_the_delivered_attitude"] is True
    assert small["pads_can_follow_the_corner"] is False
    assert small["accepted"] is False
    assert nominal["accepted"] is True
    assert large["lead_ins_admit_the_delivered_attitude"] is False
    assert large["pads_can_follow_the_corner"] is True
    assert large["accepted"] is False


def test_repository_evidence_stays_unqualified_and_retains_every_boundary_arm() -> None:
    reports, bindings = BOUNDARY.load_inputs(
        ROOT / "evidence", ROOT / "evidence" / "MANIFEST.json"
    )
    current = BOUNDARY.recompute_geometry()
    report = BOUNDARY.build_report(
        reports,
        bindings,
        {"available": True, "commit": "a" * 40, "dirty": False},
        current,
    )

    assert report["decision"]["qualified"] is False
    assert report["decision"]["status"] == "not_qualified"
    assert report["dimensions"]["rack_clearance"]["status"] == "mismatch"
    assert report["dimensions"]["module_section"]["status"] == "mismatch"
    assert report["dimensions"]["base_offset"]["status"] == "mismatch"
    assert len(report["dimensions"]["base_offset"]["points"]) == 3
    base_y = report["dimensions"]["base_offset"]["points"][2]
    assert base_y["analytically_feasible"] is True
    assert base_y["simulation"]["statistically_separated_loss"] is True
    assert base_y["comparison"] == "does_not_support_boundary"
    assert len(report["dimensions"]["rack_clearance"]["points"]) == 3
    assert len(report["dimensions"]["module_section"]["points"]) == 3
    assert report["dimensions"]["capture_geometry"]["status"] == "analytical_only"
    assert report["dimensions"]["load_path_type"]["base_mount_ablation"]["status"] == "excluded_fixed_root"
    assert report["protocol"]["tolerances_changed_for_this_report"] is False
