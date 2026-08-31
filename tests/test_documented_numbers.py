"""Every headline number in the docs must still match the evidence behind it.

This repository's most expensive failure mode is not a wrong measurement. It is a
*right* measurement that stopped being true and stayed in the prose:
``evidence/RETRACTED.md`` lists eight, and each was found by hand, late, by
someone remembering to compare a figure against a report.

The checks already here catch two of the three ways that happens.
``check_criterion_currency.py`` asks whether a report predates the code defining
it, and ``check_source_provenance.py`` asks whether the run can still be
reproduced. Neither asks the simplest question: **does the number written in the
README still equal the number in the JSON?**

So this reads the figures out of ``evidence/`` and asserts the documents quote
them. It is deliberately narrow -- only the headline claims, the ones a reader
takes away and a reviewer checks -- because a test that pinned every number in
40 KB of prose would fail constantly and be disabled.

Source-level and CPU-only: no simulator, no GPU, no checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
README = (ROOT / "README.md").read_text(encoding="utf-8")
NOW = (ROOT / "docs" / "NOW.md").read_text(encoding="utf-8")


def _overall(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))["overall"]


def test_the_strict_chain_rate_is_quoted_as_measured() -> None:
    """The current rate includes release of both robot supports and a rack-only recheck."""
    overall = _overall("workflow_robot_carried_release_recheck_v2_certification.json")
    rate = f"{overall['success_rate'] * 100:.2f}%"
    assert overall["successes"] == 17 and overall["episodes"] == 24
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert rate in document, f"{label} does not quote the chain rate {rate}"

    wilson = overall["success_rate_wilson_95"]
    interval = f"[{wilson['low'] * 100:.1f}%, {wilson['high'] * 100:.1f}%]"
    assert interval in README or interval.replace("%", "") in README
    assert interval in NOW or interval.replace("%", "") in NOW


def test_the_legacy_supported_settle_rate_is_not_presented_as_current() -> None:
    overall = _overall("workflow_robot_carried_m130pin_guarded_certification.json")
    assert overall["successes"] == 94 and overall["episodes"] == 96
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert "97.92%" in document, f"{label} loses the preserved legacy comparator"
        assert "legacy" in document.lower(), f"{label} presents the old criterion as current"


@pytest.mark.parametrize(
    ("report", "description"),
    [
        ("grapple_grasp_v7m130_on_derived_rack_certification.json", "capture skill"),
        ("grapple_extract_v18pin_certification.json", "extraction skill"),
    ],
)
def test_the_skill_rates_are_quoted_as_measured(report: str, description: str) -> None:
    """Both miss the 95% gate, and both numbers are published rather than rounded away."""
    rate = f"{_overall(report)['success_rate'] * 100:.2f}%"
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert rate in document, f"{label} does not quote the {description} rate {rate}"


def test_the_insert_negative_result_is_quoted_with_its_sample_size() -> None:
    """A 0.00% is only meaningful with the episode count beside it."""
    overall = _overall("grapple_insert_v20chain_certification.json")
    assert overall["success_rate"] == 0.0
    assert f"{overall['episodes']:,}" in README or str(overall["episodes"]) in README
    assert f"{overall['episodes']:,}" in NOW or str(overall["episodes"]) in NOW


def test_the_interface_limit_is_quoted_from_its_own_gate() -> None:
    """The idealized force diagnostic remains visible without becoming a hardware claim."""
    gate = json.loads((EVIDENCE / "grasp_axial_pull_gate.json").read_text(encoding="utf-8"))["gate"]
    required = gate["required_axial_force_n"]
    assert round(required, 1) == 66.4, f"the required axial force moved to {required}"
    assert "66.4 N" in README, "README no longer quotes the axial force the task demands"
    assert "not a hardware load rating" in README


def test_the_insert_diagnosis_is_quoted_against_the_tolerance_it_missed() -> None:
    """84.5 mrad against a 52.4 mrad success tolerance, and the null result beside it.

    An earlier version of this test compared against 20.5 mrad and called it the
    channel's admittance. That was the *settled* attitude, not an entry limit,
    and the claim is retracted -- so this pins the comparison that survived.
    """
    report = json.loads((EVIDENCE / "insert_attitude_diagnosis.json").read_text(encoding="utf-8"))
    tolerance = report["success_orientation_tolerance_mrad"]
    assert round(tolerance, 1) == 52.4, f"the orientation tolerance moved to {tolerance}"
    for document, label in ((README, "docs"), (NOW, "docs/NOW.md")):
        assert f"{tolerance:.1f} mrad" in document, f"{label} does not quote the {tolerance:.1f} mrad tolerance"

    # The three OBJECTIVE arms must agree to within a milliradian; that
    # agreement IS the finding, so a document quoting one without the others
    # would mislead. Load-path arms are excluded on purpose -- they change the
    # interface rather than the reward, and the whole point is that those are
    # different questions.
    angles = [
        arm["orientation_error_mrad"]["median"]
        for arm in report["arms"]
        if not any(word in arm["label"].lower() for word in ("lock", "compliance"))
    ]
    assert len(angles) >= 3, "the diagnosis needs all three objective arms to make its point"
    assert max(angles) - min(angles) < 1.0, (
        f"the three objectives no longer agree ({angles}); the 'interface, not reward' "
        "conclusion rests on them landing together"
    )
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert "84.26" in document and "84.58" in document, (
            f"{label} does not show the objectives landing together, which is the result"
        )


def test_the_provenance_caveat_is_stated_where_the_number_is() -> None:
    """One source-bound chain run recovers while nine older reports remain lost."""
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert "uncommitted" in document, f"{label} drops the provenance caveat (NEXT_WORK T0)"
        assert "T0" in document, f"{label} does not point at the task that closes it"
        assert "recovered" in document.lower(), f"{label} drops the recovered current run"
        assert "nine" in document.lower(), f"{label} drops the nine lost source-bound reports"


def test_the_boundary_is_not_overstated() -> None:
    report = json.loads(
        (EVIDENCE / "serviceability_boundary_validation_v2.json").read_text(encoding="utf-8")
    )
    assert report["decision"]["qualified"] is False
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert "not qualified" in document.lower(), f"{label} overstates the current boundary"
        assert "idealized" in document.lower(), f"{label} hides the load-path limitation"
