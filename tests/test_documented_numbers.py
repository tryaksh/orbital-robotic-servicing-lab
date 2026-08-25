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


def test_the_chain_rate_is_quoted_as_measured() -> None:
    """97.92% over 94 of 96 episodes, with its Wilson interval."""
    overall = _overall("workflow_robot_carried_m130pin_guarded_certification.json")
    rate = f"{overall['success_rate'] * 100:.2f}%"
    assert overall["successes"] == 94 and overall["episodes"] == 96
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert rate in document, f"{label} does not quote the chain rate {rate}"

    wilson = overall["success_rate_wilson_95"]
    interval = f"[{wilson['low'] * 100:.1f}%, {wilson['high'] * 100:.1f}%]"
    assert interval in README or interval.replace("%", "") in README
    assert interval in NOW or interval.replace("%", "") in NOW


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
    """6 N held against 66.4 N demanded is the measurement the project rests on."""
    gate = json.loads((EVIDENCE / "grasp_axial_pull_gate.json").read_text(encoding="utf-8"))["gate"]
    required = gate["required_axial_force_n"]
    assert round(required, 1) == 66.4, f"the required axial force moved to {required}"
    assert "66.4 N" in README, "README no longer quotes the axial force the task demands"


def test_the_insert_diagnosis_is_quoted_against_the_channel_it_failed() -> None:
    """84.6 mrad against 20.5 mrad admissible is why the insert skill does not seat."""
    report = json.loads((EVIDENCE / "insert_attitude_diagnosis.json").read_text(encoding="utf-8"))
    admits = report["channel_admittance_mrad"]
    assert f"{admits:.1f} mrad" in README or f"**{admits:.1f} mrad**" in README
    assert f"{admits:.1f} mrad" in NOW or f"**{admits:.1f} mrad**" in NOW

    converged = [
        arm
        for arm in report["arms"]
        if "v21time" in arm["label"] and "lock" not in arm["label"].lower()
    ]
    assert converged, "the converged time-cost arm is missing from the diagnosis"
    median = converged[0]["orientation_error_mrad"]["median"]
    assert f"{median:.1f} mrad" in README or f"**{median:.1f} mrad**" in README
    assert f"{median:.1f} mrad" in NOW or f"**{median:.1f} mrad**" in NOW


def test_the_provenance_caveat_is_stated_where_the_number_is() -> None:
    """The chain rate cannot be quoted without saying it is not reproducible yet.

    While T0 is open this is the single most important sentence in either
    document, and it has to sit next to the figure it qualifies rather than in a
    footnote somewhere else.
    """
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert "uncommitted" in document, f"{label} drops the provenance caveat (NEXT_WORK T0)"
        assert "T0" in document, f"{label} does not point at the task that closes it"
