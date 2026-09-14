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
import re
import subprocess
import sys
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
    overall = _overall("workflow_robot_carried_release_rack_retention_v1_certification.json")
    rate = f"{overall['success_rate'] * 100:.2f}%"
    assert overall["successes"] == 22 and overall["episodes"] == 24
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
    """The caveat travels with the number, and the number is counted rather than remembered.

    This used to assert the word "ten" appeared, which pinned a count in prose and
    nothing else. Recovering the 2026-09-04 campaign moved it -- twenty more reports
    with source bindings arrived at once -- and a test that asserts a spelled-out
    numeral cannot notice that. So the count is read off the reports.
    """

    bound = [
        path
        for path in sorted(EVIDENCE.glob("*.json"))
        if "runtime_source_bindings" in path.read_text(encoding="utf-8")
    ]
    assert len(bound) >= 30, (
        f"only {len(bound)} reports carry runtime source bindings; the documents describe a larger "
        "provenance gap than the evidence now shows"
    )
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert "uncommitted" in document, f"{label} drops the provenance caveat (NEXT_WORK T0)"
        assert "T0" in document, f"{label} does not point at the task that closes it"
        assert "recovered" in document.lower(), f"{label} drops the recovered current run"
        assert f"{len(bound)} reports carry" in document, (
            f"{label} does not state how many reports carry a source binding; the count is "
            f"{len(bound)}"
        )


def test_the_manifest_counts_are_quoted_as_generated() -> None:
    """The counts drifted and nothing caught it, which is what this exists for.

    ``docs/NOW.md`` said 53 canonical, 11 retracted and 159 historical while the
    generated manifest said 64, 12 and 167. Both numbers were true once. Only one
    was current, and the difference is exactly the failure mode the rest of this
    file is about -- so the mechanically generated counts are now pinned in the two
    documents that quote them.
    """

    counts = json.loads((EVIDENCE / "MANIFEST.json").read_text(encoding="utf-8"))["counts"]
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        for group in ("canonical", "retracted", "historical"):
            number = counts[group]
            assert f"{number} {group}" in document or f"**{number} {group}" in document, (
                f"{label} does not quote the generated count of {number} {group} reports"
            )


def test_the_cpu_suite_size_is_quoted_as_collected() -> None:
    """The README names a test count, so the count is collected rather than trusted.

    Collection takes about a second and runs in its own process. If this fails with
    a number a little larger than the README's, the usual cause is an optional
    dependency: ``tests/test_fiducial.py`` skips at import without OpenCV, and
    installing it adds its tests to the collection. The README states the figure for
    the dependency set ``docs/INSTALL.md`` installs.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            "-m",
            "not isaac and not camera and not benchmark",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    match = re.search(r"(\d[\d,]*) tests collected", result.stdout)
    assert match, f"could not read a collected count from pytest:\n{result.stdout[-2000:]}"
    collected = int(match.group(1).replace(",", ""))
    assert f"{collected:,} tests" in README, (
        f"README does not quote the collected CPU suite size of {collected:,} tests"
    )


def test_the_boundary_is_not_overstated() -> None:
    report = json.loads(
        (EVIDENCE / "serviceability_boundary_validation_v2.json").read_text(encoding="utf-8")
    )
    assert report["decision"]["qualified"] is False
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert "not qualified" in document.lower(), f"{label} overstates the current boundary"
        assert "idealized" in document.lower(), f"{label} hides the load-path limitation"


def test_the_contact_feeling_seating_policy_is_reported_as_a_rate_not_a_reward() -> None:
    """The one experiment that could still overturn a standing result, pinned.

    Its training reward reached 98.2 against the blind policy's 43.9 and **a reward
    is not a rate**. Both halves are now measured, so the documents must quote the
    rates and must not quote the reward as one.
    """

    skill = _overall("grapple_insert_v33force_c11065_certification.json")
    assert skill["successes"] == 2977 and skill["episodes"] == 3001
    skill_rate = f"{skill['success_rate'] * 100:.2f}%"

    guarded = _overall("workflow_robot_carried_insert_v33force_c11065_chain_guarded_n96_certification.json")
    policy = _overall("workflow_robot_carried_insert_v33force_c11065_chain_policy_n96_certification.json")
    assert guarded["episodes"] == 96 and policy["episodes"] == 96
    assert guarded["successes"] == 23 and policy["successes"] == 24

    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert skill_rate in document, (
            f"{label} does not quote the contact-feeling skill rate {skill_rate}"
        )
        assert "23/96" in document and "24/96" in document, (
            f"{label} does not carry the paired chain arms, which are what decide the seating phase"
        )
        assert "not a rate" in document or "is not a success rate" in document, (
            f"{label} quotes the training reward without saying it is not a rate"
        )


def test_the_skill_gate_attribution_is_quoted_with_its_denominator() -> None:
    """A failure mode without its count is a story, so both are pinned."""

    report = json.loads((EVIDENCE / "skill_gate_attrition_v1.json").read_text(encoding="utf-8"))
    capture = report["capture"]["pooled"]
    extraction = report["extraction"]["pooled"]
    grip = capture["terms"]["grip_position"]["broken_by"]
    settling = extraction["terms"]["linear_settling"]["broken_by"]

    assert capture["failures"] == 1180 and grip == 1170
    assert extraction["failures"] == 1113 and settling == 1024
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert f"{grip:,} of its {capture['failures']:,}" in document or (
            f"{grip:,}" in document and f"{capture['failures']:,}" in document
        ), f"{label} does not give the capture attribution its denominator"
        assert f"{settling:,}" in document and f"{extraction['failures']:,}" in document, (
            f"{label} does not give the extraction attribution its denominator"
        )


def test_the_shipped_bay_verdict_is_quoted_as_the_tool_computes_it() -> None:
    """The design rule's own verdict on this repository's bay: incompatible."""

    report = json.loads(
        (EVIDENCE / "channel_verdict_shipped_bay_v1.json").read_text(encoding="utf-8")
    )
    verdict = report["verdict"]
    assert verdict["compatible"] is False
    held = verdict["what_the_gate_would_have_to_accept"]["attitude_rad"] * 1000.0
    allowed = verdict["what_the_gate_does_accept"]["attitude_rad"] * 1000.0
    assert round(allowed, 2) == 52.36, f"the acceptance tolerance moved to {allowed:.2f} mrad"
    for document, label in ((README, "README.md"), (NOW, "docs/NOW.md")):
        assert f"{held:.2f} mrad" in document, (
            f"{label} does not quote the {held:.2f} mrad this bay can hold a resting module at"
        )
