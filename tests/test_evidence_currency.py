"""Unit tests for the evidence-currency gate.

The rule this defends is the one that cost a session: ``evidence/`` named grasp
v2, extract v2 and insert v3 while the demonstration loaded v3, v4 and v5. The
tests below reconstruct exactly that situation and assert the gate fails on it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_evidence_currency", ROOT / "scripts/check_evidence_currency.py"
)
assert _spec is not None and _spec.loader is not None
currency = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(currency)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest().upper()


def _certification(path: Path, digest: str, checkpoint: str) -> None:
    path.write_text(
        json.dumps({"policy": {"checkpoint_sha256": digest, "checkpoint": checkpoint}}),
        encoding="utf-8",
    )


def _workflow_report(path: Path, digests: dict[str, str], *, set_hash: str | None = None) -> None:
    payload: dict[str, object] = {
        "workflow": "install",
        "checkpoints": {name: f"logs/{name}.pth" for name in digests},
        "checkpoint_sha256": dict(digests),
    }
    payload["policy_set_sha256"] = (
        set_hash
        if set_hash is not None
        else currency.combined_policy_sha256(digests, tuple(digests))
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir()
    return directory


def test_a_workflow_whose_policies_are_all_certified_passes(tmp_path: Path, evidence: Path) -> None:
    digests = {"capture": _sha("grasp-v4"), "extract": _sha("extract-v5"), "insert": _sha("insert-v7")}
    for name, digest in digests.items():
        _certification(evidence / f"grapple_{name}_certification.json", digest, f"{name}.pth")
    report = tmp_path / "install_report.json"
    _workflow_report(report, digests)

    assert currency.main([str(report), "--evidence", str(evidence)]) == 0


def test_the_v2_v2_v3_mistake_is_caught(tmp_path: Path, evidence: Path) -> None:
    """The exact failure this tool exists for: evidence describes older policies."""

    loaded = {"capture": _sha("grasp-v3"), "extract": _sha("extract-v4"), "insert": _sha("insert-v5")}
    for name, label in (("capture", "grasp-v2"), ("extract", "extract-v2"), ("insert", "insert-v3")):
        _certification(evidence / f"grapple_{name}_certification.json", _sha(label), f"{name}.pth")
    report = tmp_path / "install_report.json"
    _workflow_report(report, loaded)

    assert currency.main([str(report), "--evidence", str(evidence)]) == 1

    result = currency.check_report(report, currency.load_certified_digests(evidence))
    assert result["passed"] is False
    assert sorted(result["uncertified"]) == ["capture", "extract", "insert"]


def test_one_stale_policy_out_of_three_still_fails(tmp_path: Path, evidence: Path) -> None:
    """A chain is only as current as its least current policy."""

    digests = {"capture": _sha("grasp-v4"), "extract": _sha("extract-v5"), "insert": _sha("insert-v7")}
    _certification(evidence / "capture.json", digests["capture"], "capture.pth")
    _certification(evidence / "extract.json", digests["extract"], "extract.pth")
    _certification(evidence / "insert_old.json", _sha("insert-v6"), "insert.pth")
    report = tmp_path / "install_report.json"
    _workflow_report(report, digests)

    assert currency.main([str(report), "--evidence", str(evidence)]) == 1
    result = currency.check_report(report, currency.load_certified_digests(evidence))
    assert result["uncertified"] == ["insert"]


def test_hashes_are_compared_case_insensitively(tmp_path: Path, evidence: Path) -> None:
    digest = _sha("grasp-v4")
    _certification(evidence / "capture.json", digest.lower(), "capture.pth")
    report = tmp_path / "report.json"
    _workflow_report(report, {"capture": digest})

    assert currency.main([str(report), "--evidence", str(evidence)]) == 0


def test_a_tampered_policy_set_hash_fails_even_when_every_policy_is_certified(
    tmp_path: Path, evidence: Path
) -> None:
    digests = {"capture": _sha("grasp-v4"), "extract": _sha("extract-v5")}
    for name, digest in digests.items():
        _certification(evidence / f"{name}.json", digest, f"{name}.pth")
    report = tmp_path / "report.json"
    _workflow_report(report, digests, set_hash=_sha("something-else"))

    assert currency.main([str(report), "--evidence", str(evidence)]) == 1


def test_a_pooled_single_skill_certification_is_resolved_by_its_own_hash(
    tmp_path: Path, evidence: Path
) -> None:
    """The pooled shape carries one hash under ``policy`` rather than a dict."""

    digest = _sha("insert-v7")
    _certification(evidence / "grapple_insert_v7_certification.json", digest, "insert.pth")
    report = tmp_path / "pooled.json"
    report.write_text(json.dumps({"policy": {"checkpoint_sha256": digest, "checkpoint": "insert.pth"}}), "utf-8")

    assert currency.main([str(report), "--evidence", str(evidence)]) == 0


def test_a_report_with_no_hash_at_all_fails_rather_than_passing_vacuously(
    tmp_path: Path, evidence: Path
) -> None:
    _certification(evidence / "capture.json", _sha("grasp-v4"), "capture.pth")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"workflow": "install"}), encoding="utf-8")

    assert currency.main([str(report), "--evidence", str(evidence)]) == 1


def test_an_evidence_directory_that_records_no_hashes_fails(tmp_path: Path, evidence: Path) -> None:
    (evidence / "prose_only.json").write_text(json.dumps({"title": "no policy block"}), encoding="utf-8")
    report = tmp_path / "report.json"
    _workflow_report(report, {"capture": _sha("grasp-v4")})

    assert currency.main([str(report), "--evidence", str(evidence)]) == 1


def test_unreadable_evidence_files_are_skipped_rather_than_crashing(tmp_path: Path, evidence: Path) -> None:
    (evidence / "broken.json").write_text("{not json", encoding="utf-8")
    digest = _sha("grasp-v4")
    _certification(evidence / "capture.json", digest, "capture.pth")
    report = tmp_path / "report.json"
    _workflow_report(report, {"capture": digest})

    assert currency.main([str(report), "--evidence", str(evidence)]) == 0


def test_the_audit_can_be_written_as_json(tmp_path: Path, evidence: Path) -> None:
    digest = _sha("grasp-v4")
    _certification(evidence / "capture.json", digest, "capture.pth")
    report = tmp_path / "report.json"
    _workflow_report(report, {"capture": digest})
    audit = tmp_path / "out" / "audit.json"

    assert currency.main([str(report), "--evidence", str(evidence), "--json", str(audit)]) == 0
    written = json.loads(audit.read_text(encoding="utf-8"))
    assert written["reports"][0]["policies"][0]["certified_by"] == ["capture.json"]
