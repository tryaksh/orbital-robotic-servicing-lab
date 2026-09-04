"""The checks this repository's discipline rests on, run where CI can see them.

Three generated indexes and one link checker enforce the rules the documentation
claims to follow: every number names the file it came from, every cited file
exists, no withdrawn report is quoted without saying so, and the manifest and
script index describe what is actually on disk.

None of them were in CI. The workflow runs ``ruff`` and ``pytest`` and nothing
else, so a stale manifest or a dead citation reached ``main`` unchallenged --
which is how nine undisclosed citations of retracted evidence accumulated in the
working documents. Running them from the suite puts them behind the same gate as
everything else without teaching the workflow file about them one at a time.

These are text-only and take about a second. They read the repository as it
stands, so a failure here means the working tree is inconsistent, not that the
test is wrong: regenerate the index the message names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script", "argv", "regenerate"),
    [
        ("check_evidence_links", [], "fix the citation the message names"),
        ("build_evidence_manifest", ["--check"], "python scripts/build_evidence_manifest.py"),
        ("build_script_index", ["--check"], "python scripts/build_script_index.py"),
    ],
)
def test_generated_index_is_current(monkeypatch, capsys, script: str, argv: list[str], regenerate: str) -> None:
    module = _load(script)
    monkeypatch.setattr("sys.argv", [f"{script}.py", *argv])
    status = module.main()
    captured = capsys.readouterr().out
    assert status == 0, f"{script} failed; to fix: {regenerate}\n{captured}"


def test_no_document_quotes_retracted_evidence_silently() -> None:
    """The rule directly: a withdrawn number may be named, never leaned on.

    Stated separately from the parametrised run above because it is the one a
    manuscript drafted out of these documents would break first, and a failure
    should say so rather than reading as a generic link error.
    """

    links = _load("check_evidence_links")
    retracted = links.retracted_from_manifest()
    assert retracted, "the manifest lists no retracted evidence; the check would pass vacuously"

    offenders: list[str] = []
    for name in links.DOCUMENTS:
        path = ROOT / name
        if not path.exists() or name == "evidence/RETRACTED.md":
            continue
        for line_no, report in links.undisclosed_retractions(path.read_text(encoding="utf-8"), retracted):
            offenders.append(f"{name}:{line_no} cites {report}")
    assert not offenders, "retracted evidence cited with no disclosure nearby:\n  " + "\n  ".join(offenders)
