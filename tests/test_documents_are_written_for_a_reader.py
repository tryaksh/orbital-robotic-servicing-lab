"""The maintained documents describe the project, not the work of producing it.

A reader has no idea what a session is. Headings like *"Done in this session"*, a
repository map that opens by addressing an agent, and a task list described as
*"the open task list with costs, as the sessions wrote it"* all leaked the working
frame into documents whose audience is someone reading the repository. Git history
already records who changed what and when.

This is a lint, not a style opinion, and it is here because the cleanup was done
once by hand and would come back the next time someone writes a summary of their
own work into a maintained file.

Three exemptions, and they are real. `AGENTS.md`, `CLAUDE.md` and everything under
`docs/handover/` are addressed to whoever picks the work up next, so process words
belong there.

**Genuine domain vocabulary is not process-speak.** A *handoff* here is one skill
handing state to the next skill in the chain -- a technical term, load-bearing in
several results -- and a *handover* is the same thing. Neither is swept away, which
is why this checks specific phrasings rather than a word list.

CPU only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Reader-facing and therefore linted. `AGENTS.md`, `CLAUDE.md` and
#: `docs/handover/` are exempt on purpose.
MAINTAINED = (
    "README.md",
    "ROADMAP.md",
    "docs/NOW.md",
    "docs/NEXT_WORK.md",
    "docs/REPO_MAP.md",
    "docs/DEMOS.md",
    "docs/INSTALL.md",
    "docs/sim_to_real.md",
    "docs/seating_controller.md",
    "docs/paper_position.md",
    "docs/PAPER_PLAN.md",
    "docs/service_interface_spec.md",
    "docs/compute_service_demo.md",
)

#: Each pattern names a way the working frame gets into a reader-facing document,
#: with the thing to write instead. Narrow on purpose: "session" inside a
#: filename, a branch name or a domain term must not trip.
FORBIDDEN = (
    (
        r"\b(?:this|the|last|previous|current|next|a) session\b",
        "name the date, or say what is true now; a reader does not know what a session is",
    ),
    (
        r"\bsessions (?:wrote|ran|did|found|left)\b",
        "say what was found, not who was working when",
    ),
    (
        r"\bin this session\b",
        "state what is true now; git history records when it changed",
    ),
    (
        r"\bif you are an agent\b",
        "address the reader, or nobody",
    ),
    (
        r"\bso an agent knows\b",
        "say what the structure is for, not who reads it",
    ),
    (
        r"\ban agent works under\b",
        "describe the rules, not their audience",
    ),
    (
        r"[Pp]lain English",
        "the writing standard is an instruction to whoever writes, not content for whoever reads",
    ),
)


@pytest.mark.parametrize("name", MAINTAINED)
def test_no_maintained_document_carries_the_working_frame(name: str) -> None:
    path = ROOT / name
    assert path.is_file(), f"{name} is listed as maintained and does not exist"
    text = path.read_text(encoding="utf-8")

    offences: list[str] = []
    for pattern, instead in FORBIDDEN:
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            offences.append(f"{name}:{line}: {match.group(0)!r} -- {instead}")
    assert not offences, "\n".join(offences)


def test_the_exempt_documents_are_still_exempt_and_still_exist() -> None:
    """The exemptions are deliberate, so they are named rather than implied.

    If one of these is ever renamed or removed, this test should be what notices,
    not a lint failure in a file that was never meant to be linted.
    """

    for name in ("AGENTS.md", "CLAUDE.md"):
        assert (ROOT / name).is_file(), f"{name} is exempt from the reader lint and is missing"
    handover = ROOT / "docs" / "handover"
    assert handover.is_dir() and any(handover.glob("*.md"))
    assert not any(name.startswith("docs/handover/") for name in MAINTAINED)


def test_the_lint_does_not_sweep_away_domain_vocabulary() -> None:
    """A handoff is one skill handing state to the next, and it has to survive.

    The failure mode this guards against is a future tightening of the patterns
    above into a word list that deletes a technical term because it looks like
    process-speak. Several results are about the hand-over distribution, and a
    document that cannot say so is worse than one with a stray "session" in it.
    """

    sample = (
        "The transit hands the insertion over at 46 mrad, and the skill's certificate "
        "describes late-stroke states rather than the handoff its caller delivers. "
        "A handover trace would settle it."
    )
    for pattern, _ in FORBIDDEN:
        assert not re.search(pattern, sample), f"{pattern!r} would delete domain vocabulary"
