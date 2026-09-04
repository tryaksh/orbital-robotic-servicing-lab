"""No shell script may read ``$?`` after a command substitution on the same line.

``echo "[$(date +%H:%M:%S)] thing exit=$?"`` does not report the exit status of
the thing. Expansion runs left to right, so ``date`` executes first and
overwrites the status; the line prints ``exit=0`` whatever happened. Forty-five
lines across twenty-two shipped scripts and twenty-eight more across the campaign
queues did exactly that, including every certification script in the repository.
Nothing branched on the value, so no published result is wrong -- but every
``exit=`` line in every campaign log was reporting the clock, and a run that died
would have been logged as a success.

The fix is one line earlier::

    rc=$?
    echo "[$(date +%H:%M:%S)] thing exit=$rc"

Order is what makes it a bug, and the same left-to-right rule makes the mirror
image correct: ``say "exit=$? -> $(ckpt)"`` reads the status before anything can
disturb it. Five lines in this repository do that and they are fine. A lint that
flagged them too would train people to ignore it.

Text only, so CI runs it. It exists because the failure is invisible: the log
looks right, and the only way to notice is to already suspect it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: Shell that reports what a long GPU job did. Both places it lives.
SHELL_DIRECTORIES = (ROOT / "scripts", ROOT / "artifacts" / "campaign")
#: A substitution -- ``$(...)`` or a backquote -- reached before ``$?`` on the
#: same line. It runs during expansion and overwrites the status.
CLOBBERED = re.compile(r"(?:\$\(|`)[^\n]*\$\?")

#: Campaign queues that were mid-run when the rule was introduced. Bash reads a
#: script incrementally and remembers a byte offset, so editing one while it
#: executes can make it run garbage; each of these had hours of GPU work left.
#: Fix them once they finish and delete them from this list -- the test below
#: fails if an entry no longer needs to be here, so the exemption cannot rot.
IN_FLIGHT = {
    "queue_noised_skill_cert.sh": "running the noised extraction certification, 2026-09-03",
    "queue_training_slot_a.sh": "training extraction seeds 71 and 72, 2026-09-03",
    "queue_training_slot_b.sh": "training capture seed 71, the wedge-gated insert, capture 72, 2026-09-03",
}


def _shell_files() -> list[Path]:
    found: list[Path] = []
    for directory in SHELL_DIRECTORIES:
        if directory.exists():
            found.extend(sorted(directory.glob("*.sh")))
    return found


def _offenders(path: Path) -> list[str]:
    found: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if CLOBBERED.search(line):
            found.append(f"  {path.name}:{number}  {stripped[:100]}")
    return found


def test_there_are_shell_scripts_to_check() -> None:
    """Guard against the lint passing because it found nothing."""

    assert _shell_files(), "no shell scripts found; the lint below would pass vacuously"


@pytest.mark.parametrize("path", _shell_files(), ids=lambda p: p.name)
def test_status_is_captured_before_it_can_be_clobbered(path: Path) -> None:
    if path.name in IN_FLIGHT:
        pytest.skip(f"exempt while running: {IN_FLIGHT[path.name]}")

    offenders = _offenders(path)
    assert not offenders, (
        "$? read after a command substitution on the same line; the substitution "
        "runs first and overwrites the status. Capture it on the line before:\n"
        "  rc=$?\n"
        '  echo "[$(date +%H:%M:%S)] thing exit=$rc"\n' + "\n".join(offenders)
    )


def test_no_exemption_outlives_its_reason() -> None:
    """An exemption for a script that no longer needs one must be deleted.

    Without this the list becomes a place where the rule quietly stops applying.
    """

    stale = []
    for name in IN_FLIGHT:
        path = next((p for p in _shell_files() if p.name == name), None)
        if path is None:
            stale.append(f"{name} (no longer exists)")
        elif not _offenders(path):
            stale.append(f"{name} (already fixed)")
    assert not stale, "remove these from IN_FLIGHT; they do not need an exemption:\n  " + "\n  ".join(stale)
