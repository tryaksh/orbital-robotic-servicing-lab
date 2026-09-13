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

#: Campaign queues that are mid-run *right now*. Bash reads a script
#: incrementally and remembers a byte offset, so editing one while it executes
#: can make it run garbage; a queue with hours of GPU work left goes in here
#: until it finishes. Fix it once it does and delete it from this list -- the
#: test below fails if an entry no longer needs to be here, so an exemption
#: cannot quietly rot.
#:
#: Empty since 2026-09-13. The three entries added on 2026-09-03 -- the noised
#: extraction certification and the two training slots -- had finished long
#: before then, and the exemption outlived the run by ten days without anyone
#: noticing, because a skip is silent. The scripts are fixed and the entries are
#: gone. If this list is not empty, the rule is not being enforced everywhere.
IN_FLIGHT: dict[str, str] = {}


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
            # Not stale -- unreachable. `artifacts/` is gitignored except for
            # `artifacts/campaign/*.sh`, so the queues a clean checkout carries
            # are the shipped ones and any queue written ad hoc during a campaign
            # is not. Absence here means "cannot check", and reporting it as
            # "deleted" turned this guard into a red CI on every push. The cost
            # is that an exemption for a genuinely deleted script is only caught
            # on a machine that has the file.
            continue
        if not _offenders(path):
            stale.append(f"{name} (already fixed)")
    assert not stale, "remove these from IN_FLIGHT; they do not need an exemption:\n  " + "\n  ".join(stale)
