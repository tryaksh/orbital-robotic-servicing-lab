"""Can the source that produced a report still be recovered from git?

Several reports record ``runtime_source_bindings``: the SHA-256 of each source
file **as it was on disk when the run happened**. That is a strong provenance
record and nothing was checking it, so it was answering a question nobody asked.

It answers a sharper question than ``check_criterion_currency.py`` does. That
script compares timestamps and says "this evidence predates a commit that could
have moved its criterion" -- a prompt, and often a false one, because a session
runs its measurements and commits the code afterwards. This compares *content*,
so its answer is not a prompt:

``recovered``   the recorded hash matches the file at some commit reachable from
                any ref here -- ``HEAD``, a branch or a tag. The run is
                reproducible: check that commit out and the source is the source.
``working``     the recorded hash matches the working tree but no commit. The
                run happened on uncommitted state that is still on disk.
``lost``        the recorded hash matches neither. **The bytes that produced the
                number exist nowhere.** The run cannot be reproduced and nobody
                can say what differed.

``lost`` is not a claim that the number is wrong. The run happened and the
episodes are the episodes. It is a claim that the relationship between the
published number and the committed code is unverified, which is a different
thing and has to be said out loud rather than assumed benign.

Line endings are handled, and handling them wrongly is what made most of this
look unrecoverable. The repository is checked out with ``core.autocrlf`` true, so
a run usually hashes CRLF bytes while git stores LF -- but a file a tool wrote
with LF and left in place is byte-identical to the blob. Both renderings of a
blob are therefore accepted, because git treats them as the same content. The
commit walk also runs when the working tree still matches, so a report bound to
an earlier commit is reported against that commit rather than as ``working``.

CPU only. Reads JSON and ``git show``; imports nothing from Isaac Lab.

Usage::

    python scripts/check_source_provenance.py             # every report, every ref
    python scripts/check_source_provenance.py --head_only # HEAD's history alone
    python scripts/check_source_provenance.py --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def _find(node: object, key: str):
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                return value
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _as_checked_out(blob: bytes) -> tuple[bytes, ...]:
    """Every rendering of one blob a working tree here can legitimately hold.

    **This used to assume one, and that assumption is what made most of T0 look
    unrecoverable.** Git stores LF; with ``core.autocrlf=true`` a Windows
    checkout holds CRLF, and the runtime hashes the bytes it read from disk. So
    the old single CRLF expansion matched every file that had been checked out
    and missed every file a tool had written with LF and left there --
    ``rack_retention.py``, ``servicing_camera.py`` and ``provenance.py`` in the
    most recent RGB-D report, all three of which are byte-identical to the
    committed blob and were reported lost anyway.

    Git treats the two renderings as the same content under ``autocrlf``, so a
    recorded hash matching either one identifies the same source. Returning both
    is the fix; comparing normalised text on both sides is not, because the
    recorded hash is of raw bytes and cannot be re-normalised after the fact.
    """

    normalised = blob.replace(b"\r\n", b"\n")
    return (normalised, normalised.replace(b"\n", b"\r\n"))


def _matches(blob: bytes, recorded: str) -> bool:
    return any(
        hashlib.sha256(rendering).hexdigest() == recorded for rendering in _as_checked_out(blob)
    )


def _history(path: str, depth: int, all_refs: bool) -> list[tuple[str, str]]:
    """The commits that changed this one path, newest first, and their subjects.

    **Only the commits that touched the path, because the blob cannot differ at
    any other.** Walking whole history and running ``git show`` per commit per
    file is the same answer at a hundred times the cost: with a thousand commits
    across every ref and four bindings a report, that is a hundred thousand git
    invocations and the check stops being something you run before a push.

    ``--all`` here is every ref in the repository, and that breadth is what
    recovered the 2026-09-04 campaign. Its commits live only in
    ``archive/assembly-recovery-training``, which is a tag here, so
    ``git checkout`` reaches the bytes and the run *is* reproducible. Reporting
    those bindings as lost said the opposite. The interleaving problem that makes
    ``git log --all`` a bad idea for a whole-history walk does not apply once the
    walk is path-scoped: a path is touched by few enough commits that the depth
    limit is not reached.
    """

    command = ["git", "log", f"-{depth}", "--format=%h\t%s"]
    if all_refs:
        command.append("--all")
    command.extend(["--", path])
    out = subprocess.run(command, capture_output=True, text=True, cwd=ROOT).stdout
    rows = []
    for line in out.splitlines():
        short, _, subject = line.partition("\t")
        if short:
            rows.append((short, subject))
    return rows


def _blob(commit: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], capture_output=True, cwd=ROOT
    )
    return result.stdout if result.returncode == 0 else None


def classify(path: str, recorded: str, depth: int, all_refs: bool) -> dict:
    """Where, if anywhere, do these bytes still exist."""
    on_disk = ROOT / path
    still_on_disk = (
        on_disk.is_file() and hashlib.sha256(on_disk.read_bytes()).hexdigest() == recorded
    )
    if still_on_disk:
        head = _blob("HEAD", path)
        if head is not None and _matches(head, recorded):
            return {"path": path, "state": "recovered", "commit": "HEAD"}
    # **Falling through here is the point.** This used to return "working" the
    # moment the file on disk matched and HEAD did not, and never looked at
    # history -- so a report produced on a verified-clean tree a few commits ago
    # was reported unrecoverable purely because HEAD had moved since. The commit
    # walk below is what answers the question; "working" is only the right
    # answer when it finds nothing.
    for short, subject in _history(path, depth, all_refs):
        blob = _blob(short, path)
        if blob is None:
            continue
        if _matches(blob, recorded):
            return {"path": path, "state": "recovered", "commit": short, "subject": subject}
    return {"path": path, "state": "working" if still_on_disk else "lost"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify recorded source hashes against git.")
    parser.add_argument("reports", nargs="*", help="Reports to check (default: all of evidence/).")
    parser.add_argument(
        "--depth",
        type=int,
        default=200,
        help="How many commits that touched a path to search back through. Only commits that "
        "changed the file are candidates, so this is a deep window rather than a costly one.",
    )
    parser.add_argument(
        "--head_only",
        action="store_true",
        help="Search only HEAD's history. The default searches every tag and branch too, because a "
        "commit reachable from any ref in this repository is one a reader can check out.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the result as JSON.")
    parser.add_argument(
        "--fail_on_lost",
        action="store_true",
        help="Exit non-zero if any binding is lost. Off by default: existing lost bindings are "
        "a recorded fact, not a regression to block on.",
    )
    args = parser.parse_args()

    paths = [Path(name) for name in args.reports] or sorted(EVIDENCE.glob("*.json"))
    all_refs = not args.head_only

    summary: dict[str, dict] = {}
    any_lost = False
    for report_path in paths:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bindings = _find(report, "runtime_source_bindings")
        if not bindings:
            continue
        rows = [
            classify(entry["path"], entry["sha256"], args.depth, all_refs) for entry in bindings
        ]
        states = {row["state"] for row in rows}
        verdict = "lost" if "lost" in states else ("working" if "working" in states else "recovered")
        any_lost |= verdict == "lost"
        summary[report_path.name] = {"verdict": verdict, "bindings": rows}

        print(f"{verdict.upper():<10} {report_path.name}")
        for row in rows:
            where = row.get("commit", "")
            note = f"  {where} {row.get('subject', '')}".rstrip() if where else ""
            if row["state"] != "recovered":
                print(f"           [{row['state']}] {row['path']}{note}")

    total = len(summary)
    lost = sum(1 for value in summary.values() if value["verdict"] == "lost")
    print()
    print(f"{total} reports carry source bindings; {lost} cannot be fully recovered from git.")
    if lost:
        print(
            "A lost binding does not make a number wrong -- the run happened. It means the run "
            "is not reproducible from this repository and nobody can say what differed. "
            "Re-run the affected certification on committed code to restore it."
        )

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if (args.fail_on_lost and any_lost) else 0


if __name__ == "__main__":
    raise SystemExit(main())
