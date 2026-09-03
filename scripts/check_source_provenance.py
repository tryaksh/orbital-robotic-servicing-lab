"""Can the source that produced a report still be recovered from git?

Several reports record ``runtime_source_bindings``: the SHA-256 of each source
file **as it was on disk when the run happened**. That is a strong provenance
record and nothing was checking it, so it was answering a question nobody asked.

It answers a sharper question than ``check_criterion_currency.py`` does. That
script compares timestamps and says "this evidence predates a commit that could
have moved its criterion" -- a prompt, and often a false one, because a session
runs its measurements and commits the code afterwards. This compares *content*,
so its answer is not a prompt:

``recovered``   the recorded hash matches the file at some commit. The run is
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

Line endings are handled: this repository is checked out with ``core.autocrlf``
true, so a run hashes CRLF bytes while git stores LF. Blobs are converted before
hashing, and the conversion is proved on every file that does match.

CPU only. Reads JSON and ``git show``; imports nothing from Isaac Lab.

Usage::

    python scripts/check_source_provenance.py            # every report
    python scripts/check_source_provenance.py --depth 60 # search further back
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


def _commits(depth: int) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "log", f"-{depth}", "--format=%h\t%s"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
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


def classify(path: str, recorded: str, commits: list[tuple[str, str]]) -> dict:
    """Where, if anywhere, do these bytes still exist."""
    on_disk = ROOT / path
    if on_disk.is_file() and hashlib.sha256(on_disk.read_bytes()).hexdigest() == recorded:
        head = _blob("HEAD", path)
        if head is not None and _matches(head, recorded):
            return {"path": path, "state": "recovered", "commit": "HEAD"}
        return {"path": path, "state": "working"}
    for short, subject in commits:
        blob = _blob(short, path)
        if blob is None:
            continue
        if _matches(blob, recorded):
            return {"path": path, "state": "recovered", "commit": short, "subject": subject}
    return {"path": path, "state": "lost"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify recorded source hashes against git.")
    parser.add_argument("reports", nargs="*", help="Reports to check (default: all of evidence/).")
    parser.add_argument("--depth", type=int, default=40, help="How many commits back to search.")
    parser.add_argument("--json", type=Path, default=None, help="Write the result as JSON.")
    parser.add_argument(
        "--fail_on_lost",
        action="store_true",
        help="Exit non-zero if any binding is lost. Off by default: existing lost bindings are "
        "a recorded fact, not a regression to block on.",
    )
    args = parser.parse_args()

    paths = [Path(name) for name in args.reports] or sorted(EVIDENCE.glob("*.json"))
    commits = _commits(args.depth)

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
        rows = [classify(entry["path"], entry["sha256"], commits) for entry in bindings]
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
