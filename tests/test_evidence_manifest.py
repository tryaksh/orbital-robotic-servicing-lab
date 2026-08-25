"""The manifest must stay true, because it is what an agent reads instead of evidence/.

``evidence/MANIFEST.json`` exists so that arriving cold costs one small file
rather than a megabyte of JSON. That trade is only safe while the manifest is
current, so this holds three things:

* it is not stale -- regenerating it produces the file on disk, byte for byte;
* every report it calls canonical exists, because a canonical entry is a claim;
* nothing is both canonical and retracted, which would be the manifest telling
  an agent to quote a number that RETRACTED.md says describes a system that has
  since changed.

Source-level and CPU-only, like every check this project expects to keep
running: no simulator, no checkpoint weights, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_evidence_manifest as manifest_builder  # noqa: E402

MANIFEST = ROOT / "evidence" / "MANIFEST.json"


def test_the_manifest_on_disk_is_what_the_builder_would_write() -> None:
    """A stale manifest is worse than none: it is confidently wrong."""
    rendered = json.dumps(manifest_builder.build(), indent=2, sort_keys=False) + "\n"
    assert MANIFEST.read_text(encoding="utf-8") == rendered, (
        "evidence/MANIFEST.json is stale. Run: python scripts/build_evidence_manifest.py"
    )


def test_every_canonical_report_exists() -> None:
    """A canonical entry names the evidence behind a published claim."""
    for name, _ in manifest_builder.CANONICAL:
        assert (ROOT / "evidence" / name).is_file(), f"canonical evidence is missing: {name}"


def test_no_report_is_both_canonical_and_retracted() -> None:
    """The two lists disagreeing is the one failure that would mislead directly."""
    canonical = {name for name, _ in manifest_builder.CANONICAL}
    overlap = sorted(canonical & manifest_builder.retracted_names())
    assert not overlap, f"canonical and retracted both claim: {', '.join(overlap)}"


def test_every_canonical_entry_says_what_it_holds_up() -> None:
    """A filename does not tell an agent which sentence rests on it."""
    for name, holds_up in manifest_builder.CANONICAL:
        assert holds_up.strip(), f"{name} is canonical but says nothing about what it supports"


def test_the_manifest_states_that_checkpoints_are_not_in_the_clone() -> None:
    """The weights behind every learned number are gitignored, and that has to be said.

    A clone carries the reports but not ``logs/`` or ``checkpoints/``, so a
    learned rate here is readable and not reproducible. Claiming a capability
    whose checkpoint is unreachable is the specific mistake this pins.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    outside = manifest["checkpoints_live_outside_git"]
    assert "gitignored" in outside["note"]
    assert outside["policy_checkpoints"].startswith("logs/rl_games/")
    assert outside["pose_head_checkpoints"] == "checkpoints/"
