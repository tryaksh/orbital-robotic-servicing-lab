#!/usr/bin/env python3
"""Correct the lead-in label that ``play.py`` misread, without re-measuring.

``play.py`` read the slot flares' ``collision_props.collision_enabled`` with
``bool()``.  That field is Isaac Lab's tri-state: ``None`` means "leave the
authored USD alone", and nothing in ``assets.py`` ever sets it, so the flares
are collidable in every run this project has ever made.  ``bool(None)`` is
``False``, so every grapple-pin evaluation recorded ``lead_in_present: false``,
which fed ``out_of_distribution: true``, which stamped the report
``evidence_type: simulation_capability_envelope`` with ``gate.applies: false``.

**No success rate is affected.** The simulation ran with the flares present, the
episodes are the episodes, and the pooled arithmetic is unchanged.  Only the
label is wrong, so the fix is to relabel the archived rows and the reports
derived from them rather than to spend GPU hours reproducing identical numbers.

Two passes, and both refuse to guess:

``--rows``
    Rewrites the ``stress`` block inside archived ``.npz`` files so a future
    re-aggregation is right.  It only touches a file whose sole reason for being
    out of distribution was this misread; a run that widened the reset noise, the
    mass range or the belief bias keeps its label, and so would a genuine
    ``--no_lead_in`` probe.

``--reports``
    Applies the same correction to the JSON under ``evidence/``, in place and
    **keeping each report's original ``generated_utc``** -- the measurement is
    not new and must not read as new to ``scripts/check_evidence_currency.py``.
    Each patched report gains a ``label_correction`` block saying what changed.

``--verify NAME`` re-aggregates one report from the corrected rows and checks the
patch reproduces it field for field, so the in-place edit is a proof rather than
an assumption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CORRECTION_UTC = "2026-08-18"
MARKER = "lead_in_label_corrected_utc"
NOTE = (
    "play.py read the flares' tri-state collision_enabled with bool(), so Isaac Lab's "
    "None ('leave as authored', i.e. collidable) recorded as lead_in_present: false and "
    "marked the run out of distribution. The flares were present. Rates are unchanged; "
    "only the label was wrong."
)


def _stress_of(metadata: dict) -> dict:
    return metadata.get("stress") or {}


def _other_ood_reasons(stress: dict) -> list[str]:
    """Every reason this run is out of distribution *apart from* the misread.

    Mirrors ``play.py._apply_stress`` so the two cannot drift: if that function
    grows a reason, this one has to grow it too or a real stress run would be
    silently relabelled as certification.
    """

    reasons = []
    if float(stress.get("pose_noise_scale") or 1.0) > 1.0:
        reasons.append("pose_noise_scale")
    trained_mass = stress.get("trained_blade_mass_kg")
    mass = stress.get("blade_mass_range_kg")
    if mass and trained_mass and (mass[0] < trained_mass[0] or mass[1] > trained_mass[1]):
        reasons.append("blade_mass_range_kg")
    ceiling = stress.get("trained_belief_bias_ceiling_mm")
    bias = stress.get("belief_bias_mm")
    if bias is not None and ceiling is not None and bias > ceiling + 1.0e-9:
        reasons.append("belief_bias_mm")
    return reasons


def _relabel_rows(paths: list[Path], apply: bool) -> tuple[int, int, int]:
    corrected = skipped = untouched = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            if "metadata" not in archive.files:
                untouched += 1
                continue
            payload = {name: archive[name] for name in archive.files}
            metadata = json.loads(str(payload["metadata"]))
        stress = _stress_of(metadata)
        if stress.get("lead_in_present") is not False or MARKER in stress:
            untouched += 1
            continue
        others = _other_ood_reasons(stress)
        if others:
            # A real stress run. Its lead-in field is still wrong, but the run is
            # genuinely out of distribution, so the gate must stay switched off.
            stress["lead_in_present"] = True
            stress[MARKER] = CORRECTION_UTC
            stress["lead_in_label_note"] = f"{NOTE} Still out of distribution via: {', '.join(others)}."
            skipped += 1
        else:
            stress["lead_in_present"] = True
            stress["out_of_distribution"] = False
            stress[MARKER] = CORRECTION_UTC
            stress["lead_in_label_note"] = NOTE
            corrected += 1
        if apply:
            payload["metadata"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(path, **payload)
    return corrected, skipped, untouched


def _index_rows(paths: list[Path]) -> dict[str, list[dict]]:
    """Every archived run, grouped by the checkpoint it evaluated.

    Deriving the decision from the rows is the whole point: a report alone cannot
    distinguish this misread from a deliberate sweep, and four genuine envelope
    reports would be relabelled as certifications if it tried.

    The grouping is by checkpoint rather than by an attempt to reconstruct which
    files fed which report, and the rule that uses it is correspondingly
    conservative -- see ``_patch_report``. Matching runs to reports exactly would
    mean reverse-engineering filename conventions that have changed four times,
    and getting that wrong silently is precisely the failure being fixed.
    """

    index: dict[str, list[dict]] = {}
    for path in paths:
        try:
            with np.load(path, allow_pickle=True) as archive:
                if "metadata" not in archive.files:
                    continue
                metadata = json.loads(str(archive["metadata"]))
        except Exception:
            continue
        sha = str(metadata.get("checkpoint_sha256") or "")
        if sha:
            index.setdefault(sha, []).append(metadata)
    return index


def _patch_report(path: Path, apply: bool, rows_index: dict[str, list[dict]]) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "label_correction" in report:
        return "already corrected"
    if not report.get("out_of_distribution"):
        return "not flagged"
    gate = report.get("gate") or {}
    if gate.get("note") != "capability envelope sweep; the gate certifies in-distribution runs only":
        return "flagged for another reason"

    # The rows decide, not the report, and the rule is deliberately conservative:
    # relabel only when *every* archived run that ever evaluated this checkpoint
    # carried the misread and nothing else. A checkpoint that also appears in a
    # real sweep is left alone rather than guessed at, which is why the four
    # genuine envelope reports keep their gate switched off.
    sha = ((report.get("policy") or {}).get("checkpoint_sha256")) or ""
    runs = rows_index.get(sha)
    if not runs:
        return "rows not found"
    stresses = [_stress_of(run) for run in runs]
    if any(_other_ood_reasons(stress) for stress in stresses):
        return "checkpoint also appears in a genuine stress sweep"
    if not all(MARKER in stress or stress.get("lead_in_present") is False for stress in stresses):
        return "rows do not carry the misread"

    report["out_of_distribution"] = False
    report["evidence_type"] = "simulation_only"
    gate.pop("applies", None)
    gate.pop("note", None)
    # ``_stress_label`` appends this suffix when the flares read as absent.
    by_stress = report.get("by_stress")
    if isinstance(by_stress, dict):
        report["by_stress"] = {key.replace("_no_lead_in", ""): value for key, value in by_stress.items()}
    report["label_correction"] = {
        "corrected_utc": CORRECTION_UTC,
        "field": "stress.lead_in_present",
        "was": False,
        "now": True,
        "why": NOTE,
        "changed": [
            "out_of_distribution",
            "evidence_type",
            "gate.applies",
            "gate.note",
        ],
        "unchanged": "every episode count, success rate, interval and terminal metric",
        "generated_utc_preserved": report.get("generated_utc"),
    }
    if apply:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return "corrected"


def _verify(name: str) -> int:
    """Re-aggregate one report from the corrected rows and diff against the patch."""

    report_path = REPO / "evidence" / f"grapple_{name}_certification.json"
    rows = sorted((REPO / "artifacts" / "certify").glob(f"{name}_s*_seed*.npz"))
    if not report_path.exists() or not rows:
        print(f"[VERIFY] cannot verify {name}: report={report_path.exists()} rows={len(rows)}")
        return 1
    original = json.loads(report_path.read_text(encoding="utf-8"))
    fresh_path = REPO / "artifacts" / "certify" / f"{name}_relabel_verify.json"
    command = [
        sys.executable,
        str(REPO / "scripts" / "aggregate_evaluation.py"),
        "--episodes",
        *[str(row) for row in rows],
        "--output",
        str(fresh_path),
        "--title",
        original["title"],
        "--scope",
        *original.get("scope_and_limitations", []),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # A non-zero exit is how the aggregator reports a *failed gate*, which several
    # of these reports legitimately have. Only a missing output file is an error.
    if not fresh_path.exists():
        print(f"[VERIFY] aggregation produced no file (exit {result.returncode}):\n{result.stdout}\n{result.stderr}")
        return 1
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    patched = dict(original)
    patched.pop("label_correction", None)
    ignore = {"generated_utc"}
    differences = [
        key
        for key in sorted(set(fresh) | set(patched))
        if key not in ignore and fresh.get(key) != patched.get(key)
    ]
    if differences:
        print(f"[VERIFY] {name}: DIFFERS on {differences}")
        for key in differences:
            print(f"  fresh   {key} = {json.dumps(fresh.get(key))[:300]}")
            print(f"  patched {key} = {json.dumps(patched.get(key))[:300]}")
        return 1
    print(f"[VERIFY] {name}: the in-place patch reproduces a full re-aggregation exactly")
    fresh_path.unlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", action="store_true", help="Correct the stress block inside archived .npz rows.")
    parser.add_argument("--reports", action="store_true", help="Correct the JSON reports under evidence/.")
    parser.add_argument("--verify", default=None, help="Report stem to re-aggregate and diff, e.g. extract_v14reset.")
    parser.add_argument("--apply", action="store_true", help="Write the changes. Without it this is a dry run.")
    args = parser.parse_args()

    if args.rows:
        paths = sorted((REPO / "artifacts").rglob("*.npz"))
        corrected, kept_ood, untouched = _relabel_rows(paths, args.apply)
        print(
            f"[ROWS] {corrected} relabelled in-distribution, {kept_ood} relabelled but still "
            f"out of distribution for another reason, {untouched} untouched, of {len(paths)} files"
        )
    if args.reports:
        rows_index = _index_rows(sorted((REPO / "artifacts").rglob("*.npz")))
        outcomes: dict[str, int] = {}
        for path in sorted((REPO / "evidence").glob("*.json")):
            outcome = _patch_report(path, args.apply, rows_index)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if outcome == "corrected":
                print(f"[REPORT] corrected {path.name}")
        print(f"[REPORTS] {outcomes}")
    if args.verify:
        return _verify(args.verify)
    if not args.apply and (args.rows or args.reports):
        print("[DRY RUN] nothing written; pass --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
