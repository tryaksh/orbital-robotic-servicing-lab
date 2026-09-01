"""Build ``evidence/MANIFEST.json``: one queryable index of every measurement.

Why this exists
---------------
``evidence/`` holds 160-plus reports and keeps superseded and failed runs on
purpose, because a before is what makes an after mean anything. The cost of that
policy is that an agent arriving cold cannot tell which file is a current number
without opening all of them -- roughly a megabyte of JSON to answer "what is the
chain's success rate". This writes the answer down once, mechanically, so the
question costs one small file instead.

Status is derived, never hand-authored:

``canonical``   listed in ``CANONICAL`` below -- the reports the current claims
                rest on. Changing that set is a deliberate act, so it is a
                literal here and ``tests/test_evidence_manifest.py`` holds it.
``retracted``   named in ``evidence/RETRACTED.md``. The number describes a
                system that has since changed; do not quote it.
``historical``  everything else: superseded runs, probes, sweeps and controls.
                Kept for the reasoning, not as a current figure.

CPU only. Reads JSON and Markdown; imports nothing from Isaac Lab and loads no
checkpoint weights, so it runs while the GPU is busy.

Usage::

    python scripts/build_evidence_manifest.py           # write the manifest
    python scripts/build_evidence_manifest.py --check   # fail if it is stale
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
MANIFEST = EVIDENCE / "MANIFEST.json"
RETRACTED = EVIDENCE / "RETRACTED.md"

#: The reports the current claims rest on. Each entry is (filename, what it is
#: the evidence *for*), and the second half is the part an agent needs: a
#: filename alone does not say which sentence in the README it holds up.
CANONICAL: tuple[tuple[str, str], ...] = (
    (
        "workflow_robot_carried_m130pin_guarded_certification.json",
        "Legacy supported-settle baseline: 97.92% over 96 episodes; it predates the independent robot-support release recheck.",
    ),
    (
        "workflow_robot_carried_release_recheck_v2_certification.json",
        "Preserved pre-retention strict chain: 17/24 (70.83%) after both robot-side supports release and a 0.70 s free-module recheck.",
    ),
    (
        "workflow_robot_carried_release_rack_retention_control_v1_certification.json",
        "Current-source paired no-rack control: 17/24 (70.83%), exactly reproducing the pre-retention baseline.",
    ),
    (
        "workflow_robot_carried_release_rack_retention_v1_certification.json",
        "Strict chain with visible destination retention: 22/24 (91.67%); below the unchanged 95% full-chain gate.",
    ),
    (
        "rack_retention_paired_v1.json",
        "Paired T14 result: +5/24; rack-only transfer succeeds in all 22/22 episodes that reach measured seating, while two fail upstream.",
    ),
    (
        "rack_retention_geometry_v1.json",
        "Simulator-free rack-pawl geometry, load rating, and source binding check.",
    ),
    (
        "workflow_robot_carried_release_hand_first_v3_certification.json",
        "Paired losing load-transfer arm: hand-first release scores 12/24 (50.00%) on identical seeds and states.",
    ),
    (
        "workflow_robot_carried_m130_guarded_certification.json",
        "The before, one session back: 96.88%. Preserved so the after means something.",
    ),
    (
        "workflow_robot_carried_relocate_certification.json",
        "The before, two sessions back: 31.25%.",
    ),
    (
        "robot_carried_full_chain_pin.json",
        "One end-to-end run of the certified chain, with per-policy checkpoint SHA-256.",
    ),
    (
        "grapple_grasp_v7m130_on_derived_rack_certification.json",
        "Capture skill on the derived rack: 85.69% pooled. Misses the 95% gate.",
    ),
    (
        "grapple_grasp_v7m130_c11065_certification.json",
        "Unchanged capture checkpoint re-measured on the derived rack: 86.90% over 9,009 episodes; still below gate.",
    ),
    (
        "grapple_extract_v18pin_certification.json",
        "Extraction skill, the checkpoint the chain runs: 87.75% pooled. Misses the gate.",
    ),
    (
        "grapple_extract_v18pin_c11065_certification.json",
        "Unchanged extraction checkpoint re-measured on the derived rack: 87.64% over 9,004 episodes; no clearance gain.",
    ),
    (
        "grapple_extract_v17m130_on_pin_criterion_certification.json",
        "The control that makes the extract comparison a comparison and not a re-baselining.",
    ),
    (
        "extract_attribution.json",
        "One change a row on one unchanged checkpoint: criterion, rack, reset, then epochs.",
    ),
    (
        "grapple_insert_v20chain_certification.json",
        "The learned insert skill: 0.00% over 1,536 episodes. Published negative result.",
    ),
    (
        "chain_robustness_sweep.json",
        "One variable at a time around the certified point: what breaks the chain first.",
    ),
    (
        "workcell_geometry_check.json",
        "Simulator-free: where the arm stands, what the channel admits, which sections fit.",
    ),
    (
        "insert_reset_bank.json",
        "The stations an insertion may start from, solved in closed form and gated.",
    ),
    (
        "robot_carried_rigid_mating_refuted.json",
        "Why the form lock has to soften for the mating stroke. A refutation, kept.",
    ),
    (
        "service_latch_clearance.json",
        "The form lock's clearances, derived from the measured gripper envelope.",
    ),
    (
        "fiducial_rgbd_flush_v2_seed283.json",
        "Current flush-tag negative qualification: 43.27% critical-bay detection against a 99% gate.",
    ),
    (
        "rgbd_strict_capture_gate_v2_seed5070.json",
        "One strict RGB-D negative run after capture-gated dropout propagation; it ended during extraction and claims no relocation.",
    ),
    (
        "full_chain_state_16_report.json",
        "The state-task chain over 16 environments, with the seating conditions itemised.",
    ),
    (
        "insert_attitude_diagnosis.json",
        "Why the insert skill does not seat, eight arms: not creep, not the reward, the rack.",
    ),
    (
        "destination_channel_geometry.json",
        "The destination bay as each entry point actually builds it. Read from the config, not the source.",
    ),
    (
        "insert_attitude_wall_moved.json",
        "The attitude floor moves with the channel throat: 56.03 -> 45.75 mrad, one checkpoint.",
    ),
    (
        "workflow_robot_carried_m130pin_guarded_c11065_certification.json",
        "Legacy supported-settle comparator on the derived rack: 97.92%, seed for seed identical to the 12.689 mm run.",
    ),
    (
        "grapple_insert_v24rack_certification.json",
        "The learned insert skill on the derived rack: 36.77% pooled over 3,000 episodes, from 0.00%.",
    ),
    (
        "workflow_robot_carried_insert_v24rack_chain_policy_certification.json",
        "The learned seating policy inside the chain: 0.00%, against 36.77% alone. The gap is the result.",
    ),
    (
        "seating_controller_head_to_head.json",
        "Which controller seats the module, decided arithmetically on the same rack and seeds.",
    ),
    (
        "insertion_conditioned_controller_v3.json",
        "Paired v24 and guarded insertion at all nine reset stations and real chain handoffs; every losing arm retained.",
    ),
    (
        "insert_depth_is_attitude.json",
        "What is left of the insert skill: depth is attitude one layer down, through 2c/theta.",
    ),
    (
        "robot_carried_full_chain_c11065.json",
        "One end-to-end run of that chain -- and the first report here whose source bindings all recover from git.",
    ),
    (
        "serviceability_boundary_validation_v2.json",
        "The current fail-closed boundary result: entry attitude supported; five dimensions unresolved or contradicted.",
    ),
)


def _summarise(report: dict) -> dict:
    """Lift the fields that answer the common questions without opening the file."""
    out: dict = {}
    for key in ("title", "evidence_type", "generated_utc", "task"):
        value = report.get(key)
        if isinstance(value, (str, int, float)):
            out[key] = value
    overall = report.get("overall")
    if isinstance(overall, dict):
        picked = {k: overall[k] for k in ("episodes", "successes", "success_rate") if k in overall}
        wilson = overall.get("success_rate_wilson_95")
        if isinstance(wilson, dict):
            picked["wilson_95"] = [wilson.get("low"), wilson.get("high")]
        if picked:
            out["overall"] = picked
    gate = report.get("gate")
    if isinstance(gate, dict) and "passed" in gate:
        out["gate_passed"] = gate["passed"]
    policy = report.get("policy")
    if isinstance(policy, dict):
        for key in ("policy_set_sha256", "checkpoint_sha256"):
            if isinstance(policy.get(key), str):
                out["policy_sha256"] = policy[key]
                break
    for key in ("policy_set_sha256", "checkpoint_sha256"):
        value = report.get(key)
        if isinstance(value, str):
            out.setdefault("policy_sha256", value)
        elif isinstance(value, dict):
            out.setdefault("checkpoints_sha256", value)
    return out


def retracted_names() -> set[str]:
    """Filenames named in the RETRACTED.md table.

    Parsed rather than duplicated, so the manifest cannot drift behind an entry
    someone added there and nowhere else.
    """
    if not RETRACTED.is_file():
        return set()
    body = RETRACTED.read_text(encoding="utf-8").split("## Not retracted", 1)[0]
    found: set[str] = set()
    for row in body.splitlines():
        if not row.lstrip().startswith("|"):
            continue
        first = row.split("|")[1] if row.count("|") > 1 else ""
        found.update(re.findall(r"`([a-z0-9_]+\.json)`", first))
    return found


def build() -> dict:
    canonical = dict(CANONICAL)
    retracted = retracted_names()
    missing = sorted(name for name in canonical if not (EVIDENCE / name).is_file())
    if missing:
        raise SystemExit(
            "CANONICAL names reports that do not exist, which would publish a claim with no "
            f"evidence behind it: {', '.join(missing)}"
        )

    # Grouped by status, canonical first, because the whole point is that
    # ``head`` on this file answers "what are the current numbers" without
    # paging through 137 superseded runs. Historical entries are deliberately
    # thin -- a title and a date is enough to decide whether to open one.
    groups: dict[str, dict[str, dict]] = {"canonical": {}, "retracted": {}, "historical": {}}
    for path in sorted(EVIDENCE.glob("*.json")):
        if path.name == MANIFEST.name:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            groups["historical"][path.name] = {"error": str(error)}
            continue
        entry = _summarise(report if isinstance(report, dict) else {})
        if path.name in canonical:
            entry["holds_up"] = canonical[path.name]
            groups["canonical"][path.name] = entry
        elif path.name in retracted:
            entry["holds_up"] = "Nothing. See evidence/RETRACTED.md before quoting."
            groups["retracted"][path.name] = entry
        else:
            groups["historical"][path.name] = {
                k: entry[k] for k in ("title", "generated_utc") if k in entry
            }

    counts = {status: len(entries) for status, entries in groups.items()}

    return {
        "title": "Evidence manifest",
        "what_this_is": (
            "A queryable index of every report in evidence/. Status is derived: canonical from "
            "the list in scripts/build_evidence_manifest.py, retracted from evidence/RETRACTED.md, "
            "historical otherwise. Quote canonical reports. Never quote a retracted one."
        ),
        "generated_by": "scripts/build_evidence_manifest.py",
        "checkpoints_live_outside_git": {
            "note": (
                "logs/ and checkpoints/ are gitignored, so a clone does not carry the weights any "
                "learned number depends on. A report whose checkpoint is unreachable can be read "
                "but not reproduced."
            ),
            "policy_checkpoints": "logs/rl_games/zero_g_blade_insertion_contact/<run>/nn/",
            "pose_head_checkpoints": "checkpoints/",
        },
        "counts": counts,
        "canonical": groups["canonical"],
        "retracted": groups["retracted"],
        "historical": groups["historical"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify evidence/MANIFEST.json.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the manifest on disk differs from what would be written.",
    )
    args = parser.parse_args()

    manifest = build()
    rendered = json.dumps(manifest, indent=2, sort_keys=False) + "\n"

    if args.check:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.is_file() else ""
        if current != rendered:
            print("evidence/MANIFEST.json is stale. Run: python scripts/build_evidence_manifest.py")
            return 1
        print(f"evidence/MANIFEST.json is current: {manifest['counts']}")
        return 0

    MANIFEST.write_text(rendered, encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
