"""Fail if a workflow loaded a policy that nothing in ``evidence/`` certifies.

This mechanises the rule that cost this project a whole session. ``evidence/``
named grasp v2, extract v2 and insert v3 while ``run_workflow_demo.py`` loaded
v3, v4 and v5, so every success rate quoted about the demonstration described a
superseded policy. Nothing caught it, because the check was a human remembering
to compare two filenames.

Filenames are the wrong thing to compare. ``run_workflow_demo.py`` already
records the SHA-256 of every checkpoint it loads, and ``aggregate_evaluation.py``
records the SHA-256 of the checkpoint each certification describes, so the
question "is this number about this policy" has an exact answer that no naming
convention can drift away from.

Usage::

    python scripts/check_evidence_currency.py artifacts/workflow_cert/*_report.json

Each report is a run of the chained workflow. Every checkpoint it loaded must
have at least one file in ``evidence/`` whose ``policy.checkpoint_sha256`` is
the same hash. Exit status is 1 if any does not, so this is usable as a gate.

CPU only. It imports nothing from Isaac Lab and reads no checkpoint weights, so
it runs while the GPU is busy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Where ``aggregate_evaluation.py`` writes the SHA-256 of the policy a report
#: describes. A file without this key is not a certification and is skipped.
CERTIFICATION_SHA_PATH = ("policy", "checkpoint_sha256")


def _normalize(digest: str) -> str:
    return digest.strip().upper()


def combined_policy_sha256(digests: dict[str, str], order: tuple[str, ...]) -> str:
    """Reproduce ``run_workflow_demo.py``'s policy-set hash.

    It hashes the concatenated per-policy hex digests in phase order, so a
    report that names three policies and a set hash can be checked against
    itself. Kept here rather than imported because that module cannot be
    imported without a simulator.
    """

    return hashlib.sha256("".join(digests[name] for name in order).encode()).hexdigest().upper()


def load_certified_digests(evidence_dir: Path) -> dict[str, list[str]]:
    """Map every certified checkpoint hash to the evidence files claiming it."""

    certified: dict[str, list[str]] = {}
    for path in sorted(evidence_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        block = report.get(CERTIFICATION_SHA_PATH[0])
        if not isinstance(block, dict):
            continue
        digest = block.get(CERTIFICATION_SHA_PATH[1])
        if isinstance(digest, str) and digest:
            certified.setdefault(_normalize(digest), []).append(path.name)
    return certified


def report_policies(report: dict) -> dict[str, dict[str, str]]:
    """Return ``{phase: {"sha256": ..., "checkpoint": ...}}`` for one report.

    Two shapes reach here. A per-run workflow report carries one hash per
    phase, which is the case this tool exists for. A pooled certification
    carries a single hash under ``policy``, which for a chained workflow is the
    *set* hash and for a single skill is that skill's own; both are returned
    under the phase name ``policy`` and resolved the same way.
    """

    policies: dict[str, dict[str, str]] = {}
    digests = report.get("checkpoint_sha256")
    paths = report.get("checkpoints") if isinstance(report.get("checkpoints"), dict) else {}
    if isinstance(digests, dict):
        for phase, digest in digests.items():
            policies[phase] = {"sha256": _normalize(str(digest)), "checkpoint": str(paths.get(phase, "unknown"))}
        return policies
    block = report.get("policy")
    if isinstance(block, dict) and isinstance(block.get("checkpoint_sha256"), str):
        policies["policy"] = {
            "sha256": _normalize(block["checkpoint_sha256"]),
            "checkpoint": str(block.get("checkpoint", "unknown")),
        }
    return policies


def check_report(report_path: Path, certified: dict[str, list[str]]) -> dict:
    """Resolve one report's policies against the certified hashes."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    policies = report_policies(report)
    findings = []
    for phase, policy in sorted(policies.items()):
        matches = certified.get(policy["sha256"], [])
        findings.append(
            {
                "phase": phase,
                "checkpoint": Path(policy["checkpoint"].replace("\\", "/")).name,
                "sha256": policy["sha256"],
                "certified_by": matches,
                "certified": bool(matches),
            }
        )

    set_hash_consistent = None
    declared_set = report.get("policy_set_sha256")
    digests = report.get("checkpoint_sha256")
    if isinstance(declared_set, str) and isinstance(digests, dict):
        order = tuple(digests)
        set_hash_consistent = _normalize(declared_set) == combined_policy_sha256(
            {name: str(value) for name, value in digests.items()}, order
        )

    return {
        "report": str(report_path),
        "workflow": report.get("workflow"),
        "policies": findings,
        "uncertified": [finding["phase"] for finding in findings if not finding["certified"]],
        "policy_set_sha256_consistent": set_hash_consistent,
        "passed": bool(findings)
        and not any(not finding["certified"] for finding in findings)
        and set_hash_consistent is not False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, nargs="+", help="Workflow or certification reports to audit.")
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence", help="Directory of certifications.")
    parser.add_argument("--json", type=Path, default=None, help="Write the full audit here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    certified = load_certified_digests(args.evidence)
    if not certified:
        print(f"FAIL  no certification in {args.evidence} records a checkpoint hash")
        return 1

    results = []
    for report_path in args.reports:
        if not report_path.is_file():
            print(f"FAIL  {report_path}: no such report")
            results.append({"report": str(report_path), "passed": False, "policies": []})
            continue
        result = check_report(report_path, certified)
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'}  {report_path}")
        if not result["policies"]:
            print("      report records no checkpoint hash at all, so nothing can be traced")
        for finding in result["policies"]:
            mark = "ok " if finding["certified"] else "!! "
            certifiers = ", ".join(finding["certified_by"]) if finding["certified"] else "NOTHING IN evidence/"
            print(f"      {mark}{finding['phase']:<8} {finding['checkpoint']}")
            print(f"          {finding['sha256'][:16]}...  <- {certifiers}")
        if result["policy_set_sha256_consistent"] is False:
            print("      !! policy_set_sha256 does not match the policies the report names")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"reports": results}, indent=2), encoding="utf-8")

    failed = [result for result in results if not result["passed"]]
    if failed:
        print(
            f"\n{len(failed)} of {len(results)} reports quote a policy that no file in "
            f"{args.evidence} certifies. Certify the checkpoint the workflow loads, or load the "
            "checkpoint the certification describes."
        )
        return 1
    print(f"\nAll {len(results)} reports load only policies certified in {args.evidence}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
