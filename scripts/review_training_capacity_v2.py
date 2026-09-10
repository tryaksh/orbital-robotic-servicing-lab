"""Capacity reviewer v2: compare frozen runtime dependencies, retaining the v1 CPU failure."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.protocol import sha256  # noqa: E402
from scripts.review_training_capacity_v1 import adoption_decision, verify_run  # noqa: E402


def runtime_source_hashes(archive_path):
    """The predeclared protocol supplies the runtime dependency set, including __init__."""
    with zipfile.ZipFile(archive_path) as archive:
        protocol = json.loads(archive.read("configs/protocol_v4.json"))
        frozen = protocol["behavior_source_sha256"]
        required = set(frozen) | {
            "scripts/profile_training_capacity_v1.py", "scripts/run_training_capacity_v1.py",
            "configs/training_capacity_v1.json", "configs/protocol_v4.json", "configs/study.json",
            "src/assembly_recovery/__init__.py",
        }
        if "reward_correction" in protocol:
            required.add(protocol["reward_correction"]["path"])
        hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(required)}
        for name, expected in frozen.items():
            if hashes[name] != expected:
                raise ValueError("Captured frozen runtime dependency changed: " + name)
        nodes = ast.walk(ast.parse(archive.read("src/assembly_recovery/__init__.py")))
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in nodes):
            raise ValueError("Package __init__ imports additional modules; audit its runtime dependency closure")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-failure", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence and choose a new versioned output path")
    result = {"schema": 1, "version": "training_capacity_review_v2", "research_result": False,
        "input_manifests": [str(args.baseline.resolve()), str(args.candidate.resolve())],
        "corrected_cpu_verifier_failure": {"path": args.previous_failure.as_posix(), "sha256": sha256(args.previous_failure),
            "reason": "Reviewer v1 compared every archived assembly_recovery module; unrelated recovery_teaching_v1.py changed during independent development. Reviewer v2 compares the predeclared protocol's exact runtime dependencies, including package __init__, plus the new capacity worker/launcher/config. Both source archives and all original artifacts remain independently verified.",
            "additional_simulator_transitions": 0},
        "verifier_source_sha256": {name: sha256(ROOT / name) for name in (
            "scripts/review_training_capacity_v1.py", "scripts/review_training_capacity_v2.py",
            "scripts/profile_training_capacity_v1.py", "scripts/run_training_capacity_v1.py")},
        "scope_and_limitations": [
            "Representative engineering capacity measurement with genuine PPO; no policy-score or equal-cost curriculum comparison.",
            "2048 performs twice the requested jobs and charged work. Active samples and optimizer steps are measured, not matched.",
            "Two fresh-policy cohorts do not establish mature-policy utilization or multi-hour stability.",
            "Physics sensitivity remains unresolved; adopting an environment count does not pass a physics or recovery gate.",
            "Remeasure representative capacity if the physical/contact implementation changes. No 4096 probe is required before the main experiment.",
            "No final tests, camera perception, hardware transfer or force-certified safety claim."]}
    try:
        baseline, candidate = verify_run(args.baseline), verify_run(args.candidate)
        if baseline["specification"]["num_envs"] != 1024 or candidate["specification"]["num_envs"] != 2048:
            raise ValueError("Expected the registered 1024 baseline and 2048 candidate")
        if baseline["registration_sha256"] != candidate["registration_sha256"]:
            raise ValueError("The registered capacity specifications differ")
        runtime_left = runtime_source_hashes(args.baseline.parent / "source.zip")
        runtime_right = runtime_source_hashes(args.candidate.parent / "source.zip")
        if runtime_left != runtime_right:
            raise ValueError("An actual frozen capacity runtime dependency changed between runs")
        archived_left, archived_right = baseline.pop("behavior_source_sha256"), candidate.pop("behavior_source_sha256")
        unrelated = [{"path": name, "baseline_sha256": archived_left.get(name), "candidate_sha256": archived_right.get(name)}
                     for name in sorted(set(archived_left) | set(archived_right))
                     if archived_left.get(name) != archived_right.get(name)]
        if any(item["path"] in runtime_left for item in unrelated):
            raise ValueError("A changed module overlaps the verified runtime dependency set")
        result.update(runs=[baseline, candidate], verified_runtime_source_sha256=runtime_left,
                      unrelated_archived_source_changes=unrelated,
                      decision=adoption_decision(baseline, candidate),
                      total_charged_development_transitions=sum(r["measured_cost"]["charged_control_equivalent_transitions"] for r in (baseline, candidate)))
        result["status"] = "verified" if baseline["status"] == candidate["status"] == "verified" else "failed_verification"
        result["next_action"] = "Use the verified 2048 capacity for the next bounded experiment under the same runtime if the adoption checks pass. Prioritize the physics-valid research experiment over a 4096 probe; reprofile after any material physics change."
    except Exception as exc:
        result.update(status="failed_verification", error=f"{type(exc).__name__}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "decision": result.get("decision"), "total_charged": result.get("total_charged_development_transitions")}))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
