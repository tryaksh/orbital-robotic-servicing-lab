"""Versioned CPU-only correction for sparse retry overrides versus resolved defaults."""
from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import sha256  # noqa: E402
from assembly_recovery.retry_controller import ActorRetryController, RetrySettings  # noqa: E402
from scripts import review_contact_impact_v1 as original  # noqa: E402


def known_settings_verifier_failure(manifest, report, saved_verification):
    """A preserved CPU-only mistake may not excuse process or physical failure."""
    return (
        manifest.get("status") == "verification_failed"
        and report.get("status") == "completed"
        and manifest.get("returncode") == 0
        and manifest.get("timed_out") is False
        and not manifest.get("resource_guard_stop")
        and saved_verification.get("status") == "check_failed"
        and saved_verification.get("failed_checks") == ["frozen_script_settings"]
        and {key for key, passed in saved_verification.get("checks", {}).items() if not passed} == {"frozen_script_settings"}
        and saved_verification.get("checks", {}).get("process_completed") is True
        and {key for key, passed in manifest.get("checks", {}).items() if not passed} == {"frozen_script_settings"}
    )


def verify_single(run: Path):
    try:
        return _verify_single(Path(run))
    except Exception as exc:
        return {"status": "verification_error", "checks": {"corrected_verifier_completed": False},
                "failed_checks": ["corrected_verifier_completed"], "error": f"{type(exc).__name__}: {exc}"}


def _verify_single(run):
    # Keep all physical/source/replay/parity checks from the immutable v1.
    result = original._verify_single(run)
    superseded = {key: result["checks"][key] for key in ("frozen_script_settings", "process_completed")}
    manifest = json.loads((run / "manifest.json").read_text())
    report = json.loads((run / "validation/report.json").read_text())
    with zipfile.ZipFile(run / "source.zip") as archive:
        overrides = json.loads(archive.read("configs/retry_unload_realign_v1.json"))
        resolved = asdict(RetrySettings(**overrides))
        result["checks"]["frozen_script_settings"] = resolved == report["controller_settings"]
        # The action replay in v1 already constructed RetrySettings(**overrides).
        # This new equality now checks that same resolved controller object.
        result["checks"]["resolved_settings_observation_contract"] = ActorRetryController.observation_order == [
            "fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "ft_force", "force_threshold"]
        if "continuation_registration" in manifest:
            raw = archive.read("configs/contact_impact_continuation_v2.json")
            import hashlib

            result["checks"]["archived_continuation_registration"] = (
                hashlib.sha256(raw).hexdigest() == manifest["continuation_registration_sha256"]
                and json.loads(raw) == manifest["continuation_registration"])
            result["checks"]["corrected_verifier_matches_launch_source"] = (
                manifest["source_hashes"]["scripts/review_contact_impact_v2.py"] == sha256(Path(__file__)))
    saved_path = run / "verification.json"
    saved_verification = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    known_failure = known_settings_verifier_failure(manifest, report, saved_verification)
    if manifest.get("status") == "verification_failed":
        result["checks"]["known_prior_cpu_only_failure"] = known_failure
        if known_failure:
            result["checks"]["process_completed"] = True
    result.update(
        status="verified" if all(result["checks"].values()) else "check_failed",
        failed_checks=[name for name, passed in result["checks"].items() if not passed],
        verifier_version="contact_impact_review_v2",
        verifier_source_sha256=sha256(Path(__file__)),
        original_verifier_source_sha256=sha256(ROOT / "scripts/review_contact_impact_v1.py"),
        superseded_verifier_checks=superseded,
        correction={
            "reason": "V1 compared the five sparse JSON overrides directly with the worker's 21 resolved RetrySettings fields. V2 compares asdict(RetrySettings(**archived_overrides)); original scripted-action replay already used these resolved defaults and remains unchanged.",
            "sparse_override_count": len(overrides), "resolved_field_count": len(resolved),
            "resolved_settings": resolved, "prior_cpu_failure_recognized": known_failure,
            "prior_verification_sha256": sha256(saved_path) if known_failure else None,
            "prior_manifest_sha256": sha256(run / "manifest.json") if known_failure else None,
            "simulator_rerun": False,
            "scope": "Original failed verification/run status stay immutable. This correction does not excuse a process failure, timeout, resource stop, additional failed check or altered physical evidence.",
        },
    )
    return result


def main():
    # Reuse immutable metrics, all-case denominators, thresholds and CLI parsing.
    # Only substitute the versioned CPU verifier; no original file is modified.
    original.verify_single = verify_single
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())

