"""CPU-only float32-aware correction of two contact precision checks.

No task, force threshold, native abort, controller, loss or evidence changes.
Historical workers and v1/v2 failed verdicts remain immutable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import sha256  # noqa: E402
from scripts import review_contact_impact_v1 as original  # noqa: E402
from scripts import review_contact_impact_v2 as inherited  # noqa: E402

PRECISION_CHECKS = {"native_impulse_to_force", "contact_evaluator_input"}


def gamma_float32(operations):
    """Standard round-to-nearest relative-error accumulation, u=2**-24."""
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    if type(operations) is not int or operations < 1 or operations * unit_roundoff >= 1:
        raise ValueError("Use a positive finite operation count")
    return operations * unit_roundoff / (1 - operations * unit_roundoff)


def rounding_comparison(left, right, active, *, absolute_floor, operations):
    """Use a local magnitude bound, retaining tight active checks near20N.

    The conversion budget allows four roundings across reciprocal/division and
    scalar representation paths. The norm budget allows six elementary
    operations for each three-component norm (three squares,two adds,square
    root), hence twelve for the difference of two evaluations. Positive sums
    do not amplify cancellation. The pinned matrix has one contributing body.
    Existing absolute floors cover zero/subnormal-scale quantities.
    """
    if left.shape != right.shape or active.dtype != torch.bool or active.shape != left.shape[:2]:
        raise ValueError("Paired tensors require a matching active [time,environment] mask")
    difference = (left.double() - right.double()).abs()
    scale = torch.maximum(left.double().abs(), right.double().abs())
    tolerance = torch.maximum(
        torch.full_like(scale, absolute_floor), gamma_float32(operations) * scale)
    mask = active.reshape(active.shape + (1,) * (left.ndim - 2)).expand_as(left)
    scalar_scale = scale.float()
    ulp = (torch.nextafter(scalar_scale, torch.full_like(scalar_scale, float("inf"))) - scalar_scale).double()
    ulp_distance = torch.where(ulp > 0, difference / ulp, torch.zeros_like(difference))
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    low_active = mask & (scale <= 20)
    checks = {
        "scaled_roundoff_bound": finite and bool((difference <= tolerance).all()),
        "active_at_most20n_original_absolute_bound": finite and bool((difference[low_active] < absolute_floor).all()),
    }
    summary = {
        "absolute_floor_n": absolute_floor, "rounding_operations": operations,
        "unit_roundoff": torch.finfo(torch.float32).eps / 2,
        "relative_bound_gamma_n": gamma_float32(operations),
    }
    for name, region in (("all", torch.ones_like(mask)), ("active", mask), ("absorbing", ~mask)):
        if not bool(region.any()):
            summary[name] = {"samples": 0}
            continue
        worst = torch.unravel_index(difference.masked_fill(~region, -1).flatten().argmax(), difference.shape)
        summary[name] = {
            "samples": int(region.sum()), "max_absolute_error_n": float(difference[region].max()),
            "max_float32_ulp_error": float(ulp_distance[region].max()),
            "entries_exceeding_old_absolute_bound": int((difference[region] >= absolute_floor).sum()),
            "max_local_magnitude_n": float(scale[region].max()),
            "worst_index": [int(value) for value in worst],
            "left_at_worst_n": float(left[worst]), "right_at_worst_n": float(right[worst]),
            "ulp_at_worst_n": float(ulp[worst]), "bound_at_worst_n": float(tolerance[worst]),
        }
    return checks, summary


def contact_precision_checks(report, data):
    """Precision checks with exact active contact-predicate parity."""
    native = data["native"]
    force, impulse = native["contact_force"], native["contact_impulse"]
    if force.dtype != torch.float32 or impulse.dtype != torch.float32 or force.shape[2] != 1 or force.shape[-1] != 3:
        raise ValueError("Precision model requires recorded float32 3D contact vectors with one contributing body")
    dt = report["criteria"]["physics_dt"]
    terminal_steps = torch.tensor([round(job["elapsed_s"] / dt) for job in report["jobs"]])
    active = torch.arange(1, force.shape[0] + 1)[:, None] <= terminal_steps[None, :]
    conversion_checks, conversion = rounding_comparison(
        force, impulse / dt, active, absolute_floor=3e-5, operations=4)
    computed = force.norm(dim=-1).sum(dim=2)
    recorded = data["physics"][..., [11, 9, 10]]
    norm_checks, norm = rounding_comparison(
        computed, recorded, active, absolute_floor=1e-5, operations=12)
    predicates, margins = {}, {}
    for threshold in (0.01, 0.1, 5.0):
        key = str(threshold)
        predicates[key] = bool(((computed.double() > threshold)[active] == (recorded > threshold)[active]).all())
        margins[key] = {
            "computed_minimum_active_distance_n": float((computed.double()[active] - threshold).abs().min()),
            "recorded_minimum_active_distance_n": float((recorded[active] - threshold).abs().min()),
            "active_predicate_differences": int(((computed.double() > threshold)[active] != (recorded > threshold)[active]).sum()),
        }
    checks = {
        "native_impulse_to_force": conversion_checks["scaled_roundoff_bound"],
        "contact_evaluator_input": norm_checks["scaled_roundoff_bound"],
        "conversion_active_20n_original_absolute_precision": conversion_checks["active_at_most20n_original_absolute_bound"],
        "contact_active_20n_original_absolute_precision": norm_checks["active_at_most20n_original_absolute_bound"],
        "recorded_contact_values_are_float32_exact": torch.equal(recorded, recorded.float().double()),
        "active_contact_threshold_predicates_unchanged": all(predicates.values()),
    }
    return checks, {
        "force_conversion": conversion, "contact_norm": norm,
        "active_physics_environment_samples": int(active.sum()),
        "absorbing_physics_environment_samples": int((~active).sum()),
        "contact_threshold_checks": predicates, "contact_threshold_margins": margins,
        "rule": "abs(a-b)<=max(original_absolute_floor,gamma_n*max(abs(a),abs(b))); original absolute bounds additionally retained for active local magnitudes<=20N.",
        "interpretation": "Measured high finger-contact loads remain physical evidence; this corrects only arithmetic agreement checks. Native raw wrist20N aborts and CPU replay are unchanged.",
    }


def known_precision_verifier_failure(manifest, report, saved):
    """Recognize only the preserved successful960Hz run's two CPU failures."""
    saved_checks, manifest_checks = saved.get("checks", {}), manifest.get("checks", {})
    return (
        manifest.get("status") == "verification_failed"
        and report.get("status") == "completed"
        and manifest.get("returncode") == 0 and manifest.get("timed_out") is False
        and not manifest.get("resource_guard_stop")
        and manifest.get("refinement") == report.get("refinement") == 8
        and report.get("cost", {}).get("charged_reference_transitions") == 13674.5
        and saved.get("status") == "check_failed"
        and saved.get("verifier_version") == "contact_impact_review_v2"
        and set(saved.get("failed_checks", [])) == PRECISION_CHECKS
        and {key for key, passed in saved_checks.items() if not passed} == PRECISION_CHECKS
        and {key for key, passed in manifest_checks.items() if not passed} == PRECISION_CHECKS
        and all(saved_checks.get(key) is True and manifest_checks.get(key) is True for key in (
            "process_completed", "independent_cpu_job_and_witness_replay", "source_archive_hash",
            "all_archived_source_hashes", "first_raw_20n_crossing_ends_active_job", "full_native_cost"))
    )


def verify_single(run: Path):
    try:
        return _verify_single(Path(run))
    except Exception as exc:
        return {
            "status": "verification_error", "checks": {"precision_verifier_completed": False},
            "failed_checks": ["precision_verifier_completed"], "error": f"{type(exc).__name__}: {exc}",
            "verifier_version": "contact_impact_review_v3",
        }


def _verify_single(run):
    result = inherited._verify_single(run)
    manifest = json.loads((run / "manifest.json").read_text())
    report = json.loads((run / "validation/report.json").read_text())
    data = torch.load(run / "validation/trajectory.pt", map_location="cpu", weights_only=True)
    superseded = {key: result["checks"].get(key) for key in (
        "native_impulse_to_force", "contact_evaluator_input", "process_completed", "known_prior_cpu_only_failure")}
    checks, precision = contact_precision_checks(report, data)
    result["checks"].update(checks)
    saved_path = run / "verification.json"
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    known_failure = known_precision_verifier_failure(manifest, report, saved)
    if known_failure:
        # Recognition never hides another current physical/source/replay failure:
        # all other inherited booleans still participate in the final verdict.
        result["checks"]["process_completed"] = True
        result["checks"]["known_prior_cpu_only_failure"] = True
        result["checks"]["known_prior_precision_failure_artifact_hash"] = (
            saved["trajectory_sha256"] == sha256(run / "validation/trajectory.pt")
            and saved["verifier_source_sha256"] == sha256(ROOT / "scripts/review_contact_impact_v2.py")
        )
    source_name = "scripts/review_contact_impact_v3.py"
    if source_name in manifest["source_hashes"]:
        result["checks"]["precision_verifier_matches_launch_source"] = manifest["source_hashes"][source_name] == sha256(Path(__file__))
    result.update(
        status="verified" if all(result["checks"].values()) else "check_failed",
        failed_checks=[key for key, passed in result["checks"].items() if not passed],
        verifier_version="contact_impact_review_v3", verifier_source_sha256=sha256(Path(__file__)),
        inherited_verifier_source_sha256=sha256(ROOT / "scripts/review_contact_impact_v2.py"),
        superseded_precision_checks=superseded, precision_diagnostics=precision,
        precision_correction={
            "known_prior_precision_failure_recognized": known_failure,
            "prior_verification_sha256": sha256(saved_path) if known_failure else None,
            "prior_manifest_sha256": sha256(run / "manifest.json") if known_failure else None,
            "simulator_rerun": False, "additional_simulator_transitions": 0,
            "scope": "Only float32 arithmetic agreement and the exact known CPU-only process verdict are corrected. Original data, workers, failed verification, physical20N force rule and all native replay checks remain unchanged.",
        },
    )
    return result


def main():
    original.verify_single = verify_single
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())
