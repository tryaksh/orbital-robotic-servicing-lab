"""Preserve the rejected V5 seed-4070 screen without altering either cohort."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path("D:/portfolio/.lab-work/orbital-improvement")
RELEASE = ROOT / "release"
INPUT = ROOT / "paired-24-v1"
COMMIT = "88235d8c27314b6092b0aac38506af9526ceea40"
CORRECTION = "4a433e048b954e88360242fd78e5bb8e5399a988"
ARCHIVE = RELEASE / "evidence/stability_camera_cadence_v5_20260914"
OUTPUT = RELEASE / "evidence/workflow_stability_camera_cadence_regression_v5_seed4070.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


sys.path.insert(0, str(RELEASE / "src"))
auditor_path = RELEASE / "scripts/audit_workflow_stability.py"
spec = importlib.util.spec_from_file_location("camera_regression_auditor", auditor_path)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
from zero_g_blade_swap.evaluation import wilson_interval  # noqa: E402
from zero_g_blade_swap.service import verification  # noqa: E402

manifest = json.loads((INPUT / "comparison.json").read_text(encoding="utf-8"))
if manifest["source_commit"] != COMMIT or [(r["seed"], r["arm"]) for r in manifest["runs"]] != [
    (4070, "legacy_rail"), (4070, "refined_fixed_base"),
]:
    raise ValueError("Require precisely the completed first pair from the rejected V5 source")
if any(record.get("exit_code") != 0 or "cohort" not in record for record in manifest["runs"]):
    raise ValueError("The first pair is incomplete")

by_arm, bindings, policy_sets, originals, metrics = {}, [], set(), [], {}
for arm in auditor.ARMS:
    rows, inputs, policy_set = auditor.load_run(INPUT, arm, 4070, COMMIT)
    by_arm[arm] = rows
    bindings.extend(inputs)
    policy_sets.add(policy_set)
    folder = INPUT / arm / "seed4070"
    originals.extend((folder / name, Path("paired") / arm / "seed4070" / name)
                     for name in ("workflow_report.json", "episodes.npz", "execution.log"))
    with np.load(folder / "episodes.npz", allow_pickle=False) as npz:
        fields = list(npz["fields"])
        values = np.asarray(npz["rows"])
    metrics[arm] = [{str(name): (float(value) if math.isfinite(float(value)) else None)
                     for name, value in zip(fields, row, strict=True)} for row in values]
    record = next(record for record in manifest["runs"] if record["arm"] == arm)
    if record["cohort"]["sha256"] != sha(folder / "episodes.npz"):
        raise ValueError("Original manifest does not match its NPZ bytes")
    if record["cohort"]["successes"] != sum(row["npz_success"] for row in rows):
        raise ValueError("Original manifest does not match its recorded outcome count")
if len(policy_sets) != 1:
    raise ValueError("The first pair has mismatched checkpoints")

arms = {arm: auditor.summarize(rows) for arm, rows in by_arm.items()}
paired = auditor.paired_changes(*(by_arm[arm] for arm in auditor.ARMS))
for values in arms.values():
    low, high = wilson_interval(values["npz_successes"], values["episodes"])
    values["npz_wilson_95"] = {"low": low, "high": high}
    low, high = wilson_interval(values["strict_physical_successes"], values["episodes"])
    values["strict_physical_wilson_95"] = {"low": low, "high": high}
if [arms[arm]["npz_successes"] for arm in auditor.ARMS] != [5, 3]:
    raise ValueError("Source screen differs from the rejected 5/8 versus 3/8 record")
if [row["env"] for row in paired["strict_physical_pass_to_fail"]] != [2, 6]:
    raise ValueError("Unexpected matched regression identities")

refined = json.loads((INPUT / "refined_fixed_base/seed4070/workflow_report.json").read_text(encoding="utf-8"))
motion = refined["motion_refinement"]
regressions = []
for env in (2, 6):
    row = metrics["refined_fixed_base"][env]
    regressions.append({
        "seed": 4070, "env": env, "baseline_npz": metrics["legacy_rail"][env], "refined_npz": row,
        "terminal_phase": "extract" if row["reached_phase"] == 2.0 else "other",
        "guard_started_at_driver_step": motion["extraction_finish_started_at_step"][env],
        "guard_motion_steps": motion["extraction_finish_steps"][env],
        "reported_guard_hold_steps": motion["extraction_finish_sensor_holds"][env],
        "max_joint_trim_bias_rad": motion["joint_trim_max_abs_bias_rad"][env],
    })
demo_path = ROOT / "smooth-fixed-v5/workflow_report.json"
demo = json.loads(demo_path.read_text(encoding="utf-8"))
if demo["motion_refinement"]["extraction_finish_sensor_holds"] != [0]:
    raise ValueError("Selected V5 demonstration did execute the corrected blocked branch")

originals.extend([
    (INPUT / "comparison.json", Path("paired/comparison.json")),
    (ROOT / "paired-24-v1.log", Path("original_supervisor.log")),
    (ROOT / "paired-continuation-plan.json", Path("scheduling_plan_before_pair_result.json")),
    (ROOT / "acceptance-plan.json", Path("original_acceptance_plan.json")),
    (demo_path, Path("selected_demo_v5_workflow_report.json")),
    (auditor_path, Path("derivation_tools/audit_workflow_stability.py")),
    (Path(verification.__file__), Path("derivation_tools/mission_verification.py")),
    (RELEASE / "tests/test_extraction_sensor_hold.py", Path("derivation_tools/test_extraction_sensor_hold.py")),
    (Path(__file__), Path("derivation_tools/preserve_camera_regression.py")),
])

reserved = INPUT / "legacy_rail/seed5070"
if not reserved.is_dir() or any(reserved.iterdir()):
    raise ValueError("The reserved 5070 directory is no longer empty")
if any((INPUT / arm / f"seed{seed}").exists()
       for arm, seed in (("refined_fixed_base", 5070), ("legacy_rail", 6070), ("refined_fixed_base", 6070))):
    raise ValueError("Unexpected later V5 cohort directory; audit its provenance before packaging")
if ARCHIVE.exists() or OUTPUT.exists():
    raise FileExistsError("Preservation refuses to overwrite an archive or derived report")
ARCHIVE.mkdir(parents=True)
copied = []
for source, relative in originals:
    destination = ARCHIVE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(source) != sha(destination):
        raise ValueError(f"Archive copy changed bytes: {source}")
    copied.append({"path": destination.relative_to(RELEASE).as_posix(), "source_path": str(source),
                   "sha256": sha(destination), "size_bytes": destination.stat().st_size})

scheduling = {
    "original_scheduling_plan_utc": "2026-09-14T21:01:00Z",
    "reserved_directory": "paired-24-v1/legacy_rail/seed5070", "reserved_directory_was_empty": True,
    "original_supervisor_stop": "FileExistsError at the next seed5070 mkdir, after both seed4070 cohorts completed normally.",
    "planned_reason": "Scheduling handoff to two independent cohorts; this plan preceded the refined4070 result.",
    "subsequent_decision": "The completed first pair regressed. V5 was rejected before any remaining cohort launched; the scheduling continuation was not executed.",
    "completed_seeds": [4070], "not_run_seeds": [5070, 6070],
    "partial_comparison": True, "episodes_per_arm_observed": 8, "episodes_per_arm_originally_planned": 24,
    "no_active_cohort_interrupted": True, "no_result_rescoring": True,
    "new_comparison": "All six seed/arm cohorts rerun from the separate corrected clean revision; no V5 first-pair reuse.",
}
write(ARCHIVE / "scheduling_note.json", scheduling)
copied.append({"path": (ARCHIVE / "scheduling_note.json").relative_to(RELEASE).as_posix(),
               "sha256": sha(ARCHIVE / "scheduling_note.json"), "size_bytes": (ARCHIVE / "scheduling_note.json").stat().st_size})
zip_path = ARCHIVE / "originals.zip"
with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
    for record in copied:
        source = RELEASE / record["path"]
        bundle.write(source, source.relative_to(ARCHIVE).as_posix())
with zipfile.ZipFile(zip_path) as bundle:
    for record in copied:
        relative = (RELEASE / record["path"]).relative_to(ARCHIVE).as_posix()
        if hashlib.sha256(bundle.read(relative)).hexdigest() != record["sha256"]:
            raise ValueError("ZIP did not preserve a recorded input's exact bytes")

result = {
    "schema": "workflow_stability_rejected_seed_screen_v1",
    "title": "V5 seed-4070 camera-cadence regression retained before corrective hold",
    "evidence_type": "partial_paired_simulation_regression_with_supplemental_physical_audit",
    "created_utc": datetime.now(UTC).isoformat(), "source_commit": COMMIT,
    "runtime_source_bindings": refined["runtime_source_bindings"],
    "status": "rejected_for_promotion", "full_three_seed_comparison_complete": False,
    "source_repository": "https://github.com/tryaksh/orbital-robotic-servicing-lab",
    "source_snapshot_registry": "evidence/source_snapshots.json",
    "protocol": {"seed": 4070, "environments_per_arm": 8, "steps": 1900,
                 "task": auditor.TASK, "lighting": "randomized", "video": False,
                 "other_seeds_not_run": [5070, 6070]},
    "policy_set_sha256": next(iter(policy_sets)), "arms": arms, "paired_changes": paired,
    "episode_results": by_arm, "regressed_environments": regressions,
    "diagnosis": {
        "mechanism": "During guarded terminal extraction, a blocked observation/grip gate set solved_joint_hold=False while Cartesian arm actions were zero. The ordinary relative-IK path therefore replaced the accepted biased joint target on blocked ticks; fresh observations restored the absolute target.",
        "observed_support": "The two newly failing environments remained in extraction, with nearly as many reported guard holds as commanded guard steps. Production-method tests reproduce the actuator-ownership discontinuity under alternating fresh/missing camera flags.",
        "corrective_source_commit": CORRECTION,
        "corrective_change": "Retain solved_joint_hold=True on blocked ticks; retain the accepted joint target and freeze profile time/trim until observations permit progress.",
        "criterion_changes": False, "camera_frequency_hz_claimed": None,
        "limits": "These report counters combine missing-camera and lost-grip guard pauses. The reports do not contain a per-control-step detection/ownership trace, so counts alone do not establish an exact alternating sequence or a camera frequency. Physical performance of the corrected revision belongs to its separate fresh comparison.",
    },
    "selected_video": {
        "source_commit": COMMIT, "seed": 6070,
        "reported_guard_hold_steps": demo["motion_refinement"]["extraction_finish_sensor_holds"],
        "scope": "The selected V5 single-environment video has zero guarded-extraction blocked steps, so the later blocked-branch change was not exercised by this recorded episode. The video remains attributed to V5, not relabelled as a new-source run.",
    },
    "audit": {
        "helpers": ["load_run", "summarize", "paired_changes"],
        "auditor_sha256": sha(auditor_path), "mission_verifier_sha256": sha(Path(verification.__file__)),
        "thresholds": {"position_limit_m": auditor.POSITION_LIMIT_M,
                       "orientation_limit_rad": auditor.ORIENTATION_LIMIT_RAD, "rack_hold_s": auditor.RACK_HOLD_S},
        "original_input_bindings": bindings,
    },
    "archive": {"directory": ARCHIVE.relative_to(RELEASE).as_posix(), "files": copied,
                "raw_byte_recovery_zip": {"path": zip_path.relative_to(RELEASE).as_posix(),
                                          "sha256": sha(zip_path), "size_bytes": zip_path.stat().st_size}},
    "scheduling": scheduling,
    "scope_and_limitations": [
        "Only the first seed pair completed: eight initial conditions per arm at seed 4070, not 24 per arm. Seeds 5070 and 6070 were not run for V5.",
        "Both original NPZ counts and existing physical retention/release checks give 5/8 legacy and 3/8 V5; matched environments 2 and 6 change pass to fail, with no gained successes.",
        "Original reports, NPZs, logs, comparison manifest, pre-result scheduling plan and supervisor termination log are preserved unchanged. No outcome is rescored under a new criterion.",
        "The supplemental audit uses the existing mission position/orientation/hold limits; it does not independently certify video, camera quality, or all seven seating conditions beyond the original NPZ outcome.",
        "Controllers and transport mechanisms differ together. This partial development screen is not a new independent reliability estimate or a completed 95% gate evaluation.",
        "Seeds 6070 and 4070 both informed subsequent development. The corrected controller requires a separate all-six-run comparison on its own clean revision.",
        "Simulation only: fixed-base and idealized lock abstractions remain; no hardware or flight dynamics qualification.",
    ],
}
write(OUTPUT, result)
for record in copied:
    if sha(RELEASE / record["path"]) != record["sha256"]:
        raise ValueError("Archive integrity changed before completion")
print(json.dumps({"report": str(OUTPUT), "archive": str(ARCHIVE), "files_preserved": len(copied),
                  "arms": arms, "regressed_environments": [r["env"] for r in regressions]}, indent=2))
