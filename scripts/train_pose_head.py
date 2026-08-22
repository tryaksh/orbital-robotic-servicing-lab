"""Train the module-pose head that replaces the simulator's answer.

Supervised regression from an RGB frame to the module's pose, on frames
recorded by ``scripts/collect_grapple_vision.py`` under randomized orbital
lighting, rack albedo, camera noise, and an unknown per-episode module
displacement.

Two choices here are about honesty rather than accuracy.

**The validation split is the tail of the collection run, not a random subset.**
Frames from one round share nothing but the reset that produced them, so a
random split would still be defensible — but a contiguous tail additionally
catches anything that drifts during collection, and it costs nothing.

**The reported number is millimetres, not loss.** A normalised MSE can look
excellent while the estimator is useless at the scale that matters, and the
scale that matters here is written into the interface specification: the capture
tolerance is 20 mm and the insertion needs 4 mm laterally. Those are the numbers
the head is judged against, so those are the numbers printed.

CPU-trainable in principle; a few minutes on the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Imported without Isaac Lab: the head is a plain ``nn.Module`` and this script
#: must run on a machine that has no simulator.
from zero_g_blade_swap.pose_head import (  # noqa: E402
    MODULE_POSE_DIM,
    POSE_HEAD_ARCHITECTURE_V1,
    POSE_HEAD_ARCHITECTURE_V2,
    POSE_HEAD_LEGACY_GRID_SIZE,
    POSE_HEAD_OVERVIEW_GRID_SIZE,
    ModulePoseHead,
)

#: What the interface specification requires, for context in the report.
CAPTURE_TOLERANCE_MM = 20.0
INSERTION_LATERAL_TOLERANCE_MM = 4.0

#: How hard the occupancy branch is charged against the pose regression.
#:
#: One, and deliberately not tuned. The pose is the quantity the policies
#: consume and the occupancy is a read-out; if the two ever traded against each
#: other the honest response is to report the trade, not to weight it away until
#: the headline number recovers. The report prints both, so a trade is visible.
OCCUPANCY_LOSS_WEIGHT = 1.0

# Promotion is a two-capability claim on the two-bay head: locate the module and
# read the rack state. This is an explicit product gate, not a confidence score.
OCCUPANCY_EXACT_MATCH_THRESHOLD = 0.95

# Measured valid 256px collections have minimum per-frame std >52 and maximum
# 255. The rejected all-zero render, after radiation noise, had std 4.31 and
# maximum 60. These conservative floors reject absent scene signal without
# tuning the gate to model results.
MIN_FRAME_SIGNAL_STD = 15.0
MIN_FRAME_SIGNAL_MAX = 128.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: np.ndarray):
    scalar = value.item()
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def _dataset_signal_statistics(images: np.ndarray) -> dict[str, float | int | bool]:
    sample_count = min(len(images), 256)
    indices = np.linspace(0, len(images) - 1, sample_count, dtype=np.int64)
    sample = images[indices].astype(np.float32, copy=False).reshape(sample_count, -1)
    per_frame_std = sample.std(axis=1)
    per_frame_max = sample.max(axis=1)
    return {
        "sampled_frames": sample_count,
        "per_frame_std_min": float(per_frame_std.min()),
        "per_frame_std_median": float(np.median(per_frame_std)),
        "per_frame_max_min": float(per_frame_max.min()),
        "global_mean": float(sample.mean()),
        "global_std": float(sample.std()),
        "global_max": int(per_frame_max.max()),
        "passed": bool(
            np.all(per_frame_std >= MIN_FRAME_SIGNAL_STD)
            and np.all(per_frame_max >= MIN_FRAME_SIGNAL_MAX)
        ),
    }


def _checkpoint_selection_key(
    entry: dict[str, float | int], occupancy_present: bool
) -> tuple[float, ...]:
    """Rank one epoch using the same capabilities the promotion gate names."""

    pose = (
        float(entry["position_error_mm_p95"]),
        float(entry["position_error_mm_mean"]),
    )
    if not occupancy_present:
        return pose
    exact = float(entry["occupancy_exact_match"])
    if exact >= OCCUPANCY_EXACT_MATCH_THRESHOLD:
        return (0.0, *pose, -exact)
    # If no epoch reaches the occupancy gate, preserve the least-bad rack-state
    # reader for diagnosis rather than silently selecting a pose-only prior.
    return (1.0, -exact, *pose)


def _occupancy_pattern_name(pattern: tuple[bool, bool]) -> str:
    return {
        (False, False): "neither_bay",
        (True, False): "bay_0",
        (False, True): "bay_1",
        (True, True): "both_bays",
    }[pattern]


def _occupancy_diagnostics(
    position_mm: torch.Tensor,
    lateral_mm: torch.Tensor,
    predicted: torch.Tensor,
    truth: torch.Tensor,
) -> dict[str, object]:
    """Report discrete rack-state performance without calling logits confidence."""

    position_mm = position_mm.cpu()
    lateral_mm = lateral_mm.cpu()
    predicted = predicted.to(torch.bool).cpu()
    truth = truth.to(torch.bool).cpu()
    patterns = ((False, False), (True, False), (False, True), (True, True))

    truth_names = [_occupancy_pattern_name(tuple(bool(value) for value in row.tolist())) for row in truth]
    predicted_names = [
        _occupancy_pattern_name(tuple(bool(value) for value in row.tolist())) for row in predicted
    ]
    pattern_counts = {name: truth_names.count(name) for name in map(_occupancy_pattern_name, patterns)}
    predicted_pattern_counts = {
        name: predicted_names.count(name) for name in map(_occupancy_pattern_name, patterns)
    }

    category_metrics: dict[str, dict[str, float | int]] = {}
    for pattern in patterns:
        name = _occupancy_pattern_name(pattern)
        mask = torch.tensor([value == name for value in truth_names], dtype=torch.bool)
        count = int(mask.sum())
        metrics: dict[str, float | int] = {"frames": count}
        if count:
            category_correct = (predicted[mask] == truth[mask]).all(dim=1).to(torch.float32)
            metrics.update(
                {
                    "position_error_mm_mean": float(position_mm[mask].mean()),
                    "position_error_mm_p95": float(position_mm[mask].quantile(0.95)),
                    "lateral_error_mm_mean": float(lateral_mm[mask].mean()),
                    "lateral_error_mm_p95": float(lateral_mm[mask].quantile(0.95)),
                    "occupancy_exact_match": float(category_correct.mean()),
                }
            )
        category_metrics[name] = metrics

    confusion = {
        truth_name: {
            predicted_name: sum(
                actual == truth_name and estimate == predicted_name
                for actual, estimate in zip(truth_names, predicted_names, strict=True)
            )
            for predicted_name in predicted_pattern_counts
        }
        for truth_name in pattern_counts
    }
    correct = predicted == truth
    majority_pattern_rate = max(pattern_counts.values()) / len(truth_names)
    return {
        "per_bay_accuracy": [float(value) for value in correct.to(torch.float32).mean(dim=0)],
        "per_bay_positive_fraction": [float(value) for value in truth.to(torch.float32).mean(dim=0)],
        "exact_match": float(correct.all(dim=1).to(torch.float32).mean()),
        "majority_pattern_rate": majority_pattern_rate,
        "pattern_counts": pattern_counts,
        "predicted_pattern_counts": predicted_pattern_counts,
        "category_metrics": category_metrics,
        "confusion_counts": confusion,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/grapple_vision.npz"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/module_pose_head.pth"))
    parser.add_argument("--report", type=Path, default=Path("evidence/module_pose_head.json"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--feature_grid_size",
        type=int,
        choices=(POSE_HEAD_LEGACY_GRID_SIZE, POSE_HEAD_OVERVIEW_GRID_SIZE),
        default=POSE_HEAD_LEGACY_GRID_SIZE,
        help="4 selects legacy architecture v1; 8 selects overview architecture v2.",
    )
    parser.add_argument(
        "--deployment_task",
        default=None,
        help="Exact runtime task ID this checkpoint is intended for; recorded in the deployment contract.",
    )
    parser.add_argument(
        "--camera_scale_evidence",
        type=Path,
        default=None,
        help="Optional rendered camera-scale evidence file to bind by SHA256.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    torch.manual_seed(args.seed)
    data = np.load(args.dataset)
    image_array = data["images"]
    dataset_signal = _dataset_signal_statistics(image_array)
    dataset_metadata = {
        key: _json_scalar(data[key])
        for key in data.files
        if key not in {"images", "labels", "occupancy"} and data[key].ndim == 0
    }
    dataset_provenance = {
        "path": str(args.dataset),
        "sha256": _sha256(args.dataset),
        "size_bytes": args.dataset.stat().st_size,
        "image_shape": [int(value) for value in image_array.shape],
        "image_dtype": str(image_array.dtype),
        "metadata": dataset_metadata,
        "image_signal": dataset_signal,
    }
    project_root = Path(__file__).resolve().parents[1]
    camera_config_source = project_root / "src/zero_g_blade_swap/tasks/blade_swap/scene_cfg.py"
    architecture_version = (
        POSE_HEAD_ARCHITECTURE_V1
        if args.feature_grid_size == POSE_HEAD_LEGACY_GRID_SIZE
        else POSE_HEAD_ARCHITECTURE_V2
    )
    architecture = {
        "name": "ModulePoseHead",
        "version": architecture_version,
        "feature_grid_size": args.feature_grid_size,
    }
    deployment_contract = {
        "deployment_task_id": args.deployment_task,
        "collection_task_id": dataset_metadata.get("collection_task"),
        "image_shape_hwc": [int(value) for value in image_array.shape[1:]],
        "pose_distribution": dataset_metadata.get("pose_distribution"),
        "pose_head_architecture": architecture,
        "camera_config_source": {
            "path": str(camera_config_source.relative_to(project_root)),
            "sha256": _sha256(camera_config_source),
        },
        "camera_scale_evidence": (
            None
            if args.camera_scale_evidence is None
            else {
                "path": str(args.camera_scale_evidence),
                "sha256": _sha256(args.camera_scale_evidence),
            }
        ),
    }
    print(f"[INFO] dataset provenance: sha256={dataset_provenance['sha256']}")
    print(f"[INFO] dataset image-signal gate: {dataset_signal}")
    if not dataset_signal["passed"]:
        failure_report = {
            "status": "failed",
            "failure_stage": "dataset_image_signal_gate",
            "title": "Module pose training rejected a noise-only camera corpus",
            "evidence_type": "simulation_perception_regression",
            "dataset_provenance": dataset_provenance,
            "deployment_contract": deployment_contract,
            "architecture": architecture,
            "gate": {
                "minimum_per_frame_std": MIN_FRAME_SIGNAL_STD,
                "minimum_per_frame_max": MIN_FRAME_SIGNAL_MAX,
                "passed": False,
                "rationale": (
                    "Training cannot recover scene state from an all-zero render hidden by sensor noise. "
                    "The checkpoint path is left untouched when this gate fails."
                ),
            },
            "checkpoint": str(args.output),
            "checkpoint_written": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
        raise RuntimeError(
            "dataset failed the image-signal gate; report written and existing checkpoint preserved"
        )

    images = torch.from_numpy(image_array)
    labels = torch.from_numpy(data["labels"]).to(torch.float32)
    if labels.shape[1] != MODULE_POSE_DIM:
        raise ValueError(f"expected {MODULE_POSE_DIM} label channels, got {labels.shape[1]}")
    # Present only in a two-bay dataset. A single-bay run trains exactly the head
    # it trained before, so `evidence/module_pose_head.json` keeps describing a
    # comparable network.
    occupancy = None if "occupancy" not in data.files else torch.from_numpy(data["occupancy"]).to(torch.float32)
    occupancy_slots = 0 if occupancy is None else int(occupancy.shape[1])

    split = int(len(images) * (1.0 - args.validation_fraction))
    if split < 1 or split >= len(images):
        raise ValueError("validation_fraction leaves no data on one side of the split")
    print(f"[INFO] {split} training frames, {len(images) - split} held out")
    if occupancy is not None:
        print(
            f"[INFO] occupancy branch on, {occupancy_slots} bays, per-bay positive fraction "
            f"{[round(float(v), 4) for v in occupancy[:split].mean(dim=0)]}"
        )

    # Statistics from the training half only. Computing them over everything
    # would leak the validation set into the normalisation.
    mean = labels[:split].mean(dim=0)
    std = labels[:split].std(dim=0).clamp_min(1e-6)

    head = ModulePoseHead(
        output_dim=MODULE_POSE_DIM,
        occupancy_slots=occupancy_slots,
        feature_grid_size=args.feature_grid_size,
        architecture_version=architecture_version,
    ).to(args.device)
    head.label_mean.copy_(mean.to(args.device))
    head.label_std.copy_(std.to(args.device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()
    # Per-bay independent indicators, so binary cross-entropy per logit rather
    # than a softmax over bays: during a relocation the module is genuinely in
    # neither bay for the whole transit, and a softmax cannot say that.
    occupancy_loss_fn = nn.BCEWithLogitsLoss()

    # Carried as a third tensor so the single-bay path builds the same two-tensor
    # dataset it always did. A zero-width column would be simpler and would let a
    # bug produce an empty prediction silently.
    columns = [images[:split], labels[:split]]
    if occupancy is not None:
        columns.append(occupancy[:split])
    loader = DataLoader(
        TensorDataset(*columns),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_images = images[split:].to(args.device)
    validation_labels = labels[split:].to(args.device)
    validation_occupancy = None if occupancy is None else occupancy[split:].to(args.device)

    selected_key: tuple[float, ...] | None = None
    selected_entry: dict[str, float | int] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        for batch in loader:
            batch_images = batch[0].to(args.device, non_blocking=True).to(torch.float32).div_(255.0)
            batch_labels = batch[1].to(args.device, non_blocking=True)
            if occupancy is None:
                predicted = head(batch_images)
            else:
                predicted, logits = head.forward_with_occupancy(batch_images)
            # Loss in normalised space so the three position channels and the
            # three rotation channels contribute comparably.
            loss = loss_fn((predicted - head.label_mean) / head.label_std, (batch_labels - head.label_mean) / head.label_std)
            if occupancy is not None:
                loss = loss + OCCUPANCY_LOSS_WEIGHT * occupancy_loss_fn(
                    logits, batch[2].to(args.device, non_blocking=True)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch_images)
        schedule.step()

        head.eval()
        with torch.inference_mode():
            errors = []
            predicted_occupancy = []
            for start in range(0, len(validation_images), 1024):
                chunk = validation_images[start : start + 1024].to(torch.float32).div_(255.0)
                if occupancy is None:
                    errors.append(head(chunk) - validation_labels[start : start + 1024])
                else:
                    pose, logits = head.forward_with_occupancy(chunk)
                    errors.append(pose - validation_labels[start : start + 1024])
                    predicted_occupancy.append(logits > 0.0)
            error = torch.cat(errors, dim=0)
        position_mm = 1_000.0 * torch.linalg.vector_norm(error[:, :3], dim=-1)
        entry = {
            "epoch": epoch,
            "train_loss": total / max(1, split),
            "position_error_mm_mean": float(position_mm.mean()),
            "position_error_mm_p95": float(position_mm.quantile(0.95)),
            "orientation_error_rad_mean": float(torch.linalg.vector_norm(error[:, 3:], dim=-1).mean()),
        }
        if occupancy is not None:
            correct = torch.cat(predicted_occupancy, dim=0) == (validation_occupancy > 0.5)
            # Both numbers, because they answer different questions: per-bay
            # accuracy is what a per-logit loss optimises, and exact-match is
            # whether the head got the whole rack right on that frame -- which
            # is the claim "the camera reads the state of the rack" actually
            # makes.
            entry["occupancy_accuracy"] = float(correct.to(torch.float32).mean())
            entry["occupancy_exact_match"] = float(correct.all(dim=-1).to(torch.float32).mean())
        history.append(entry)
        selection_key = _checkpoint_selection_key(entry, occupancy is not None)
        if selected_key is None or selection_key < selected_key:
            selected_key = selection_key
            selected_entry = entry.copy()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), args.output)
        if epoch % 5 == 0 or epoch == 1:
            occupancy_note = (
                ""
                if occupancy is None
                else f"  occupancy {entry['occupancy_accuracy']:.4f} exact {entry['occupancy_exact_match']:.4f}"
            )
            print(
                f"[INFO] epoch {epoch:3d}  loss {entry['train_loss']:.5f}  "
                f"position {entry['position_error_mm_mean']:.2f} mm  p95 {entry['position_error_mm_p95']:.2f} mm"
                f"{occupancy_note}",
                flush=True,
            )

    final = history[-1]
    if selected_entry is None:
        raise RuntimeError("training produced no selectable checkpoint")
    best_entry = selected_entry

    # Re-load exactly what was written and derive the published metrics from
    # those weights. This prevents a report from combining one epoch's numbers
    # with another epoch's checkpoint.
    head.load_state_dict(torch.load(args.output, map_location=args.device, weights_only=True))
    head.eval()
    with torch.inference_mode():
        selected_errors = []
        selected_occupancy = []
        for start in range(0, len(validation_images), 1024):
            chunk = validation_images[start : start + 1024].to(torch.float32).div_(255.0)
            if occupancy is None:
                selected_errors.append(head(chunk) - validation_labels[start : start + 1024])
            else:
                pose, logits = head.forward_with_occupancy(chunk)
                selected_errors.append(pose - validation_labels[start : start + 1024])
                selected_occupancy.append(logits > 0.0)
    selected_error = torch.cat(selected_errors, dim=0)
    selected_position_mm = 1_000.0 * torch.linalg.vector_norm(selected_error[:, :3], dim=-1)
    selected_lateral_mm = 1_000.0 * torch.linalg.vector_norm(selected_error[:, 1:3], dim=-1)
    selected_occupancy_diagnostics = None
    if occupancy is not None:
        selected_occupancy_diagnostics = _occupancy_diagnostics(
            selected_position_mm,
            selected_lateral_mm,
            torch.cat(selected_occupancy, dim=0),
            validation_occupancy > 0.5,
        )

    pose_gate_passed = bool(float(selected_position_mm.quantile(0.95)) < CAPTURE_TOLERANCE_MM)
    occupancy_gate_passed = True
    if selected_occupancy_diagnostics is not None:
        exact_match = float(selected_occupancy_diagnostics["exact_match"])
        majority_pattern_rate = float(selected_occupancy_diagnostics["majority_pattern_rate"])
        occupancy_gate_passed = bool(
            exact_match >= OCCUPANCY_EXACT_MATCH_THRESHOLD and exact_match > majority_pattern_rate
        )
    promotion_passed = pose_gate_passed and occupancy_gate_passed
    checkpoint_sha256 = _sha256(args.output)

    report = {
        "status": "passed" if promotion_passed else "failed",
        "title": f"Module pose regressed from a {images.shape[1]}x{images.shape[2]} servicing camera",
        "evidence_type": "simulation_perception_regression",
        "selected_epoch": int(best_entry["epoch"]),
        "dataset_provenance": dataset_provenance,
        "deployment_contract": deployment_contract,
        "architecture": architecture,
        "protocol": {
            "dataset": str(args.dataset),
            "dataset_sha256": dataset_provenance["sha256"],
            "image_resolution_px": [int(images.shape[1]), int(images.shape[2])],
            "pose_distribution": dataset_metadata.get("pose_distribution"),
            "frames": int(len(images)),
            "training_frames": split,
            "held_out_frames": int(len(images) - split),
            "held_out_split": "contiguous tail of the collection run",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "randomization": [
                "orbital sun intensity, angle, pitch, yaw, colour temperature",
                "rack albedo: steel and gold, metallic and roughness",
                "camera radiation noise",
                "unknown per-episode module displacement",
            ],
        },
        "held_out": {
            "position_error_mm_mean": float(selected_position_mm.mean()),
            "position_error_mm_p95": float(selected_position_mm.quantile(0.95)),
            "lateral_error_mm_mean": float(selected_lateral_mm.mean()),
            "lateral_error_mm_p95": float(selected_lateral_mm.quantile(0.95)),
            "orientation_error_rad_mean": float(
                torch.linalg.vector_norm(selected_error[:, 3:], dim=-1).mean()
            ),
            "final_epoch_position_error_mm_mean": final["position_error_mm_mean"],
            **(
                {}
                if occupancy is None
                else {
                    "occupancy_bays": occupancy_slots,
                    "occupancy_accuracy": float(
                        np.mean(selected_occupancy_diagnostics["per_bay_accuracy"])
                    ),
                    "occupancy_exact_match": selected_occupancy_diagnostics["exact_match"],
                    "occupancy_per_bay_accuracy": selected_occupancy_diagnostics["per_bay_accuracy"],
                    "occupancy_per_bay_positive_fraction": selected_occupancy_diagnostics[
                        "per_bay_positive_fraction"
                    ],
                    "occupancy_majority_pattern_rate": selected_occupancy_diagnostics[
                        "majority_pattern_rate"
                    ],
                    "occupancy_pattern_counts": selected_occupancy_diagnostics["pattern_counts"],
                    "occupancy_predicted_pattern_counts": selected_occupancy_diagnostics[
                        "predicted_pattern_counts"
                    ],
                    "occupancy_confusion_counts": selected_occupancy_diagnostics["confusion_counts"],
                    "category_metrics": selected_occupancy_diagnostics["category_metrics"],
                }
            ),
        },
        "gates": {
            "dataset_image_signal": {
                "passed": dataset_signal["passed"],
                "minimum_per_frame_std": MIN_FRAME_SIGNAL_STD,
                "minimum_per_frame_max": MIN_FRAME_SIGNAL_MAX,
            },
            "pose_p95": {
                "value_mm": float(selected_position_mm.quantile(0.95)),
                "threshold_mm_exclusive": CAPTURE_TOLERANCE_MM,
                "passed": pose_gate_passed,
            },
            "occupancy_exact_match": {
                "applies": occupancy is not None,
                "value": (
                    None
                    if selected_occupancy_diagnostics is None
                    else selected_occupancy_diagnostics["exact_match"]
                ),
                "threshold_inclusive": OCCUPANCY_EXACT_MATCH_THRESHOLD,
                "majority_pattern_rate": (
                    None
                    if selected_occupancy_diagnostics is None
                    else selected_occupancy_diagnostics["majority_pattern_rate"]
                ),
                "must_strictly_beat_majority_pattern": True,
                "passed": occupancy_gate_passed,
            },
            "promotion_passed": promotion_passed,
        },
        "context": {
            "capture_tolerance_mm": CAPTURE_TOLERANCE_MM,
            "insertion_lateral_tolerance_mm": INSERTION_LATERAL_TOLERANCE_MM,
            "note": (
                "A pose error is only meaningful against the tolerance it has to fit inside. The capture "
                "predicate accepts 20 mm of grip error and the insertion needs 4 mm laterally, both from "
                "docs/service_interface_spec.md."
            ),
        },
        "history": history,
        "checkpoint": str(args.output),
        "checkpoint_sha256": checkpoint_sha256,
        "scope_and_limitations": [
            "Simulation only. This is a rendered camera, not a calibrated real one.",
            "Held-out frames come from the same collection run, so this measures generalisation across "
            "lighting, albedo, noise and module pose -- not across a different renderer or a real sensor.",
            "Occupancy logits are thresholded at zero and reported as discrete outcomes; no calibrated "
            "confidence claim is made.",
            "The number that matters is the workflow success rate with this head in the loop, which is "
            "certified separately.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[INFO] selected epoch {int(best_entry['epoch'])}: held-out position error "
        f"{float(selected_position_mm.mean()):.2f} mm mean, "
        f"{float(selected_position_mm.quantile(0.95)):.2f} mm p95"
    )
    if selected_occupancy_diagnostics is not None:
        print(
            f"[INFO] selected occupancy exact-match "
            f"{float(selected_occupancy_diagnostics['exact_match']):.4f}, majority-pattern baseline "
            f"{float(selected_occupancy_diagnostics['majority_pattern_rate']):.4f}"
        )
    print(f"[INFO] promotion gate {'passed' if promotion_passed else 'failed'}; checkpoint {checkpoint_sha256}")
    print(f"[INFO] wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
