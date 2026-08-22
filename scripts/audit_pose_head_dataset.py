"""Audit why a camera pose head generalizes or fails without retraining it.

This is diagnostic evidence, not another promotion gate. It compares train and
held-out errors per axis, pose and image-signal distributions, collection order,
post-physics drift outside the authored workflow envelope, and the label
separation between visually nearest examples across the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zero_g_blade_swap.grapple_geometry import TRANSIT_CLEAR_BLADE_CENTRE_X  # noqa: E402
from zero_g_blade_swap.pose_head import (  # noqa: E402
    checkpoint_matches_sha256,
    checkpoint_sha256,
    load_pose_head,
)

AXES = ("x", "y", "z")
BLADE_INSERTED_Z = 0.72
SECOND_SLOT_CENTER_Y = -0.22


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training_report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, (0.01, 0.05, 0.5, 0.95, 0.99))
    return dict(zip(("p01", "p05", "p50", "p95", "p99"), (float(value) for value in points), strict=True))


def _error_metrics(error: np.ndarray) -> dict[str, object]:
    position_mm = 1_000.0 * np.linalg.norm(error[:, :3], axis=1)
    lateral_mm = 1_000.0 * np.linalg.norm(error[:, 1:3], axis=1)
    per_axis = 1_000.0 * np.abs(error[:, :3])
    return {
        "frames": len(error),
        "position_error_mm_mean": float(position_mm.mean()),
        "position_error_mm_p95": float(np.quantile(position_mm, 0.95)),
        "lateral_error_mm_mean": float(lateral_mm.mean()),
        "lateral_error_mm_p95": float(np.quantile(lateral_mm, 0.95)),
        "per_axis_absolute_error_mm": {
            axis: {
                "mean": float(per_axis[:, index].mean()),
                "p95": float(np.quantile(per_axis[:, index], 0.95)),
            }
            for index, axis in enumerate(AXES)
        },
    }


def _label_distribution(labels: np.ndarray) -> dict[str, object]:
    return {
        "position": {
            axis: {
                "mean": float(labels[:, index].mean()),
                "std": float(labels[:, index].std()),
                **_quantiles(labels[:, index]),
            }
            for index, axis in enumerate(AXES)
        },
        "rotation_vector_norm_rad": _quantiles(np.linalg.norm(labels[:, 3:], axis=1)),
    }


def _image_signal(images: np.ndarray) -> dict[str, object]:
    count = min(len(images), 256)
    indices = np.linspace(0, len(images) - 1, count, dtype=np.int64)
    sample = images[indices].astype(np.float32).reshape(count, -1)
    frame_mean = sample.mean(axis=1)
    frame_std = sample.std(axis=1)
    return {
        "sampled_frames": count,
        "frame_mean": {"mean": float(frame_mean.mean()), **_quantiles(frame_mean)},
        "frame_std": {"mean": float(frame_std.mean()), **_quantiles(frame_std)},
        "global_max": int(sample.max()),
    }


def _envelope_drift(labels: np.ndarray, occupancy: np.ndarray) -> dict[str, object]:
    bounds = {
        "neither_bay": (
            np.array((TRANSIT_CLEAR_BLADE_CENTRE_X - 0.08, SECOND_SLOT_CENTER_Y - 0.025, BLADE_INSERTED_Z - 0.05)),
            np.array((TRANSIT_CLEAR_BLADE_CENTRE_X + 0.06, 0.025, BLADE_INSERTED_Z + 0.05)),
            np.array((0.0, 0.0)),
        ),
        "bay_0": (
            np.array((0.55, -0.012, BLADE_INSERTED_Z - 0.01)),
            np.array((0.73, 0.012, BLADE_INSERTED_Z + 0.01)),
            np.array((1.0, 0.0)),
        ),
        "bay_1": (
            np.array((0.55, SECOND_SLOT_CENTER_Y - 0.012, BLADE_INSERTED_Z - 0.01)),
            np.array((0.73, SECOND_SLOT_CENTER_Y + 0.012, BLADE_INSERTED_Z + 0.01)),
            np.array((0.0, 1.0)),
        ),
    }
    result = {}
    for name, (lower, upper, pattern) in bounds.items():
        mask = np.all(occupancy == pattern, axis=1)
        position = labels[mask, :3]
        clipped = np.minimum(np.maximum(position, lower), upper)
        drift = np.linalg.norm(position - clipped, axis=1)
        result[name] = {
            "frames": int(mask.sum()),
            "fraction_outside_authored_position_envelope": float(np.mean(drift > 1.0e-5)),
            "fraction_more_than_20mm_outside": float(np.mean(drift > 0.020)),
            "distance_outside_mm": {
                "p50": float(1_000.0 * np.quantile(drift, 0.50)),
                "p95": float(1_000.0 * np.quantile(drift, 0.95)),
                "p99": float(1_000.0 * np.quantile(drift, 0.99)),
                "max": float(1_000.0 * drift.max()),
            },
        }
    return result


def _visual_features(images: np.ndarray, batch_size: int) -> torch.Tensor:
    features = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start : start + batch_size]).to(torch.float32).mean(dim=-1).unsqueeze(1)
        pooled = F.adaptive_avg_pool2d(batch, (16, 16)).flatten(start_dim=1)
        pooled = pooled - pooled.mean(dim=1, keepdim=True)
        features.append(F.normalize(pooled, dim=1).cpu())
    return torch.cat(features, dim=0)


def _nearest_visual_neighbor(
    images: np.ndarray, labels: np.ndarray, split: int, batch_size: int
) -> dict[str, object]:
    features = _visual_features(images, batch_size)
    reference = features[:split]
    query = features[split:]
    similarities = []
    neighbor_indices = []
    for start in range(0, len(query), 256):
        similarity = query[start : start + 256] @ reference.T
        value, index = similarity.max(dim=1)
        similarities.append(value)
        neighbor_indices.append(index)
    cosine = torch.cat(similarities).numpy()
    nearest = torch.cat(neighbor_indices).numpy()
    position_delta_mm = 1_000.0 * np.linalg.norm(labels[split:, :3] - labels[nearest, :3], axis=1)
    very_similar = cosine >= 0.99
    return {
        "comparison": "each held-out frame to its nearest 16x16 grayscale train feature",
        "cosine_similarity": _quantiles(cosine),
        "position_label_delta_mm": {
            "mean": float(position_delta_mm.mean()),
            **_quantiles(position_delta_mm),
        },
        "frames_with_cosine_at_least_0_99": int(very_similar.sum()),
        "fraction_of_very_similar_pairs_over_20mm_apart": (
            None if not np.any(very_similar) else float(np.mean(position_delta_mm[very_similar] > 20.0))
        ),
        "interpretation_limit": (
            "This coarse feature is an ambiguity diagnostic, not a learned perceptual metric; high label "
            "separation among near-identical features is evidence against merely adding network capacity."
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    data = np.load(args.dataset)
    images = data["images"]
    labels = data["labels"].astype(np.float32)
    occupancy = data["occupancy"].astype(np.float32)
    split = int(len(images) * (1.0 - args.validation_fraction))

    head = load_pose_head(args.checkpoint, args.device)
    predictions = []
    occupancy_predictions = []
    for start in range(0, len(images), args.batch_size):
        batch = torch.from_numpy(images[start : start + args.batch_size]).to(args.device, dtype=torch.float32).div_(255.0)
        with torch.inference_mode():
            pose, logits = head.forward_with_occupancy(batch)
        predictions.append(pose.cpu().numpy())
        occupancy_predictions.append((logits > 0.0).cpu().numpy())
    prediction = np.concatenate(predictions)
    predicted_occupancy = np.concatenate(occupancy_predictions)
    error = prediction - labels

    training_evidence = json.loads(args.training_report.read_text(encoding="utf-8"))
    recorded_checkpoint_sha = training_evidence.get("checkpoint_sha256", "")
    num_envs = int(data["collection_num_envs"].item())
    rounds = np.arange(len(images)) // num_envs
    environments = np.arange(len(images)) % num_envs
    occupancy_exact = np.all(predicted_occupancy == (occupancy > 0.5), axis=1)

    report = {
        "status": "diagnostic",
        "title": "Overview pose-head generalization failure audit",
        "evidence_type": "simulation_perception_failure_analysis",
        "identity": {
            "dataset": str(args.dataset),
            "dataset_sha256": _sha256(args.dataset),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
            "training_report": str(args.training_report),
            "training_report_checkpoint_hash_matches": checkpoint_matches_sha256(
                args.checkpoint, recorded_checkpoint_sha
            ),
            "architecture_version": head.architecture_version,
            "feature_grid_size": head.feature_grid_size,
        },
        "split_and_order": {
            "frames": len(images),
            "training_frames": split,
            "held_out_frames": len(images) - split,
            "collection_num_envs": num_envs,
            "row_order": "round-major, then environment index",
            "training_round_range_inclusive": [int(rounds[0]), int(rounds[split - 1])],
            "held_out_round_range_inclusive": [int(rounds[split]), int(rounds[-1])],
            "held_out_first_environment": int(environments[split]),
            "held_out_last_environment": int(environments[-1]),
            "round_and_environment_arrays_inferred": True,
            "note": (
                "The collector version stored num_envs but not row-level round/environment IDs; these are "
                "unambiguous because it appends one complete vectorized batch per round and only truncates the tail."
            ),
        },
        "model_error": {
            "training": _error_metrics(error[:split]),
            "held_out": _error_metrics(error[split:]),
            "occupancy_exact_match_training": float(occupancy_exact[:split].mean()),
            "occupancy_exact_match_held_out": float(occupancy_exact[split:].mean()),
            "training_curve_context": {
                "selected_epoch": training_evidence.get("selected_epoch"),
                "selected_epoch_train_loss": next(
                    (
                        row.get("train_loss")
                        for row in training_evidence.get("history", [])
                        if row.get("epoch") == training_evidence.get("selected_epoch")
                    ),
                    None,
                ),
                "final_epoch_train_loss": (
                    None
                    if not training_evidence.get("history")
                    else training_evidence["history"][-1].get("train_loss")
                ),
            },
        },
        "distribution_comparison": {
            "training_labels": _label_distribution(labels[:split]),
            "held_out_labels": _label_distribution(labels[split:]),
            "training_image_signal_proxy": _image_signal(images[:split]),
            "held_out_image_signal_proxy": _image_signal(images[split:]),
            "lighting_metadata_available": False,
            "lighting_note": (
                "Per-frame sun parameters were not recorded. Frame mean/std are reported only as observable "
                "lighting/exposure proxies and are not presented as the actual randomized sun state."
            ),
        },
        "image_label_alignment": {
            "held_out_occupancy_exact_match": float(occupancy_exact[split:].mean()),
            "what_it_establishes": (
                "The frame and label agree at bay-category scale; a one-round category shift would not preserve "
                "the rotating bay0/bay1/neither schedule. It does not establish millimetre pose observability."
            ),
            "nearest_cross_split_neighbor": _nearest_visual_neighbor(images, labels, split, args.batch_size),
        },
        "post_physics_drift": _envelope_drift(labels, occupancy),
        "diagnosis": {
            "capacity_hypothesis_supported": False,
            "findings": [
                "Architecture v2 drove the training objective down, but both selected-checkpoint train and held-out tails remain large.",
                "Train and held-out pose distributions and image-signal proxies are similar.",
                "The free-transfer sampler frequently moves outside its authored envelope during the physics/render wait.",
                "Occupancy remains easy because neither-bay classification does not require millimetre localization.",
            ],
            "recommended_pivot": (
                "Treat perception as part of the service interface: add a high-contrast fiducial with known geometry "
                "to the module, calibrate the overview camera, and solve pose geometrically (PnP, optionally depth) "
                "with reprojection/error rejection. Collect only collision-free, kinematically held poses and record "
                "the intended pose, rendered pose, round, environment, and lighting parameters. This is more "
                "industrial and falsifiable than another end-to-end RGB regressor capacity increase."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["model_error"], indent=2))
    print(json.dumps(report["post_physics_drift"], indent=2))
    print(f"[INFO] wrote {args.report}")


if __name__ == "__main__":
    main()
