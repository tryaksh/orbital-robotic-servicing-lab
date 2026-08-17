"""Train the module-pose head that replaces the simulator's answer.

Supervised regression from a 64x64 RGB frame to the module's pose, on frames
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
from zero_g_blade_swap.pose_head import MODULE_POSE_DIM, ModulePoseHead  # noqa: E402

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
    return parser


def main() -> None:
    args = _parser().parse_args()
    torch.manual_seed(args.seed)
    data = np.load(args.dataset)
    images = torch.from_numpy(data["images"])
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

    head = ModulePoseHead(output_dim=MODULE_POSE_DIM, occupancy_slots=occupancy_slots).to(args.device)
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

    best = float("inf")
    history: list[dict[str, float]] = []
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
        if entry["position_error_mm_mean"] < best:
            best = entry["position_error_mm_mean"]
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
    best_entry = min(history, key=lambda row: row["position_error_mm_mean"])
    report = {
        "status": "passed" if best_entry["position_error_mm_p95"] < CAPTURE_TOLERANCE_MM else "failed",
        "title": "Module pose regressed from a 64x64 servicing camera",
        "evidence_type": "simulation_perception_regression",
        "protocol": {
            "dataset": str(args.dataset),
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
            "position_error_mm_mean": best_entry["position_error_mm_mean"],
            "position_error_mm_p95": best_entry["position_error_mm_p95"],
            "orientation_error_rad_mean": best_entry["orientation_error_rad_mean"],
            "final_epoch_position_error_mm_mean": final["position_error_mm_mean"],
            **(
                {}
                if occupancy is None
                else {
                    "occupancy_bays": occupancy_slots,
                    "occupancy_accuracy": best_entry["occupancy_accuracy"],
                    "occupancy_exact_match": best_entry["occupancy_exact_match"],
                    # The prior a head could score by always answering "empty".
                    # Printed beside the accuracy so the two can be compared
                    # without going back to the dataset.
                    "occupancy_majority_class_rate": float(
                        torch.maximum(
                            validation_occupancy.mean(dim=0), 1.0 - validation_occupancy.mean(dim=0)
                        ).mean()
                    ),
                }
            ),
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
        "scope_and_limitations": [
            "Simulation only. This is a rendered camera, not a calibrated real one.",
            "Held-out frames come from the same collection run, so this measures generalisation across "
            "lighting, albedo, noise and module pose -- not across a different renderer or a real sensor.",
            "The number that matters is the workflow success rate with this head in the loop, which is "
            "certified separately.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] best held-out position error {best_entry['position_error_mm_mean']:.2f} mm mean, "
          f"{best_entry['position_error_mm_p95']:.2f} mm p95")
    print(f"[INFO] wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
