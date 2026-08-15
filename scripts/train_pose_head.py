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

    split = int(len(images) * (1.0 - args.validation_fraction))
    if split < 1 or split >= len(images):
        raise ValueError("validation_fraction leaves no data on one side of the split")
    print(f"[INFO] {split} training frames, {len(images) - split} held out")

    # Statistics from the training half only. Computing them over everything
    # would leak the validation set into the normalisation.
    mean = labels[:split].mean(dim=0)
    std = labels[:split].std(dim=0).clamp_min(1e-6)

    head = ModulePoseHead(output_dim=MODULE_POSE_DIM).to(args.device)
    head.label_mean.copy_(mean.to(args.device))
    head.label_std.copy_(std.to(args.device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()

    loader = DataLoader(
        TensorDataset(images[:split], labels[:split]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_images = images[split:].to(args.device)
    validation_labels = labels[split:].to(args.device)

    best = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(args.device, non_blocking=True).to(torch.float32).div_(255.0)
            batch_labels = batch_labels.to(args.device, non_blocking=True)
            predicted = head(batch_images)
            # Loss in normalised space so the three position channels and the
            # three rotation channels contribute comparably.
            loss = loss_fn((predicted - head.label_mean) / head.label_std, (batch_labels - head.label_mean) / head.label_std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch_images)
        schedule.step()

        head.eval()
        with torch.inference_mode():
            errors = []
            for start in range(0, len(validation_images), 1024):
                chunk = validation_images[start : start + 1024].to(torch.float32).div_(255.0)
                errors.append(head(chunk) - validation_labels[start : start + 1024])
            error = torch.cat(errors, dim=0)
        position_mm = 1_000.0 * torch.linalg.vector_norm(error[:, :3], dim=-1)
        entry = {
            "epoch": epoch,
            "train_loss": total / max(1, split),
            "position_error_mm_mean": float(position_mm.mean()),
            "position_error_mm_p95": float(position_mm.quantile(0.95)),
            "orientation_error_rad_mean": float(torch.linalg.vector_norm(error[:, 3:], dim=-1).mean()),
        }
        history.append(entry)
        if entry["position_error_mm_mean"] < best:
            best = entry["position_error_mm_mean"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), args.output)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[INFO] epoch {epoch:3d}  loss {entry['train_loss']:.5f}  "
                f"position {entry['position_error_mm_mean']:.2f} mm  p95 {entry['position_error_mm_p95']:.2f} mm",
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
