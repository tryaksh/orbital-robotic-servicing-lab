"""Offline behavioural cloning for the multimodal vision actor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required to read demonstration datasets") from exc

from zero_g_blade_swap.tasks.blade_swap.agents.network import VisionActor


class Demonstrations(Dataset):
    """Lazily read HDF5 so the 250k-image dataset never resides in system RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: h5py.File | None = None
        with h5py.File(path, "r") as handle:
            required = {"rgb", "proprio", "phase", "teacher_action"}
            missing = required.difference(handle)
            if missing:
                raise KeyError(f"Dataset is missing {sorted(missing)}")
            lengths = {len(handle[key]) for key in required}
            if len(lengths) != 1:
                raise ValueError(f"Dataset arrays have inconsistent lengths: {lengths}")
            self.length = lengths.pop()
            self.rgb_shape = tuple(handle["rgb"].shape[1:])
            self.proprio_shape = tuple(handle["proprio"].shape[1:])
            self.phase_shape = tuple(handle["phase"].shape[1:])
            self.action_dim = int(np.prod(handle["teacher_action"].shape[1:]))

    def __len__(self) -> int:
        return self.length

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        handle = self._handle()
        return {
            "rgb": torch.from_numpy(handle["rgb"][index]).float().div_(255.0),
            "proprio": torch.from_numpy(handle["proprio"][index]).float(),
            "phase": torch.from_numpy(handle["phase"][index]).float(),
            "action": torch.from_numpy(handle["teacher_action"][index]).float().flatten(),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/teacher_250k.h5"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/vision_bc.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument("--gripper_loss_weight", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=0, help="Keep zero on Windows unless HDF5 is thread-safe.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def _loss(prediction: torch.Tensor, target: torch.Tensor, gripper_weight: float) -> tuple[torch.Tensor, dict]:
    if prediction.shape[-1] < 2:
        raise ValueError("Expected at least one arm action and one gripper action")
    arm = nn.functional.mse_loss(prediction[:, :-1], target[:, :-1])
    gripper_target = target[:, -1].clamp(-1.0, 1.0).add(1.0).mul(0.5)
    gripper = nn.functional.binary_cross_entropy_with_logits(prediction[:, -1], gripper_target)
    total = arm + gripper_weight * gripper
    return total, {"arm_mse": arm.detach(), "gripper_bce": gripper.detach()}


def _run_epoch(model, loader, device, optimizer, scaler, gripper_weight: float) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "arm_mse": 0.0, "gripper_bce": 0.0, "samples": 0}
    for batch in loader:
        observation = {
            "rgb": batch["rgb"].to(device, non_blocking=True),
            "proprio": batch["proprio"].to(device, non_blocking=True),
        }
        target = batch["action"].to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(observation)
            loss, components = _loss(prediction, target, gripper_weight)
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        amount = target.shape[0]
        totals["loss"] += float(loss.detach()) * amount
        totals["arm_mse"] += float(components["arm_mse"]) * amount
        totals["gripper_bce"] += float(components["gripper_bce"]) * amount
        totals["samples"] += amount
    return {key: value / totals["samples"] for key, value in totals.items() if key != "samples"}


def main() -> None:
    args = _parser().parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation_fraction must be between 0 and 0.5")
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = Demonstrations(dataset_path)
    indices = np.random.default_rng(args.seed).permutation(len(dataset))
    validation_count = max(1, int(len(dataset) * args.validation_fraction))
    validation = Subset(dataset, indices[:validation_count].tolist())
    training = Subset(dataset, indices[validation_count:].tolist())
    pin_memory = str(args.device).startswith("cuda")
    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    device = torch.device(args.device)
    model = VisionActor(
        {"rgb": dataset.rgb_shape, "proprio": dataset.proprio_shape},
        dataset.action_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, scaler, args.gripper_loss_weight)
        with torch.inference_mode():
            validation_metrics = _run_epoch(model, validation_loader, device, None, scaler, args.gripper_loss_weight)
        scheduler.step()
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record))
        if validation_metrics["loss"] < best:
            best = validation_metrics["loss"]
            torch.save(
                {
                    "format": "zero-g-blade-swap-vision-bc-v1",
                    "actor_state_dict": model.state_dict(),
                    "observation_shapes": {"rgb": dataset.rgb_shape, "proprio": dataset.proprio_shape},
                    "phase_shape": dataset.phase_shape,
                    "actions_num": dataset.action_dim,
                    "validation_loss": best,
                    "dataset": str(dataset_path),
                    "epoch": epoch,
                },
                args.output,
            )
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[INFO] Best BC actor: {args.output.resolve()} (validation loss {best:.6f})")


if __name__ == "__main__":
    main()
