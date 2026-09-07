"""Inspect a trusted, locally generated PPO checkpoint without claiming policy competence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-epoch", required=True, type=int)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite checkpoint validation")
    import torch

    # Only use on checkpoints produced by this project on this machine.
    value = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = value.get("model", {})
    finite = bool(model) and all(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()) for tensor in model.values())
    epoch = int(value.get("epoch", -1))
    passed = finite and epoch == args.expected_epoch
    report = {"status": "passed" if passed else "failed", "checkpoint": str(args.checkpoint),
              "epoch": epoch, "expected_epoch": args.expected_epoch, "frame": value.get("frame"),
              "model_tensor_count": len(model), "model_parameters_finite": finite,
              "scope": "Checkpoint structure/load check, not policy inference or task-success validation."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
