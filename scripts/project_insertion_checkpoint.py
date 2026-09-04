#!/usr/bin/env python3
"""Project the frozen insertion actor onto local assembly observations.

The v27 actor consumes 45 values. Thirteen identify absolute robot posture
rather than assembly error: six joint positions and the seven-value wrist pose.
This utility removes exactly those input columns from the actor, observation
normalizer, and Adam moments. Every other weight and training counter remains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

SOURCE_LAYOUT = (
    ("joint_pos", 6),
    ("joint_vel", 6),
    ("end_effector", 7),
    ("grip_error", 6),
    ("gripper_state", 2),
    ("blade_velocity", 6),
    ("previous_action", 6),
    ("blade_goal_error", 6),
)
DROP_TERMS = frozenset({"joint_pos", "end_effector"})
TARGET_LAYOUT = tuple(term for term in SOURCE_LAYOUT if term[0] not in DROP_TERMS)


def layout_width(layout: Iterable[tuple[str, int]]) -> int:
    return sum(width for _, width in layout)


def retained_feature_indices() -> tuple[int, ...]:
    retained: list[int] = []
    start = 0
    for name, width in SOURCE_LAYOUT:
        if name not in DROP_TERMS:
            retained.extend(range(start, start + width))
        start += width
    return tuple(retained)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_hash = args.expected_source_sha256.upper()
    actual_hash = sha256(args.source)
    if actual_hash != expected_hash:
        raise SystemExit(f"source SHA-256 is {actual_hash}, expected {expected_hash}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    import torch

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    mean_key = "running_mean_std.running_mean"
    var_key = "running_mean_std.running_var"
    weight_key = "a2c_network.trunk.0.weight"
    source_width = layout_width(SOURCE_LAYOUT)
    target_width = layout_width(TARGET_LAYOUT)
    keep = torch.tensor(retained_feature_indices(), dtype=torch.long)

    for key in (mean_key, var_key):
        if tuple(model[key].shape) != (source_width,):
            raise SystemExit(f"{key} has shape {tuple(model[key].shape)}, expected {(source_width,)}")
        model[key] = model[key].index_select(0, keep)

    first_weight = model[weight_key]
    if first_weight.ndim != 2 or first_weight.shape[1] != source_width:
        raise SystemExit(
            f"{weight_key} has shape {tuple(first_weight.shape)}, expected (*, {source_width})"
        )
    model[weight_key] = first_weight.index_select(1, keep)

    projected_moments = 0
    for state in checkpoint["optimizer"]["state"].values():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if value is not None and value.ndim == 2 and value.shape[1] == source_width:
                state[name] = value.index_select(1, keep)
                projected_moments += 1
    if projected_moments < 2:
        raise SystemExit("did not find both first-layer Adam moments to project")

    checkpoint["observation_projection"] = {
        "source_sha256": actual_hash,
        "source_layout": list(SOURCE_LAYOUT),
        "dropped_terms": sorted(DROP_TERMS),
        "target_layout": list(TARGET_LAYOUT),
        "source_width": source_width,
        "target_width": target_width,
        "retained_indices": retained_feature_indices(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    output_hash = sha256(args.output)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "source_sha256": actual_hash,
                "output": str(args.output),
                "output_sha256": output_hash,
                "source_width": source_width,
                "target_width": target_width,
                "projected_optimizer_moments": projected_moments,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
