"""Split-safe case sampling at cohort boundaries; never in the physics hot path."""
from __future__ import annotations

import math
import random

from assembly_recovery.faults import assert_training_case, validate_fault_support


def case_from_bin(support, item, value, angle, *, seed, case_id, split):
    case = {"case_id": case_id, "split": split, "seed": seed, "bin_id": item["id"], "family": item["family"],
            "fixture_bias_xy_m": [0., 0.], "approach_xy_m": [0., 0.],
            "fixed_static_friction": support["nominal"]["fixed_static_friction"],
            "fixed_dynamic_friction": support["nominal"]["fixed_dynamic_friction"]}
    xy = [value * math.cos(angle), value * math.sin(angle)]
    if item["family"] == "lateral_pose_bias":
        case["fixture_bias_xy_m"] = xy
    elif item["family"] == "approach_offset":
        case["approach_xy_m"] = xy
    elif item["family"] == "friction":
        case["fixed_static_friction"] = value
    return case


def uniform_training_cases(study, seed, cohort, num_envs):
    if seed not in study["training"]["seeds"] or cohort < 0 or num_envs < 4 or num_envs % 4:
        raise ValueError("Training requires a declared training seed, nonnegative cohort and multiples of four")
    support = study["fault_support"]
    bins = validate_fault_support(support)
    nominal = [b for b in bins if b["family"] == "nominal"]
    faults = [b for b in bins if b["family"] != "nominal"]
    if len(nominal) != 1 or study["training"]["nominal_fraction"] != 0.25:
        raise ValueError("Exactly one nominal bin and 25% nominal practice are required")
    rng = random.Random(f"uniform-v1:{seed}:{cohort}")
    items = nominal * (num_envs // 4)
    items += [faults[(i + cohort * (3 * num_envs // 4)) % len(faults)] for i in range(3 * num_envs // 4)]
    rng.shuffle(items)
    cases = []
    for i, item in enumerate(items):
        value = rng.uniform(*item["range"])
        case = case_from_bin(support, item, value, rng.uniform(0, 2 * math.pi), seed=seed,
                             case_id=f"train-{seed}-{cohort}-{i}", split="training")
        assert_training_case(study, seed, case)
        cases.append(case)
    return cases


def support_grid_cases(study, seed, grid):
    if seed not in study["splits"]["development_seeds"] or grid not in {"low", "interior", "high"}:
        raise ValueError("Support probes use declared development seeds and fixed grids")
    support = study["fault_support"]
    bins = validate_fault_support(support)
    cases = []
    for item in bins:
        low, high = item["range"]
        value = {"low": low, "interior": (low + high) / 2, "high": high}[grid]
        count = 8 if item["family"] == "nominal" else 4
        for direction in range(count):
            angle = direction * math.pi / 2
            cases.append(case_from_bin(support, item, value, angle, seed=seed, split="development",
                                       case_id=f"dev-grid-{seed}-{grid}-{item['id']}-{direction}"))
    return cases
