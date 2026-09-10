"""Fixed failure exposure and action-ownership masks for recovery teaching.

These helpers choose training jobs, never failure outcomes or evaluation cases.
The caller executes a real scripted prefix inside the same measured job and
charges every prefix, terminal, and absorbing simulator step. No state reset or
fault resampling is performed here. Prefix actions are never PPO samples.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

import torch


def exposure_mask(cases, *, seed, cohort, training_seeds, fraction=0.5):
    """Return a deterministic, stratified list selecting only training faults.

    Hamilton apportionment selects the nearest whole-job total (half upward),
    with stable bin-ID ties. Within each bin a separate local RNG chooses jobs;
    global Python/torch RNG state and incoming case order do not affect choices.
    Nominal cases are always learned directly. The caller must pass the frozen
    training-seed list, not a list assembled from the requested case seeds.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in training_seeds:
        raise ValueError("Exposure requires a declared training seed")
    if isinstance(cohort, bool) or not isinstance(cohort, int) or cohort < 0:
        raise ValueError("Exposure requires a nonnegative integer cohort")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Exposure fraction must be finite and in [0, 1]")
    groups = defaultdict(list)
    seen, bin_families = set(), {}
    for index, case in enumerate(cases):
        if case.get("split") != "training" or case.get("seed") != seed:
            raise ValueError("Only this seed's training cases may receive exposure")
        case_id, bin_id, family = (case.get(key) for key in ("case_id", "bin_id", "family"))
        if not all(isinstance(value, str) and value for value in (case_id, bin_id, family)):
            raise ValueError("Each case needs nonempty case_id, bin_id and family")
        if case_id in seen:
            raise ValueError("Exposure case IDs must be unique")
        seen.add(case_id)
        if bin_id in bin_families and bin_families[bin_id] != family:
            raise ValueError("A fault bin cannot contain different families")
        bin_families[bin_id] = family
        if family != "nominal":
            groups[bin_id].append((case_id, index))
    target = math.floor(sum(map(len, groups.values())) * fraction + 0.5)
    quotas = {bin_id: math.floor(len(rows) * fraction) for bin_id, rows in groups.items()}
    remainder_order = sorted(groups, key=lambda bin_id: (-(len(groups[bin_id]) * fraction - quotas[bin_id]), bin_id))
    for bin_id in remainder_order[:target - sum(quotas.values())]:
        quotas[bin_id] += 1
    chosen = [False] * len(cases)
    for bin_id, rows in groups.items():
        rows = sorted(rows)
        random.Random(f"recovery-teaching-v1:{seed}:{cohort}:{bin_id}").shuffle(rows)
        for _, index in rows[:quotas[bin_id]]:
            chosen[index] = True
    return chosen


def _boolean_tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")


def scripted_prefix_mask(exposure, *, step, prefix_controls):
    """Select script-owned slots before fixed handoff; zero-based policy clock.

    Terminal slots can remain selected here, but the simulator must hold their
    commands and mask their rewards normally. Use policy_transition_valid with
    the independently recorded pre-step active mask to determine learning rows.
    """
    _boolean_tensor(exposure, "exposure")
    if exposure.ndim != 1:
        raise ValueError("Exposure must contain one boolean per environment")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (step, prefix_controls)):
        raise ValueError("Step and prefix length must be nonnegative integers")
    return exposure.clone() if step < prefix_controls else torch.zeros_like(exposure)


def policy_transition_valid(active_before, scripted):
    """Keep true on-policy actions, including their terminal transition.

    Masks must have identical shape and device: accidental broadcasting could
    silently credit an off-policy action. This mask is for optimization only;
    it must not replace the physical job's activity or terminal record.
    """
    _boolean_tensor(active_before, "active_before")
    _boolean_tensor(scripted, "scripted")
    if active_before.shape != scripted.shape or active_before.device != scripted.device:
        raise ValueError("Activity and action-ownership masks must match exactly")
    return active_before & ~scripted


def mask_scripted_rollout(rollout, scripted):
    """Shallow-copy a time-major PPO rollout with continuation-only validity.

    The existing finite_job_gae and unclipped PPO consume this validity for
    advantages, policy/value loss and normalization statistics. Prefix rewards,
    including scripted completions, earn no policy credit; active continuations
    retain their own terminal rewards. Physical records and cost are untouched.

    Within each rollout, scripts may only precede each environment's first
    learned action. The caller must use the global job clock when constructing
    scripted_prefix_mask, including across PPO chunk boundaries. A later script
    intervention needs a separately designed return
    boundary; excluding that action alone would bootstrap into an off-policy
    continuation. Such an intervention is rejected here.
    """
    active = rollout.get("valid")
    _boolean_tensor(active, "rollout validity")
    if active.ndim != 2:
        raise ValueError("Rollout validity must be [time, environment]")
    valid = policy_transition_valid(active, scripted)
    learned_already = valid.to(torch.int64).cumsum(dim=0) > 0
    if bool((learned_already & active & scripted).any()):
        raise ValueError("Scripted control may only precede learned control")
    return {**rollout, "valid": valid}


def make_prefix_action_targets(initial_actor, *, position_bounds, seated_height_m):
    """Precompute vectorized hold/insert actions from the actor interface only.

    Before four seconds, ActorRetryController cannot enter any retry phase:
    it holds its clamped initial action for one second, then uses constant XYZ
    insertion targets and the unchanged initial orientation/termination actions.
    Neither contact truth nor the ongoing actor observations enter this prefix.
    """
    if not isinstance(initial_actor, torch.Tensor) or initial_actor.ndim != 2 or initial_actor.shape[1] != 24:
        raise ValueError("Initial actor observations must be [environment, 24]")
    if not initial_actor.is_floating_point() or not bool(torch.isfinite(initial_actor).all()):
        raise ValueError("Initial actor observations must be finite floating point")
    if len(position_bounds) != 3 or not all(math.isfinite(x) and x > 0 for x in position_bounds):
        raise ValueError("Three finite positive position bounds are required")
    if not math.isfinite(seated_height_m):
        raise ValueError("Seated height must be finite")
    hold = initial_actor[:, -7:].clamp(-1, 1).clone()
    insert = hold.clone()
    insert[:, :2] = 0
    insert[:, 2] = max(-1.0, min(1.0, seated_height_m / position_bounds[2]))
    return hold, insert


def prefix_action_at_step(hold, insert, *, step, step_dt, hold_s=1.0, first_attempt_s=4.0):
    """Return the immutable precomputed action tensor before the fixed handoff."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Prefix step must be a nonnegative integer")
    if not all(math.isfinite(x) for x in (step_dt, hold_s, first_attempt_s)) or step_dt <= 0 or not 0 <= hold_s < first_attempt_s:
        raise ValueError("Prefix timing must be finite and ordered")
    elapsed = step * step_dt
    if elapsed >= first_attempt_s:
        raise ValueError("First-attempt prefix cannot command actions after handoff")
    return hold if elapsed < hold_s else insert
