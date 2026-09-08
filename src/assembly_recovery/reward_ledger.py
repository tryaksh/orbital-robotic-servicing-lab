"""Discounted job returns from saved upstream control-step rewards."""

from __future__ import annotations

import math


def discounted_job_return(controls, job_index, terminal_physics_step, decimation, gamma=0.995):
    """Exclude endpoints after termination, including the partial terminal step.

    Upstream rewards are sampled only at control endpoints. A terminal event
    inside a decimation interval has no exact upstream endpoint reward; omit
    that interval and disclose the cutoff. All later rewards are absorbing zero.
    """
    if not 0 < gamma <= 1 or terminal_physics_step < 0 or decimation <= 0:
        raise ValueError("Invalid discount or trajectory timing")
    terms, total, count, error = {}, 0.0, 0, 0.0
    for expected, row in enumerate(controls, 1):
        if row["step"] != expected:
            raise ValueError("Control samples must be consecutive")
        reward = row["reward"][job_index]
        components = {k: v[job_index] for k, v in row["reward_terms"].items()}
        if not math.isfinite(reward) or not all(math.isfinite(v) for v in components.values()):
            raise ValueError("Nonfinite reward")
        error = max(error, abs(sum(components.values()) - reward))
        if error > 1e-5:
            raise ValueError("Reward components differ from upstream total")
        if expected * decimation > terminal_physics_step:
            continue
        discount = gamma ** (expected - 1)
        total += discount * reward
        count += 1
        for name, value in components.items():
            terms[name] = terms.get(name, 0.0) + discount * value
    if len(controls) * decimation < terminal_physics_step:
        raise ValueError("Trace ends before job termination")
    return {"gamma": gamma, "discounted_return": total, "discounted_terms": terms,
            "included_control_endpoints": count,
            "terminal_partial_physics_steps_without_endpoint_reward": terminal_physics_step % decimation,
            "excluded_control_endpoints": len(controls) - count,
            "max_reward_reconstruction_error": error,
            "terminal_handling": "Only control reward endpoints at/before evaluator termination; later rewards zero, no bootstrap. Partial terminal interval omitted. Training integration is not implemented."}
