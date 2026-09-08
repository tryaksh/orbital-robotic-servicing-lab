"""Causal load diagnostics; deliberately separate from the raw 20 N job abort."""
from __future__ import annotations

import math


class TemporalLoadMonitor:
    """CPU reference for the planned batched GPU diagnostic.

    Input is wrist-force magnitude, so opposite vector directions cannot cancel.
    A smoothed overload is not by itself a witnessed contact jam.
    """

    def __init__(self, *, threshold_n=20.0, tau_s=0.020, sustain_s=0.025):
        if not all(math.isfinite(x) and x > 0 for x in (threshold_n, tau_s, sustain_s)):
            raise ValueError("Load diagnostic settings must be finite and positive")
        self.threshold = threshold_n
        self.tau = tau_s
        self.sustain = sustain_s
        self.filtered = None
        self.duration = 0.0
        self.peak = 0.0
        self.excess_impulse = 0.0
        self.raw_violation_seen = False
        self.sustained_seen = False

    def observe(self, force_n, dt):
        if not math.isfinite(force_n) or force_n < 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Expected finite nonnegative force and positive dt")
        if self.filtered is None:
            self.filtered = force_n
        else:
            self.filtered += -math.expm1(-dt / self.tau) * (force_n - self.filtered)
        self.duration = self.duration + dt if self.filtered > self.threshold else 0.0
        self.peak = max(self.peak, force_n)
        self.excess_impulse += max(0.0, force_n - self.threshold) * dt
        self.raw_violation_seen |= force_n > self.threshold
        self.sustained_seen |= self.duration + 1e-12 >= self.sustain
        return {"filtered_force_n": self.filtered, "sustained_overload_seen": self.sustained_seen,
                "raw_violation_seen": self.raw_violation_seen, "peak_raw_force_n": self.peak,
                "excess_impulse_n_s": self.excess_impulse}
