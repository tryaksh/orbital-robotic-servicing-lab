"""Manager-based RL environment with an opt-in pre-reset metrics hook.

``ManagerBasedRLEnv.step`` resets terminated environments before it returns, so
an evaluator that reads pose or velocity error afterwards measures the *next*
episode.  This subclass only intercepts ``_reset_idx``; with no hook installed
it behaves exactly like the stock environment, so training runs, reward terms,
observation/action dimensions, and existing checkpoints are unaffected.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from zero_g_blade_swap.evaluation import TerminalMetricsMixin


class TerminalMetricsManagerBasedRLEnv(TerminalMetricsMixin, ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` that can snapshot episodes before they reset."""


__all__ = ["TerminalMetricsManagerBasedRLEnv"]
