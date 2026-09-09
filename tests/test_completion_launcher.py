import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("completion_launcher", ROOT / "scripts/run_completion.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_registered_fresh_budget_counts_initialization(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_completion", "plan", "--run-id", "plan-only"])
    assert launcher.main() == 0
    plan = json.loads(capsys.readouterr().out)["specification"]
    assert plan["cohorts"] == 22
    assert plan["rollout_control_transitions"] == 10137600
    assert plan["expected_initialization_physics_env_steps"] == 1509376
    assert plan["expected_charged_control_equivalent_transitions"] == 10326272


@pytest.mark.parametrize("arguments", [
    ["--seed", "10070"], ["--seed", "20070"], ["--cohorts", "21"],
    ["--max-minutes", "151"], ["--checkpoint", "existing.pt"],
])
def test_undeclared_limits_or_resume_are_rejected(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["run_completion", "plan", "--run-id", "refuse", *arguments])
    with pytest.raises(SystemExit):
        launcher.main()
