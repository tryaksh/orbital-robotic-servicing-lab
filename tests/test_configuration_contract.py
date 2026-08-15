from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_state_ppo_hyperparameters_are_shared_across_the_insertion_family() -> None:
    """Every insertion profile has to be trainable against every other one.

    The force-feedback experiment compared two arms that differed only in one
    observation; that comparison is only valid because the PPO configuration
    behind it is identical. Keep these four values in lockstep.
    """

    agents = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/agents"
    for name in (
        "rl_games_insertion.yaml",
        "rl_games_robust_insertion.yaml",
        "rl_games_contact_insertion.yaml",
        "rl_games_rigid_grasp.yaml",
    ):
        params = yaml.safe_load((agents / name).read_text(encoding="utf-8"))["params"]
        assert params["config"]["horizon_length"] == 32, name
        assert params["config"]["gamma"] == 0.99, name
        assert params["config"]["tau"] == 0.95, name
    # The contact profile deliberately halves the learning rate; every promoted
    # policy and both arms of the force-feedback ablation ran the rigid-grasp
    # configuration, so that is the one whose optimiser must not drift.
    rigid = yaml.safe_load((agents / "rl_games_rigid_grasp.yaml").read_text(encoding="utf-8"))["params"]
    assert rigid["config"]["learning_rate"] == 3.0e-4


def test_deleted_swap_task_left_nothing_behind() -> None:
    """The eight-phase swap task was deleted on 2026-08-10; keep it deleted.

    It is the single most likely thing to be reintroduced by accident, because
    the package is still called ``zero_g_blade_swap`` and the phrase reads as a
    project name rather than as a task. See CLAUDE.md.
    """

    source = ROOT / "src/zero_g_blade_swap"
    assert not (source / "tasks/blade_swap/mdp/commands.py").exists()
    assert not (source / "tasks/blade_swap/mdp/curricula.py").exists()
    assert not (source / "tasks/blade_swap/agents/rl_games_teacher.yaml").exists()
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "_swap_phase" not in text, path
        assert "BladeSwapCommand" not in text, path


def test_vision_config_is_memory_conscious() -> None:
    config = yaml.safe_load(
        (ROOT / "src/zero_g_blade_swap/tasks/blade_swap/agents/rl_games_vision.yaml").read_text(encoding="utf-8")
    )
    assert config["params"]["config"]["horizon_length"] == 32


def test_the_live_grapple_interface_is_declared_in_exactly_one_place() -> None:
    """Every skill must see the same interface, or none of them are comparable.

    Both the yoke flag and the latch live on the shared capture configuration
    rather than on three separate ones, and each skill re-applies the latch
    after it rebuilds its own event set. Certified 2026-08-15: the yoke is off
    because it cost insertion 67 points to buy extraction 0.13; the latch is on
    because a passive interface cannot oppose a moment about the closing axis.
    If either assertion is flipped, flip the numbers in docs/status.md with it.
    """

    source = (
        ROOT / "src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "anti_yaw_yoke: bool = False" in source
    assert "self.scene.spare_blade.spawn.anti_yaw_yoke = self.anti_yaw_yoke" in source
    assert "latch_enabled: bool = False" in source
    # One declaration each, or two skills could silently diverge.
    assert source.count("anti_yaw_yoke: bool =") == 1
    assert source.count("latch_enabled: bool =") == 1
    # Capture and the three skills each rebuild the event set in their own
    # configure_robustness, so every one of the four re-applies the latch.
    assert source.count("self._configure_latch()") == 4


def test_the_plain_pin_stays_rebuildable() -> None:
    """The baseline evidence has to describe something that can be rebuilt.

    ``evidence/grapple_pin_axial_pull_gate.json`` and
    ``evidence/grapple_pin_yaw_probe_railed_plain.json`` were measured on the
    plain pin. Turning the yoke on at the task level must not erase the ability
    to reproduce them, so the blade spawn's own default stays off and the
    diagnostics script keeps a switch that forces it off.
    """

    assets = (ROOT / "src/zero_g_blade_swap/tasks/blade_swap/assets.py").read_text(encoding="utf-8")
    assert "anti_yaw_yoke: bool = False" in assets
    diagnostics = (ROOT / "scripts/grasp_diagnostics.py").read_text(encoding="utf-8")
    assert '"--plain_pin"' in diagnostics
    gates = (ROOT / "scripts/run_yoke_gates.sh").read_text(encoding="utf-8")
    assert "--plain_pin" in gates


def test_extraction_velocity_limits_are_derived_from_the_settling_window() -> None:
    """A skill may not declare success in a state the chain cannot confirm.

    The chained workflow stops commanding the moment a phase succeeds and
    re-checks the same condition 0.70 s later. In zero gravity nothing damps
    what is left moving, so a module declared removed at velocity ``v`` drifts
    ``v * t`` before it is judged. The old limits -- 0.30 rad/s and 0.10 m/s --
    allowed 0.210 rad and 70 mm of drift against a 0.20 rad and 20 mm tolerance,
    which is 105% and 350% of the budget: mathematically guaranteed to fail.
    Measured exactly there, 191 of 192 chained removals fired their predicate
    and not one survived the re-check.

    So the limits are computed from the window rather than chosen, and this test
    exists to stop anyone replacing the derivation with the number it currently
    produces.
    """

    source = (ROOT / "src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py").read_text(encoding="utf-8")
    assert "EXTRACTION_ANGULAR_VELOCITY_LIMIT = 0.5 * CAPTURE_ATTITUDE_TOLERANCE_RAD / WORKFLOW_SETTLE_S" in source
    assert "EXTRACTION_LINEAR_VELOCITY_LIMIT = 0.5 * CAPTURE_POSITION_TOLERANCE_M / WORKFLOW_SETTLE_S" in source

    # Half the tolerance spent on drift, half left for the pull's own error.
    settle, attitude, position = 0.70, 0.20, 0.020
    assert (0.5 * attitude / settle) * settle == pytest.approx(0.5 * attitude)
    assert (0.5 * position / settle) * settle == pytest.approx(0.5 * position)

    # The workflow driver must read both limits and the window, never restate
    # them. Two constants restated instead of read have each cost this chain a
    # full certification.
    demo = (ROOT / "scripts/run_workflow_demo.py").read_text(encoding="utf-8")
    assert "EXTRACTION_ANGULAR_VELOCITY_LIMIT" in demo
    assert "EXTRACTION_LINEAR_VELOCITY_LIMIT" in demo
    assert "SETTLE_STEPS = round(WORKFLOW_SETTLE_S * 30.0)" in demo


def test_the_workflow_reads_action_scales_from_the_tasks() -> None:
    """The chain must drive each policy with the action term it trained under.

    Extract's scales were rebalanced on the task and the workflow's copy stayed
    at the old values, so the chain drove the policy at a quarter of its lateral
    authority: all 598 removals overran their budget while the same checkpoint
    certified at 94.23% alone. Reading beats restating.
    """

    workflow = (
        ROOT / "src/zero_g_blade_swap/tasks/blade_swap/workflow_demo_env_cfg.py"
    ).read_text(encoding="utf-8")
    for name in ("GRASP_ACTION_SCALE", "EXTRACT_ACTION_SCALE", "INSERT_ACTION_SCALE"):
        assert f"{name} = _certified_action_scale(" in workflow, name
    # No literal tuple of scales may survive next to them.
    assert "EXTRACT_ACTION_SCALE = (" not in workflow


def test_no_large_runtime_artifacts_are_tracked_by_layout() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for expected in (".deps/", "datasets/", "checkpoints/", "videos/", "*.hdf5", "*.onnx"):
        assert expected in ignored
