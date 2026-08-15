from __future__ import annotations

from pathlib import Path

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


def test_the_live_grapple_interface_is_the_yoked_pin() -> None:
    """Assert which pin the three skills are trained and certified against.

    The yoke went live on 2026-08-14 because extraction certified at 0.00% for
    one reason: a single-point tapered pin clamped by flat pads cannot resist
    rotation about the closing axis. Every skill has to see the same interface,
    so the flag lives on the shared capture configuration and not on three
    separate ones. If this assertion is ever flipped back, flip the numbers in
    docs/status.md with it.
    """

    source = (
        ROOT / "src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "anti_yaw_yoke: bool = True" in source
    assert "self.scene.spare_blade.spawn.anti_yaw_yoke = self.anti_yaw_yoke" in source
    # Exactly one skill config may own it, or two skills could silently diverge.
    assert source.count("anti_yaw_yoke: bool =") == 1


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


def test_no_large_runtime_artifacts_are_tracked_by_layout() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for expected in (".deps/", "datasets/", "checkpoints/", "videos/", "*.hdf5", "*.onnx"):
        assert expected in ignored
