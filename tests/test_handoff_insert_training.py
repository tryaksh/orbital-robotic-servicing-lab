"""The insertion repair changes the reset distribution and nothing else."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "handoff_insert_env_cfg.py"
REGISTRY = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py"
TRAIN = ROOT / "scripts" / "train_insert_handoff.sh"


def test_handoff_task_inherits_the_certified_problem_and_only_forces_station_zero() -> None:
    source = CONFIG.read_text(encoding="utf-8")
    assert "(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg)" in source
    assert "HANDOFF_RESET_STATION = 0" in source
    assert "forced_station" in source
    for forbidden in ("rewards =", "terminations =", "episode_length_s =", "latch_"):
        assert forbidden not in source


def test_handoff_task_is_separately_registered() -> None:
    source = REGISTRY.read_text(encoding="utf-8")
    assert "Isaac-ZeroG-Blade-GrapplePin-InsertHandoff-v0" in source
    assert "handoff_insert_env_cfg:ZeroGBladeGrapplePinInsertHandoffEnvCfg" in source


def test_training_is_a_hash_bound_timeboxed_resume_of_v24() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    assert "Isaac-ZeroG-Blade-GrapplePin-InsertHandoff-v0" in source
    assert "ep_2100_rew_43.909218.pth" in source
    assert "47AA9EFB60F7794BE5CDD1EBD0AD5EC0E94CE00345BCF975D83AE9418D9A1B9F" in source
    assert "EPOCHS" in source and "2500" in source
    assert "--checkpoint" in source and "RESUME_CKPT" in source
