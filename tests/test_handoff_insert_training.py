"""The insertion repair changes the reset distribution and nothing else."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "handoff_insert_env_cfg.py"
REGISTRY = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py"
TRAIN = ROOT / "scripts" / "train_insert_handoff.sh"


def test_handoff_task_inherits_the_certified_problem_and_uses_a_recorded_state() -> None:
    source = CONFIG.read_text(encoding="utf-8")
    assert "(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg)" in source
    assert "HANDOFF_RESET_STATION = 0" in source
    assert "forced_station" in source
    assert "RECORDED_HANDOFF_ARM_JOINTS" in source
    assert "RECORDED_HANDOFF_BLADE_POSE" in source
    assert "RECORDED_HANDOFF_ROBOT_ROOT_Y_M = -0.239" in source
    assert "CEC9E51E076486136E24484375B1C5D35E4181CDABD8AAF4258904000FAD6B31" in source
    assert "97d1ab3f409ebe5b7f395e6457cb20fd613f0401" in source
    for forbidden in ("rewards =", "terminations =", "episode_length_s =", "latch_"):
        assert forbidden not in source


def test_handoff_task_is_separately_registered() -> None:
    source = REGISTRY.read_text(encoding="utf-8")
    assert "Isaac-ZeroG-Blade-GrapplePin-InsertHandoff-v0" in source
    assert "handoff_insert_env_cfg:ZeroGBladeGrapplePinInsertHandoffEnvCfg" in source


def test_training_is_a_hash_bound_timeboxed_resume_of_v27() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    assert "Isaac-ZeroG-Blade-GrapplePin-InsertHandoff-v0" in source
    assert "ep_2300_rew_95.05724.pth" in source
    assert "010E9D14B9E6C22F99B699820C349DAE3B184436C542615088B43F3B03FD1408" in source
    assert "EPOCHS" in source and "2500" in source
    assert "--checkpoint" in source and "RESUME_CKPT" in source
