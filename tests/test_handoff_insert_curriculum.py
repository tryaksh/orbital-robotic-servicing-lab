"""The insertion curriculum expands difficulty only on measured success."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "handoff_curriculum_env_cfg.py"
REGISTRY = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py"
TRAIN = ROOT / "scripts" / "train_insert_handoff_curriculum.sh"


def test_reverse_curriculum_starts_where_v24_is_already_successful() -> None:
    source = CONFIG.read_text(encoding="utf-8")
    assert "START_FRONTIER_STATION = 6" in source
    assert '"threshold": 0.80' in source
    assert '"window_size": 256' in source
    assert '"minimum_frontier_steps": 1_600' in source
    assert 'reset.params["frontier_probability"] = 0.50' in source
    assert "mdp.reset_grapple_insert_stroke" in source


def test_reverse_curriculum_changes_no_controller_or_success_criterion() -> None:
    source = CONFIG.read_text(encoding="utf-8")
    assert "(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg)" in source
    for forbidden in ("rewards =", "terminations =", "episode_length_s =", "tolerance"):
        assert forbidden not in source


def test_reverse_curriculum_has_a_separate_task_and_timeboxed_resume() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    training = TRAIN.read_text(encoding="utf-8")
    assert "Isaac-ZeroG-Blade-GrapplePin-InsertHandoffCurriculum-v0" in registry
    assert "ZeroGBladeGrapplePinInsertHandoffCurriculumEnvCfg" in registry
    assert "Isaac-ZeroG-Blade-GrapplePin-InsertHandoffCurriculum-v0" in training
    assert "ep_2100_rew_43.909218.pth" in training
    assert "47AA9EFB60F7794BE5CDD1EBD0AD5EC0E94CE00345BCF975D83AE9418D9A1B9F" in training
    assert 'EPOCHS="${EPOCHS:-3300}"' in training
