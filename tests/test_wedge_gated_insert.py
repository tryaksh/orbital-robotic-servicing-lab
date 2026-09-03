"""The wedge-gated seating task must differ from the trained one in exactly one way.

`v24rack` certifies 36.77% on its own reset bank and 0.00% on 96 recorded
predecessor hand-offs. Six checkpoints of audit have not moved that, and three
objectives left the terminal attitude 0.4 mrad apart, so the angle is not the
reward's to give. The remaining structural difference between the skill and the
scripted guarded advance it must beat is that the advance refuses to push a
cocked module and the skill does not.

`mdp.wedged` is that refusal, expressed as a terminal condition derived from
`2c/theta` and from clearances measured out of the built configuration. It is
only interpretable if it is the *only* difference -- and in particular if the
observation width is unchanged, because a changed width would mean the run
started from scratch rather than from the frozen weights, and the comparison
would be with a different policy rather than with the same one under one rule.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap"
WEDGE_CFG = TASKS / "wedge_insert_env_cfg.py"
WEDGE_MDP = TASKS / "mdp" / "wedge.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_it_extends_the_task_v24_was_trained_on() -> None:
    source = _read(WEDGE_CFG)
    assert "class ZeroGBladeGrapplePinInsertWedgeGatedEnvCfg(ZeroGBladeGrapplePinInsertTwoSlotEnvCfg):" in source


def test_the_observation_width_is_untouched() -> None:
    """A changed observation cannot resume the frozen weights, so there is none."""

    source = _read(WEDGE_CFG)
    assert "observations" not in source.split('"""', 2)[-1], "the wedge task must not redefine observations"
    assert "wedge_margin" not in source.split('"""', 2)[-1], "the margin term is diagnostics, not an observation"


def test_the_only_added_termination_is_the_wedge() -> None:
    source = _read(WEDGE_CFG)
    body = source.split("class WedgeGatedInsertTerminationsCfg", 1)[1].split("@configclass", 1)[0]
    terms = re.findall(r"^\s{4}(\w+) = DoneTerm", body, flags=re.MULTILINE)
    base = _read(TASKS / "grapple_pin_env_cfg.py")
    base_body = base.split("class InsertTerminationsCfg", 1)[1].split("@configclass", 1)[0]
    base_terms = re.findall(r"^\s{4}(\w+) = DoneTerm", base_body, flags=re.MULTILINE)
    assert set(terms) - set(base_terms) == {"wedged"}
    assert set(base_terms) - set(terms) == set(), "a base termination was dropped"


def test_the_wedge_is_a_failure_and_not_a_timeout() -> None:
    """`time_out=True` would tell the value function the episode merely ran out."""

    source = _read(WEDGE_CFG)
    line = [row for row in source.splitlines() if "wedged = DoneTerm" in row]
    assert line and "time_out" not in line[0]


def test_the_law_reads_measured_clearances_and_divides_by_attitude() -> None:
    source = _read(WEDGE_MDP)
    assert "RELIEVED_LATERAL_CLEARANCE_M = 0.015678" in source
    assert "RELIEVED_VERTICAL_CLEARANCE_M = 0.012613" in source
    assert "WEDGE_CLEARANCE_M = min(" in source
    assert "(2.0 * WEDGE_CLEARANCE_M) / attitude.clamp_min(MINIMUM_ATTITUDE_RAD)" in source


def test_the_law_never_reads_anything_the_chain_does_not_have() -> None:
    """Attitude and depth are what the deployed estimator reports; nothing else."""

    source = _read(WEDGE_MDP)
    for forbidden in ("root_lin_vel", "root_ang_vel", "contact", "joint_pos"):
        assert forbidden not in source, f"the wedge law reads {forbidden}"


def test_both_arms_stay_registered() -> None:
    registry = _read(TASKS / "__init__.py")
    for task in (
        "Isaac-ZeroG-Blade-GrapplePin-InsertWedgeGated-v0",
        "Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0",
    ):
        assert f'"{task}"' in registry
