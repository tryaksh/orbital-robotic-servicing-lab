"""The kinematics agreement check is per environment, and says which one.

The check compares the closed-form arm against the simulator's own tool frame
before any solved joint target is commanded, and it is right: 0.5 mm against a
chain that agrees to 0.006 mm, and it has caught real defects. **The tolerance is
not the problem and must not be widened.**

What was wrong is the reduction. It took ``.max()`` over *every* environment at
one instant, the first time any one of them reached a solved leg, so one
environment whose joints had not been written for that leg was enough to fail the
run -- about one in fifteen, costing a sweep point each time.

This is a source-level test, for the same reason the registry and post-init tests
are: whether a reduction is per environment is a property of the source, so
finding out needs no simulator. Rerunning the chain to discover it costs minutes
of GPU and a seed.

CPU only.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "run_workflow_demo.py"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(DEMO.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {DEMO.name}")


def _source(node: ast.AST) -> str:
    return ast.get_source_segment(DEMO.read_text(encoding="utf-8"), node) or ""


def test_the_check_is_given_the_environments_it_is_about_to_command() -> None:
    """It takes ``ids``, so it cannot look at environments nobody is driving."""

    check = _function("_check_forward_kinematics")
    arguments = [argument.arg for argument in check.args.args]
    assert arguments[:2] == ["self", "ids"], (
        "the check must receive the environments about to be commanded; without them it is back to "
        f"reducing over all of them, and its arguments are {arguments}"
    )


def test_the_call_site_passes_the_transiting_environments() -> None:
    source = DEMO.read_text(encoding="utf-8")
    assert "self._check_forward_kinematics(ids, tool, tool_rot)" in source, (
        "the transit step must hand the check the environments it is about to solve for"
    )


def test_no_reduction_collapses_every_environment_into_one_verdict() -> None:
    """A residual per environment, never one number for all of them.

    ``.abs().max()`` over the whole residual tensor is exactly the defect: it
    cannot tell a permuted joint list from one environment that has not moved yet.
    ``amax(dim=-1)`` keeps the environment axis, which is what lets the error name
    the environment and what stops a quiet one failing the run.
    """

    body = _source(_function("_check_forward_kinematics"))
    assert "amax(dim=-1)" in body, "the position residual must keep its environment axis"
    assert ".abs().max()" not in body, (
        "a whole-tensor .max() is the defect this test exists to prevent: it reduces every "
        "environment into one verdict, so an environment that has not been written yet fails the run"
    )
    assert "dim=-1" in body, "the attitude residual must keep its environment axis too"


def test_the_error_names_the_environment_and_how_many_disagree() -> None:
    """A failure a reader can act on: which environment, its own residual, how many."""

    body = _source(_function("_check_forward_kinematics"))
    assert "environment {int(pending[first])}" in body
    assert "int(disagrees.sum())" in body


def test_every_environment_is_validated_before_it_is_commanded() -> None:
    """The mask is the guarantee, and it is stronger than checking once was.

    Checking once meant the environments that reached a solved leg later were never
    validated at all. A per-environment mask validates each the first time it is
    about to be commanded, so none is ever driven from an unvalidated chain.
    """

    body = _source(_function("_check_forward_kinematics"))
    assert "self.solved_ik_forward_checked[ids]" in body, "pending environments come from the mask"
    assert "self.solved_ik_forward_checked[pending] = True" in body, "a checked environment is recorded"

    source = DEMO.read_text(encoding="utf-8")
    assert "self.solved_ik_forward_checked = torch.zeros(count, dtype=torch.bool" in source, (
        "the mask must be one boolean per environment"
    )


def test_the_tolerance_is_unchanged() -> None:
    """Pinned, because the fix must not have been a widened tolerance.

    0.5 mm and 1.0 mrad, against a closed form that agrees with the simulator's
    recorded configurations to 0.006 mm. If a future change to this check comes
    with a looser bound, that is a different change and needs its own argument.
    """

    tree = ast.parse(DEMO.read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("SOLVED_IK_FK_AGREEMENT"):
                    found[target.id] = float(node.value.value)

    assert found["SOLVED_IK_FK_AGREEMENT_M"] == pytest.approx(0.0005)
    assert found["SOLVED_IK_FK_AGREEMENT_RAD"] == pytest.approx(0.001)


def test_a_reset_does_not_forget_that_a_chain_was_validated() -> None:
    """The chain is a property of the robot, not of the episode.

    Joint order, link lengths and the tool offset do not change when an episode
    resets, so re-validating on every reset would only add work. ``reset_envs``
    clears per-episode state and deliberately leaves the accumulators; this mask
    belongs with the accumulators.
    """

    body = _source(_function("reset_envs"))
    assert "solved_ik_forward_checked" not in body
