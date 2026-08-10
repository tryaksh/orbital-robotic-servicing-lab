"""Simulator-free contracts for the pose-belief task's numbers.

Isaac Lab configurations cannot be constructed without the Kit runtime, so a
mistake in this arithmetic would otherwise only surface on a full simulator
launch, or worse, silently inside a training run. Everything here runs on a
laptop in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_g_blade_swap.math_utils import belief_pose_error, update_sampling_bound


def _bound(successes, bound=0.0, **kwargs):
    defaults = {"increase": 0.001, "decrease": 0.0006, "maximum": 0.004, "window_size": 10}
    return update_sampling_bound(bound, successes, **{**defaults, **kwargs})


def test_bound_rises_only_after_a_full_window_of_high_success() -> None:
    # A partial window must not promote, however good it looks.
    bound, rolling, moved = _bound([1.0] * 9)
    assert (bound, moved) == (0.0, False)
    assert rolling == pytest.approx(1.0)

    bound, _, moved = _bound([1.0] * 10)
    assert moved and bound == pytest.approx(0.001)


def test_bound_falls_when_success_collapses_and_never_goes_negative() -> None:
    bound, _, moved = _bound([0.0] * 10, bound=0.001)
    assert moved and bound == pytest.approx(0.0004)
    # Already at zero: there is no easier initial state to withdraw.
    bound, _, moved = _bound([0.0] * 10, bound=0.0)
    assert not moved and bound == 0.0


def test_bound_holds_inside_the_dead_band() -> None:
    """Between 10% and 80% the agent is learning; moving the range would fight it."""

    for rate in (0.2, 0.5, 0.8):
        successes = [1.0] * int(round(rate * 10)) + [0.0] * (10 - int(round(rate * 10)))
        bound, _, moved = _bound(successes, bound=0.002)
        assert not moved and bound == pytest.approx(0.002)


def test_bound_saturates_at_the_trained_maximum() -> None:
    bound, _, _ = _bound([1.0] * 10, bound=0.0035)
    assert bound == pytest.approx(0.004)
    bound, _, moved = _bound([1.0] * 10, bound=0.004)
    assert not moved and bound == pytest.approx(0.004)


def test_bound_respects_a_minimum_practice_time() -> None:
    """Promotion before the first timeouts have completed would read noise."""

    assert not _bound([1.0] * 10, steps_elapsed=1_599, minimum_steps=1_600)[2]
    assert _bound([1.0] * 10, steps_elapsed=1_600, minimum_steps=1_600)[2]


def test_the_hard_end_of_the_range_is_present_from_the_first_step() -> None:
    """The property that separates this from a stage curriculum.

    The maximum never moves; only the easy bound rises. An agent therefore
    cannot converge on a policy that only works from easy initial states,
    which is the failure recorded for this project's first grasp policy.
    """

    maximum = 0.004
    bound = 0.0
    for _ in range(20):
        bound, _, _ = _bound([1.0] * 10, bound=bound, maximum=maximum)
        assert bound <= maximum
    assert bound == pytest.approx(maximum)


def test_invalid_sampling_bounds_raise() -> None:
    with pytest.raises(ValueError):
        _bound([1.0] * 10, bound=0.005)  # above maximum
    with pytest.raises(ValueError):
        _bound([1.0] * 10, increase=-0.001)
    with pytest.raises(ValueError):
        _bound([1.0] * 10, promote_threshold=0.05, demote_threshold=0.5)
    with pytest.raises(ValueError):
        _bound([2.0] * 10)


def test_a_constant_bias_survives_averaging_and_jitter_does_not() -> None:
    """The reason the bias is the term that matters.

    Over many control steps the policy can average the jitter to nothing. The
    bias is identical every step, so averaging leaves it exactly intact: the
    only way to discover it is to touch something.
    """

    rng = np.random.default_rng(0)
    truth = np.zeros(3)
    bias = np.asarray([0.003, -0.001, 0.0])
    samples = np.stack(
        [belief_pose_error(truth, bias, rng.normal(0.0, 0.0005, size=3)) for _ in range(4_000)]
    )
    np.testing.assert_allclose(samples.mean(axis=0), bias, atol=5.0e-5)


def test_belief_is_wrong_by_exactly_the_bias_when_jitter_is_off() -> None:
    truth = np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    bias = np.asarray([0.0, 0.002, 0.0, 0.0, 0.0, 0.0])
    belief = belief_pose_error(truth, bias)
    np.testing.assert_allclose(belief - truth, bias)
    assert np.linalg.norm(belief[:3] - truth[:3]) == pytest.approx(0.002)


def test_belief_broadcasts_over_a_batch_of_environments() -> None:
    truth = np.zeros((512, 6))
    bias = np.zeros((512, 6))
    bias[:, 1] = 0.004
    belief = belief_pose_error(truth, bias, 0.0)
    assert belief.shape == (512, 6)
    np.testing.assert_allclose(belief[:, 1], 0.004)


def test_the_ablation_differs_in_exactly_one_observation_term() -> None:
    """The whole experiment rests on this.

    If the force-aware task and its force-blind control differ in anything but
    whether the actor observes contact, the measured difference cannot be
    attributed to sensing. This is the same design the earlier force-feedback
    result used, and it is the reason that result is believable.
    """

    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src/zero_g_blade_swap/tasks/blade_swap/uncertain_insertion_env_cfg.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    def declared(name: str) -> dict[str, str]:
        node = next(item for item in ast.walk(tree) if isinstance(item, ast.ClassDef) and item.name == name)
        return {
            entry.target.id: ast.unparse(entry.value)
            for entry in node.body
            if isinstance(entry, ast.AnnAssign) and isinstance(entry.target, ast.Name) and entry.value is not None
        }

    def bases(name: str) -> list[str]:
        node = next(item for item in ast.walk(tree) if isinstance(item, ast.ClassDef) and item.name == name)
        return [ast.unparse(base) for base in node.bases]

    # The blind actor is the seeing actor minus one term, by inheritance, so no
    # other term can drift between them without the edit being obvious.
    assert bases("UncertainInsertionBlindActorObsCfg") == ["UncertainInsertionActorObsCfg"]
    blind = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.ClassDef) and item.name == "UncertainInsertionBlindActorObsCfg"
    )
    overridden = [
        entry.targets[0].id
        for entry in blind.body
        if isinstance(entry, ast.Assign) and isinstance(entry.targets[0], ast.Name)
    ]
    assert overridden == ["contact_wrench"]

    # The two tasks may differ only in their observation group.
    assert bases("ZeroGBladeUncertainInsertionBlindEnvCfg") == ["ZeroGBladeUncertainInsertionEnvCfg"]
    assert set(declared("ZeroGBladeUncertainInsertionBlindEnvCfg")) == {"observations"}

    def observation_functions(name: str) -> set[str]:
        """The ``func=`` of every ObsTerm a group declares, from the AST.

        Read structurally rather than by text search, so a docstring naming a
        term cannot pass or fail this check.
        """

        node = next(item for item in ast.walk(tree) if isinstance(item, ast.ClassDef) and item.name == name)
        functions = set()
        for entry in node.body:
            value = getattr(entry, "value", None)
            if isinstance(value, ast.Call) and ast.unparse(value.func) == "ObsTerm":
                functions.update(
                    ast.unparse(keyword.value) for keyword in value.keywords if keyword.arg == "func"
                )
        return functions

    actor = observation_functions("UncertainInsertionActorObsCfg")
    critic = observation_functions("UncertainInsertionCriticObsCfg")

    # The actor must never be handed ground truth, and the critic must have it.
    assert "mdp.insertion_goal_error" not in actor
    assert "mdp.BeliefPoseErrorObservation" in actor
    assert "mdp.insertion_goal_error" in critic
    # It must not be told how wrong its estimate is either, which would remove
    # the reason it has to touch anything.
    assert "mdp.belief_bias_magnitude" not in actor
    # FORGE's conditioning: the policy is told the force budget it is judged on.
    assert "mdp.ContactForceThresholdObservation" in actor


def test_both_arms_share_one_ppo_configuration() -> None:
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    registration = (root / "src/zero_g_blade_swap/tasks/blade_swap/__init__.py").read_text(encoding="utf-8")
    assert registration.count("rl_games_uncertain_insertion.yaml") == 1
    assert registration.count("uncertain_insertion_env_cfg") == 1

    params = yaml.safe_load(
        (root / "src/zero_g_blade_swap/tasks/blade_swap/agents/rl_games_uncertain_insertion.yaml").read_text(
            encoding="utf-8"
        )
    )["params"]
    # Asymmetric actor-critic: the critic group has to actually be wired up, or
    # the privileged observations are computed and silently thrown away.
    assert params["env"]["obs_groups"] == {"obs": ["policy"], "states": ["critic"]}
    assert params["config"]["central_value_config"]["network"]["central_value"] is True
    assert params["config"]["name"] == "zero_g_blade_insertion_uncertain"
