"""Contracts for the chained-insert task, asserted without launching Isaac Sim.

The task exists to fine-tune insert v6 on the states a real capture hands it, so
the properties that make that possible are the ones worth pinning: the policy
sees exactly the observation the chain already feeds insert v6, it commands
exactly six actions, its phase budgets are *read* from the two skills rather than
restated, and it carries no termination the chain does not have.

Each assertion below corresponds to a defect that was either measured in this
project or would have invalidated the run:

* an observation or action dimension change makes the checkpoint unresumable;
* a copied phase budget drifts from the certification it claims to match, which
  has already cost this workflow a full certification;
* a capture-failure termination made the task harsher than the chain and cost
  the pre-training gate 26 points.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
MODULE = SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "chained_insert_env_cfg.py"
WORKFLOW = SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "workflow_demo_env_cfg.py"
DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "run_workflow_demo.py"


def _source() -> str:
    return MODULE.read_text(encoding="utf-8")


def test_policy_group_is_the_chain_s_insert_group() -> None:
    """The learning policy must see what the chain feeds insert v6, exactly.

    Anything else is an observation dimension change, and this repository never
    resumes across one.
    """

    source = _source()
    assert "policy: WorkflowInsertObsCfg = WorkflowInsertObsCfg()" in source
    assert "grasp: WorkflowGraspObsCfg = WorkflowGraspObsCfg()" in source


def test_learning_policy_commands_six_actions() -> None:
    """Six, not the action manager's seven: the gripper is the phase machine's."""

    source = _source()
    assert "shape=(6,)" in source
    assert "action[:, :6]" in source


def test_phase_budgets_are_read_not_restated() -> None:
    """Both clocks come from the tasks the two policies are certified on."""

    source = _source()
    for name in ("CAPTURE_BUDGET_S", "INSERT_BUDGET_S", "SEAT_STEPS", "HANDOVER_HOLD_S"):
        assert name in source, name
        # Imported from the shared module, never assigned a literal here.
        assert f"{name} =" not in source, f"{name} is restated instead of read"

    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "def certified_episode_length_s" in workflow
    assert "CAPTURE_BUDGET_S = certified_episode_length_s(ZeroGBladeGrapplePinGraspEnvCfg)" in workflow
    assert "INSERT_BUDGET_S = certified_episode_length_s(ZeroGBladeGrapplePinInsertEnvCfg)" in workflow


def test_the_driver_reads_the_same_constants() -> None:
    """One definition of the seat length and the hand-off hold, not two.

    Two copies of a phase budget is the drift the budget mechanism exists to
    prevent, and the chained task is a second consumer of both numbers.
    """

    driver = DRIVER.read_text(encoding="utf-8")
    assert "SEAT_STEPS = 30" not in driver
    assert "HANDOVER_HOLD_S" in driver
    assert "int(round(0.30 / float(task.step_dt)))" not in driver


def test_no_capture_failure_termination() -> None:
    """Measured: the chain does not have one, and adding it cost 26 points.

    Insert v6 unchanged scored 69.27% on this task with a capture-failure
    termination and 95.31% without it, on the same seed, because the chain
    overruns its capture phase once in 192 while this predicate was killing 52.
    """

    tree = ast.parse(_source())
    terminations = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ChainedInsertTerminationsCfg"
    )
    names = {
        target.id
        for statement in terminations.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    names |= {
        statement.target.id
        for statement in terminations.body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    }
    assert names == {"time_out", "insertion_success", "extraction_failed", "non_finite"}, names


def test_terminations_are_phase_gated() -> None:
    """The prologue is not the policy's, so its predicates must not fire there."""

    source = _source()
    assert "grapple_insertion_success_mask(env) & (env.chain_phase == INSERT)" in source
    assert "extraction_failure(env) & (env.chain_phase == INSERT)" in source


def test_prologue_carries_no_reward() -> None:
    """The learning policy did not act in the prologue, so it is not paid for it."""

    assert "torch.where(prologue, torch.zeros_like(reward), reward)" in _source()


def test_the_attitude_arm_differs_by_exactly_one_number() -> None:
    """The two chained runs must be one change apart or neither is attributable.

    ``ChainedInsertAttitudeRewardsCfg`` may move ``orientation_weight`` and
    nothing else. In particular it must not quietly adopt extraction's
    ``orientation_scale`` or its raised clamp: those were sized for a failure
    that runs to 0.35 rad, and this one lives at 0.19.
    """

    tree = ast.parse(_source())
    rewards = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ChainedInsertAttitudeRewardsCfg"
    )
    call = next(
        node
        for node in ast.walk(rewards)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RewTerm"
    )
    params = next(kw.value for kw in call.keywords if kw.arg == "params")
    values = {
        key.value: value.value
        for key, value in zip(params.keys, params.values, strict=True)
        if isinstance(value, ast.Constant)
    }
    # The insert skill's own defaults, restated here only so the diff is visible.
    assert values == {
        "free_m": 0.004,
        "free_rad": 0.08,
        "orientation_scale": 0.15,
        "orientation_weight": 1.0,
        "max_penalty": 25.0,
    }, values
    # A negative literal is a UnaryOp, not a Constant.
    weight = ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == "weight"))
    assert weight == -0.50

    # And the clamp has to stay unreachable, or this becomes extraction's
    # saturation defect rather than a weight change. Raw cost at the 0.35 rad
    # extraction-failure limit, with a typical 11.7 mm grip position:
    position_excess = (0.0117 - values["free_m"]) / 0.010
    orientation_excess = (0.35 - values["free_rad"]) / values["orientation_scale"]
    raw = position_excess**2 + values["orientation_weight"] * orientation_excess**2
    assert raw < values["max_penalty"], raw


def test_the_single_slot_insert_task_is_untouched() -> None:
    """The chained task is a separate registration, as every change here is.

    Insert v6's 95.57% describes ``ZeroGBladeGrapplePinInsertEnvCfg``, and it has
    to keep describing it.
    """

    grapple = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "grapple_pin_env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "class ZeroGBladeGrapplePinInsertEnvCfg" in grapple
    # The dependency runs one way only. The chained task reads the skill's
    # configuration; the skill must never learn about the chained task, or a
    # change made for the chain could move the ground under insert v6's
    # certification.
    assert "chained_insert_env_cfg" not in grapple
    assert "InsertChain" not in grapple

    registry = (SRC / "zero_g_blade_swap" / "tasks" / "blade_swap" / "__init__.py").read_text(encoding="utf-8")
    assert "Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0" in registry
    assert "chained_insert_env_cfg:ChainedInsertEnv" in registry
