"""Prove the Torch half of the arm kinematics against the NumPy half.

The NumPy half is already proved against the simulator: every configuration in
``evidence/workcell_reach_solution.json`` is run through it by
``scripts/check_workcell_geometry.py``, which refuses to report if the tool pose
it computes disagrees with the one the simulator recorded. So agreement here is
transitive, and it is the reason the driver is allowed to command joint targets
from a solver that never launches a simulator.

The two halves have to agree on three things, in this order, because each one
makes the next meaningful: the forward kinematics, the Jacobian, and the solve.

Skipped where Torch is not installed, which is the CI environment for the
geometry checks. Run it under ``C:/isaac-sim/python.bat -m pytest``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zero_g_blade_swap.arm_kinematics import (  # noqa: E402
    JOINT_ORDER,
    batched_solve_ik,
    batched_tool_jacobian,
    batched_tool_pose,
    quaternion_to_matrix,
    tool_jacobian,
    tool_pose,
)

REACH_SOLUTION = PROJECT_ROOT / "evidence" / "workcell_reach_solution.json"
HEAD_ON = quaternion_to_matrix(0.0, 0.7071068, 0.0, 0.7071068)


def _recorded_configurations() -> np.ndarray:
    solution = json.loads(REACH_SOLUTION.read_text(encoding="utf-8"))
    return np.array(
        [pose["arm_joint_pos_rad"] for pose in solution["solution"]["poses"]], dtype=float
    )


def _spread(rows: np.ndarray, count: int = 24) -> np.ndarray:
    """Recorded configurations plus perturbations, so agreement is not local."""

    generator = np.random.default_rng(4070)
    extra = rows[generator.integers(0, len(rows), size=count)] + generator.normal(
        0.0, 0.35, size=(count, 6)
    )
    return np.concatenate((rows, extra), axis=0)


def test_joint_order_matches_the_task_definition() -> None:
    """The DH order is the order the environment names its arm joints in.

    Read as a literal rather than imported, because importing the task module
    needs a running Kit. The driver still checks the resolved order against the
    simulator at run time -- ``find_joints`` does not promise to preserve the
    order it is given -- but a rename here should break loudly on the CPU.
    """

    source = (
        PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "env_cfg.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: tuple[str, ...] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ARM_JOINTS" for target in node.targets
        ):
            names = tuple(ast.literal_eval(node.value))
    assert names == JOINT_ORDER, names


def test_forward_kinematics_agree() -> None:
    joints = _spread(_recorded_configurations())
    positions, rotations = batched_tool_pose(torch.tensor(joints, dtype=torch.float64))
    worst_position = 0.0
    worst_rotation = 0.0
    for index, row in enumerate(joints):
        reference_position, reference_rotation = tool_pose(row)
        worst_position = max(
            worst_position, float(np.abs(positions[index].numpy() - reference_position).max())
        )
        worst_rotation = max(
            worst_rotation, float(np.abs(rotations[index].numpy() - reference_rotation).max())
        )
    assert worst_position < 1.0e-12, worst_position
    assert worst_rotation < 1.0e-12, worst_rotation


def test_jacobians_agree() -> None:
    joints = _spread(_recorded_configurations())
    jacobians = batched_tool_jacobian(torch.tensor(joints, dtype=torch.float64))
    worst = 0.0
    for index, row in enumerate(joints):
        worst = max(worst, float(np.abs(jacobians[index].numpy() - tool_jacobian(row)).max()))
    assert worst < 1.0e-12, worst


def test_batched_solve_reaches_the_recorded_poses() -> None:
    """Seeded from a nearby configuration, the batched solve closes the pose.

    Seeded, not searched: this is the controller's own operating condition. It
    starts every solve from the joint positions the arm is in and chases a
    setpoint that moves by at most one action scale per control step, so the
    only thing that has to be true is that it converges locally.
    """

    recorded = _recorded_configurations()
    generator = np.random.default_rng(6070)
    targets = []
    for row in recorded:
        position, rotation = tool_pose(row)
        targets.append((position, rotation))
    seeds = recorded + generator.normal(0.0, 0.05, size=recorded.shape)

    joints, position_residual, attitude_residual = batched_solve_ik(
        torch.tensor(np.array([target[0] for target in targets]), dtype=torch.float64),
        torch.tensor(np.array([target[1] for target in targets]), dtype=torch.float64),
        torch.tensor(seeds, dtype=torch.float64),
    )
    assert float(position_residual.max()) < 1.0e-6, float(position_residual.max())
    assert float(attitude_residual.max()) < 1.0e-6, float(attitude_residual.max())
    # And the residual the solver reports is the residual its own forward
    # kinematics has, rather than an internal number about a different pose.
    positions, _rotations = batched_tool_pose(joints)
    for index, target in enumerate(targets):
        assert float(np.abs(positions[index].numpy() - target[0]).max()) < 1.0e-6


def test_solve_declares_a_pose_it_cannot_reach() -> None:
    """An unreachable target comes back with a residual, not with a lie."""

    seed = torch.tensor(_recorded_configurations()[:1], dtype=torch.float64)
    unreachable = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    _joints, position_residual, _attitude = batched_solve_ik(
        unreachable, torch.tensor(HEAD_ON, dtype=torch.float64).unsqueeze(0), seed
    )
    assert float(position_residual.max()) > 0.5
