"""Execute the production extraction pause and command publication without Isaac.

A camera frame can be absent every other control tick in an eight-environment
run. A pause must retain the last accepted joint-drive target, including load
compensation. Zero Cartesian actions with the override disabled are a different
controller and reproduce the oscillation that broke the first paired cohort.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from zero_g_blade_swap.motion_profile import MotionLimits

torch = pytest.importorskip("torch")
DRIVER = Path(os.environ.get(
    "WORKFLOW_DRIVER_SOURCE", str(Path(__file__).resolve().parents[1] / "scripts/run_workflow_demo.py"),
))


def production_methods():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"), filename=str(DRIVER))
    driver = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WorkflowDriver")
    names = {"_finish_extraction_smoothly", "_apply_joint_overrides"}
    methods = [node for node in driver.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {method.name for method in methods} == names
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods,
    ], type_ignores=[])
    namespace = {
        "torch": torch, "MotionLimits": MotionLimits, "EXTRACT": 2, "TRANSIT": 3, "INSERT": 4,
        "EXTRACTED_BLADE_CENTRE_X": 0.225, "args": SimpleNamespace(guarded_insert_solver="absolute_ik"),
    }
    exec(compile(ast.fix_missing_locations(module), str(DRIVER), "exec"), namespace)  # noqa: S102
    return namespace


class PublishedArm:
    def __init__(self, accepted, encoders):
        self.hold = torch.tensor([True, False, True])
        self.targets = accepted.clone()
        self.encoders = encoders

    def set_joint_hold_mask(self, mask):
        newly_enabled = mask & ~self.hold
        self.targets[newly_enabled] = self.encoders[newly_enabled]
        self.hold = mask.clone()

    def set_joint_target_override(self, ids, targets):
        self.targets[ids] = targets
        self.hold[ids] = True


def make_driver():
    accepted = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10.0
    bias = torch.full((3, 6), 0.05)
    encoders = accepted - bias
    estimator = SimpleNamespace(backend="fiducial_pnp", fiducial_current_detection=torch.ones(3, dtype=torch.bool))
    driver = SimpleNamespace(
        task=SimpleNamespace(step_dt=1 / 30, _module_state_estimator=estimator),
        phase=torch.tensor([2, 0, 2]),
        arm=PublishedArm(accepted, encoders), base_rail_enabled=False, rail_indexing=torch.zeros(3, dtype=torch.bool),
        actions=torch.ones((3, 7)), solved_joint_targets=accepted.clone(),
        solved_joint_hold=torch.tensor([True, False, True]), profile_joint_bias=bias.clone(),
        profile_elapsed=torch.tensor([2.0, 0.4, 3.0]), profile_trim_steps=torch.tensor([20, 8, 25]),
        extract_finish_active=torch.tensor([True, False, True]),
        extract_finish_started=torch.tensor([256, -1, 304]),
        extract_finish_target_pos=torch.tensor([[0.1, 0.2, 0.3]]).repeat(3, 1),
        extract_finish_target_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        extract_finish_steps=torch.tensor([10, 0, 15]), extract_finish_age=torch.tensor([10, 0, 15]),
        extract_finish_holds=torch.tensor([1, 0, 3]), scales=torch.full((6, 6), 0.008), command_calls=[],
    )
    position = torch.tensor([[0.222, 0.01, 0.73]]).repeat(3, 1)
    driver._payload_feedback = lambda: (position, None, torch.zeros((3, 6)))

    def unexpected_seed(*_args):
        raise AssertionError("An established terminal path must not be reseeded on a missing frame")

    driver._check_forward_kinematics = unexpected_seed
    driver._seed_solved_setpoints = unexpected_seed

    def command_solved(self, ids, *_args, **_kwargs):
        # Instrumented downstream solver: any erroneous call visibly advances
        # all protected quantities, making wrong masks observable in the test.
        self.command_calls.append(ids.clone())
        self.solved_joint_targets[ids] += 0.0001
        self.profile_joint_bias[ids] += 0.0001
        self.profile_elapsed[ids] += self.task.step_dt
        self.profile_trim_steps[ids] += 1
        self.solved_joint_hold[ids] = True

    driver._command_solved_tool_pose = MethodType(command_solved, driver)
    for name, method in production_methods().items():
        if name in {"_finish_extraction_smoothly", "_apply_joint_overrides"}:
            setattr(driver, name, MethodType(method, driver))
    return driver


def execute_tick(driver, detected, established):
    driver.task._module_state_estimator.fiducial_current_detection = torch.tensor(detected)
    driver.actions[:] = 1.0  # The outer driver has computed this tick's PPO actions.
    extracting = torch.tensor([True, False, True])
    ready = torch.tensor(detected) & torch.tensor(established)
    blocked, moving = extracting & ~ready, extracting & ready
    protected = {
        name: getattr(driver, name).clone() for name in (
            "solved_joint_targets", "profile_joint_bias", "profile_elapsed", "profile_trim_steps",
            "extract_finish_age", "extract_finish_steps", "extract_finish_started",
        )
    }
    holds = driver.extract_finish_holds.clone()
    calls = len(driver.command_calls)
    tool = torch.zeros((3, 3))
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    driver._finish_extraction_smoothly(extracting, torch.tensor(established), 400, tool, rotation)
    driver._apply_joint_overrides()
    assert bool(driver.arm.hold[blocked].all()), "A pause must keep the accepted joint override, not fall back to DLS"
    assert torch.equal(driver.arm.targets[blocked], protected["solved_joint_targets"][blocked])
    for name, before in protected.items():
        assert torch.equal(getattr(driver, name)[blocked], before[blocked]), name
        assert torch.equal(getattr(driver, name)[1], before[1]), f"Non-extracting environment: {name}"
    assert torch.equal(driver.extract_finish_holds, holds + blocked.to(torch.long))
    assert torch.equal(driver.extract_finish_steps, protected["extract_finish_steps"] + moving.to(torch.long))
    assert torch.equal(driver.actions[extracting, :6], torch.zeros((2, 6)))
    assert torch.equal(driver.actions[:, 6], torch.ones(3)), "Sensor holds must not open the physical grip"
    assert torch.equal(driver.actions[1], torch.ones(7)), "Other phase actions must remain untouched"
    assert len(driver.command_calls) == calls + int(bool(moving.any()))
    if bool(moving.any()):
        assert torch.equal(driver.command_calls[-1], torch.nonzero(moving, as_tuple=False).squeeze(-1))


def test_alternating_camera_frames_hold_targets_and_freeze_only_blocked_environments():
    driver = make_driver()
    # Two active extraction environments and an unrelated third command. Each
    # active environment alternates fresh/missing frames with a different phase.
    for detected in ([True, True, False], [False, True, True]) * 4:
        execute_tick(driver, detected, [True, True, True])
    assert torch.equal(driver.extract_finish_steps, torch.tensor([14, 0, 19]))
    assert torch.equal(driver.extract_finish_holds, torch.tensor([5, 0, 7]))


def test_all_missing_frames_never_call_ik_or_advance_trim_or_motion():
    driver = make_driver()
    for _ in range(5):
        execute_tick(driver, [False, True, False], [True, True, True])
    assert driver.command_calls == []


def test_lost_grip_uses_the_same_hold_without_releasing_command_bias():
    driver = make_driver()
    execute_tick(driver, [True, True, True], [False, True, True])
    execute_tick(driver, [True, True, True], [False, True, False])
    # Recovery resumes the same accepted path; no reseeding or policy switch.
    execute_tick(driver, [True, True, True], [True, True, True])
