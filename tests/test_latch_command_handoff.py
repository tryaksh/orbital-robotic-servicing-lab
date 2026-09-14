"""Exercise the actual handoff methods without launching the simulator.

The regression was a one-step controller switch before physical latch
engagement. A held joint command fell back to relative IK, then returned to its
original value. These tests execute the production methods, rather than a copy
of their mathematics, and keep commands distinct from measured joint positions.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

DRIVER = Path(os.environ.get(
    "WORKFLOW_DRIVER_SOURCE",
    str(Path(__file__).resolve().parents[1] / "scripts" / "run_workflow_demo.py"),
))


def _methods(*, profile="quintic", extraction="guarded"):
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"), filename=str(DRIVER))
    driver = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WorkflowDriver")
    names = {"_hold_pending_latch", "_step_rigid_transit"}
    functions = [node for node in driver.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names, "The production driver must provide both handoff methods"
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    namespace = {
        "torch": torch,
        "grapple_latched": lambda task: task._grapple_latched,
        "args": SimpleNamespace(transit_motion_profile=profile, extraction_finish=extraction),
    }
    exec(compile(ast.fix_missing_locations(module), str(DRIVER), "exec"), namespace)  # noqa: S102
    return namespace


def _driver(*, profile="quintic", extraction="guarded"):
    count = 4
    target = torch.arange(count * 8, dtype=torch.float32).reshape(count, 8) / 10
    robot_data = SimpleNamespace(joint_pos_target=target.clone(), joint_pos=target + 0.035)
    payload_data = SimpleNamespace(
        root_pos_w=torch.tensor([[0.219, 0.035, 0.729]]).repeat(count, 1),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
    )
    task = SimpleNamespace(
        num_envs=count,
        scene={"robot": SimpleNamespace(data=robot_data), "spare_blade": SimpleNamespace(data=payload_data)},
        _grapple_latched=torch.tensor([False, False, True, False]),
    )
    driver = SimpleNamespace(
        task=task,
        # Nontrivial ordering catches accidentally taking the first six robot
        # joints, including a gripper joint, instead of the accepted arm target.
        arm_joint_ids=[7, 0, 5, 2, 6, 3],
        solved_joint_hold=torch.tensor([True, False, True, False]),
        solved_joint_targets=torch.arange(count * 6, dtype=torch.float32).reshape(count, 6) / 20 - 1.0,
        actions=torch.arange(count * 7, dtype=torch.float32).reshape(count, 7) / 30,
        latch_wait_steps=torch.tensor([3, 0, 5, 7]),
        profile_elapsed=torch.tensor([-1.0, -1.0, 0.6, 0.9]),
        profile_duration=torch.tensor([2.1, 3.4, 1.6, 1.9]),
        profile_joint_bias=torch.full((count, 6), 0.012),
        waypoint_read=torch.tensor([4, 4, 2, 1]),
        transit_reference_pos_tool=torch.tensor([[0.352, 0.006, -0.003]]).repeat(count, 1),
        transit_reference_rot_tool=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        transit_reference_valid=torch.ones(count, dtype=torch.bool),
    )
    for name, value in _methods(profile=profile, extraction=extraction).items():
        if name in {"_hold_pending_latch", "_step_rigid_transit"}:
            setattr(driver, name, MethodType(value, driver))
    return driver


def _protected_state(driver):
    names = (
        "profile_elapsed", "profile_duration", "profile_joint_bias", "waypoint_read",
        "transit_reference_pos_tool", "transit_reference_rot_tool", "transit_reference_valid",
    )
    snapshot = {name: getattr(driver, name).clone() for name in names}
    for entity, attributes in (("robot", ("joint_pos", "joint_pos_target")), ("spare_blade", ("root_pos_w", "root_quat_w"))):
        for name in attributes:
            snapshot[f"{entity}.{name}"] = getattr(driver.task.scene[entity].data, name).clone()
    return snapshot


def test_pending_profiles_keep_accepted_targets_and_policy_entry_uses_actuator_targets():
    driver = _driver()
    held_target = driver.solved_joint_targets[0].clone()
    policy_target = driver.task.scene["robot"].data.joint_pos_target[1, driver.arm_joint_ids].clone()
    # These differ intentionally: copying encoder positions would unload the
    # wrist and reproduce the observed discontinuity under gripper preload.
    assert not torch.equal(held_target, driver.task.scene["robot"].data.joint_pos[0, driver.arm_joint_ids])
    ready = driver._hold_pending_latch(torch.tensor([True, True, True, False]))
    assert torch.equal(driver.solved_joint_targets[0], held_target)
    assert torch.equal(driver.solved_joint_targets[1], policy_target)
    assert torch.equal(ready, torch.tensor([False, False, True, False]))
    assert torch.equal(driver.solved_joint_hold, torch.tensor([True, True, True, False]))


def test_mixed_environments_do_not_change_ready_or_nontransiting_commands():
    driver = _driver()
    targets = driver.solved_joint_targets.clone()
    actions = driver.actions.clone()
    protected = _protected_state(driver)
    driver._hold_pending_latch(torch.tensor([True, True, True, False]))
    assert torch.equal(driver.solved_joint_targets[2:], targets[2:])
    assert torch.equal(driver.actions[2:], actions[2:])
    assert torch.equal(driver.actions[:2, :6], torch.zeros((2, 6)))
    assert torch.equal(driver.actions[:, 6], actions[:, 6]), "Waiting must keep the physical grasp command"
    assert torch.equal(driver.latch_wait_steps, torch.tensor([4, 1, 5, 7]))
    for name, value in _protected_state(driver).items():
        assert torch.equal(value, protected[name]), name


@pytest.mark.parametrize("profile,extraction", [("quintic", "legacy"), ("legacy", "guarded"), ("quintic", "guarded")])
def test_missing_latch_never_advances_motion_clock_or_reanchors_payload(profile, extraction):
    driver = _driver(profile=profile, extraction=extraction)
    driver.task._grapple_latched[:] = False
    mask = torch.tensor([True, True, False, False])
    protected = _protected_state(driver)
    expected = driver.solved_joint_targets.clone()
    expected[1] = driver.task.scene["robot"].data.joint_pos_target[1, driver.arm_joint_ids]
    tool = torch.zeros((4, 3))
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
    for step in range(294, 299):
        arrived = driver._step_rigid_transit(mask, step, tool, rotation)
        assert not bool(arrived.any()), "No latch means no successful transfer handoff"
        assert torch.equal(driver.solved_joint_targets, expected)
        for name, value in _protected_state(driver).items():
            assert torch.equal(value, protected[name]), name
    assert torch.equal(driver.latch_wait_steps, torch.tensor([8, 5, 5, 7]))


def test_engagement_releases_only_the_waiting_environment_without_resnapshotting():
    driver = _driver()
    mask = torch.tensor([True, True, False, False])
    driver._hold_pending_latch(mask)
    accepted = driver.solved_joint_targets.clone()
    driver.task._grapple_latched[0] = True
    # Later live targets differ, but the still-pending environment must retain
    # the command accepted on entry, not follow an unrelated cache change.
    driver.task.scene["robot"].data.joint_pos_target += 0.04
    ready = driver._hold_pending_latch(mask)
    assert torch.equal(ready, torch.tensor([True, False, False, False]))
    assert torch.equal(driver.solved_joint_targets, accepted)
    assert torch.equal(driver.latch_wait_steps, torch.tensor([4, 2, 5, 7]))
