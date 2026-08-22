"""Leak-prevention contracts for vision observations, without importing Isaac Lab.

The simulator is not available in the ordinary unit-test environment, so these
tests inspect the configuration and estimator syntax directly. They defend the
boundary that matters: none of the three deployed policy groups may inherit an
exact module-state term, and the camera path itself may not touch live module
state while constructing an estimate.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "src/zero_g_blade_swap/tasks/blade_swap"
PERCEPTION = TASK / "mdp/perception.py"
VISION_CONFIG = TASK / "vision_grapple_env_cfg.py"
GRAPPLE_CONFIG = TASK / "grapple_pin_env_cfg.py"
WORKFLOW_CONFIG = TASK / "workflow_demo_env_cfg.py"
WORKFLOW_DRIVER = ROOT / "scripts/run_workflow_demo.py"
COLLECTOR = ROOT / "scripts/collect_grapple_vision.py"

FORBIDDEN_EXACT_TERMS = {
    "attached_blade_velocity",
    "extraction_remaining_observation",
    "grapple_grip_error_observation",
    "insertion_goal_error",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return ast.unparse(node)


def _observation_functions(class_node: ast.ClassDef) -> dict[str, str]:
    functions: dict[str, str] = {}
    for statement in class_node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        call = statement.value
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        if _dotted_name(call.func) != "ObsTerm":
            continue
        function = next(keyword.value for keyword in call.keywords if keyword.arg == "func")
        functions[target.id] = _dotted_name(function).removeprefix("mdp.")
    return functions


def _annotated_constructor(class_node: ast.ClassDef, field: str) -> str:
    statement = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == field
    )
    assert isinstance(statement.value, ast.Call)
    return _dotted_name(statement.value.func)


def test_all_three_workflow_groups_replace_every_exact_module_channel() -> None:
    tree = _tree(VISION_CONFIG)
    expected = {
        "PerceivedGrappleSkillObsCfg": {
            "grip_error": "PerceivedGraspError",
            "blade_velocity": "PerceivedModuleVelocity",
        },
        "PerceivedWorkflowExtractObsCfg": {
            "grip_error": "PerceivedGraspError",
            "blade_velocity": "PerceivedModuleVelocity",
            "remaining_travel": "PerceivedExtractionRemaining",
        },
        "PerceivedWorkflowInsertObsCfg": {
            "grip_error": "PerceivedGraspError",
            "blade_velocity": "PerceivedModuleVelocity",
            "blade_goal_error": "PerceivedInsertionGoalError",
        },
    }
    for class_name, required in expected.items():
        active = _observation_functions(_class(tree, class_name))
        assert active == required, class_name
        assert FORBIDDEN_EXACT_TERMS.isdisjoint(active.values()), class_name

    workflow = _class(tree, "PerceivedWorkflowObsCfg")
    assert _annotated_constructor(workflow, "grasp") == "PerceivedGrappleSkillObsCfg"
    assert _annotated_constructor(workflow, "extract") == "PerceivedWorkflowExtractObsCfg"
    assert _annotated_constructor(workflow, "insert") == "PerceivedWorkflowInsertObsCfg"


def test_replacements_preserve_the_policy_layout_instead_of_adding_features() -> None:
    """Each subclass only replaces inherited terms, preserving checkpoint widths/order."""

    tree = _tree(VISION_CONFIG)
    bases = {
        "PerceivedGrappleSkillObsCfg": "GrappleSkillObsCfg",
        "PerceivedWorkflowExtractObsCfg": "WorkflowExtractObsCfg",
        "PerceivedWorkflowInsertObsCfg": "WorkflowInsertObsCfg",
    }
    for class_name, base_name in bases.items():
        node = _class(tree, class_name)
        assert [_dotted_name(base) for base in node.bases] == [base_name]

    grapple_tree = _tree(GRAPPLE_CONFIG)
    workflow_tree = _tree(WORKFLOW_CONFIG)
    common_terms = set(_observation_functions(_class(grapple_tree, "GrappleSkillObsCfg")))
    state_layouts = {
        "PerceivedGrappleSkillObsCfg": common_terms,
        "PerceivedWorkflowExtractObsCfg": common_terms
        | set(_observation_functions(_class(workflow_tree, "WorkflowExtractObsCfg"))),
        "PerceivedWorkflowInsertObsCfg": common_terms
        | set(_observation_functions(_class(workflow_tree, "WorkflowInsertObsCfg"))),
    }
    for class_name, state_terms in state_layouts.items():
        replacements = set(_observation_functions(_class(tree, class_name)))
        assert replacements <= state_terms, f"{class_name} adds policy inputs: {replacements - state_terms}"

    source = PERCEPTION.read_text(encoding="utf-8")
    vision_source = VISION_CONFIG.read_text(encoding="utf-8")
    assert vision_source.count("blade_velocity = ObsTerm(func=mdp.PerceivedModuleVelocity, scale=0.10)") == 3
    assert "return (module_pose[:, :1] - target_x).clamp_min(0.0)" in source
    assert "self._velocity = torch.zeros((env.num_envs, 6)" in source
    # Both pose-derived errors concatenate a three-vector translation and a
    # three-vector rotation, exactly matching their state-policy predecessors.
    assert source.count("torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)") == 2


def test_deployment_inference_cannot_read_simulator_module_state() -> None:
    tree = _tree(PERCEPTION)
    estimator = _class(tree, "ModuleStateEstimator")
    deployment = ast.unparse(_method(estimator, "_estimate_deployment"))
    for forbidden in (
        *FORBIDDEN_EXACT_TERMS,
        "module_pose_label",
        "_module_nominal_pose",
        "spare_blade",
        "root_state",
        "root_pos",
        "root_quat",
    ):
        assert forbidden not in deployment
    assert "camera_rgb_with_radiation_noise" in deployment
    assert "self._head(image)" in deployment


def test_one_environment_estimator_is_shared_and_cached_by_control_step() -> None:
    tree = _tree(PERCEPTION)
    estimator = _class(tree, "ModuleStateEstimator")
    estimate = ast.unparse(_method(estimator, "estimate"))
    assert "self._cached_step == step" in estimate
    assert "return (self._pose, self._velocity)" in estimate

    shared = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "shared_module_state_estimator"
    )
    shared_source = ast.unparse(shared)
    assert "getattr(env, '_module_state_estimator', None)" in shared_source
    assert "env._module_state_estimator = estimator" in shared_source

    base = _class(tree, "_PerceivedModuleObservation")
    init_source = ast.unparse(_method(base, "__init__"))
    assert "self._estimator = shared_module_state_estimator(env)" in init_source
    for name in (
        "PerceivedGraspError",
        "PerceivedExtractionRemaining",
        "PerceivedInsertionGoalError",
        "PerceivedModuleVelocity",
    ):
        assert [_dotted_name(parent) for parent in _class(tree, name).bases] == ["_PerceivedModuleObservation"]


def test_two_bay_occupancy_is_consumed_as_a_fail_closed_planning_gate() -> None:
    perception_tree = _tree(PERCEPTION)
    deployment = ast.unparse(_method(_class(perception_tree, "ModuleStateEstimator"), "_estimate_deployment"))
    assert "forward_with_occupancy(image)" in deployment
    assert "torch.sigmoid(logits)" in deployment

    driver_source = WORKFLOW_DRIVER.read_text(encoding="utf-8")
    assert '"[PLAN] visual occupancy preflight "' in driver_source
    assert "occupancy[:, 0] >= OCCUPANCY_PLAN_THRESHOLD" in driver_source
    assert "occupancy[:, 1] < OCCUPANCY_PLAN_THRESHOLD" in driver_source
    assert "self._finish(rejected, step)" in driver_source
    assert "plan_blocked = ~self.plan_checked | ~self.plan_passed" in driver_source
    assert "capturing = (phase == CAPTURE) & ~plan_blocked" in driver_source
    assert "self.actions[~capturing & ~plan_blocked, 6] = 1.0" in driver_source
    assert "self.actions[finished & plan_blocked, 6] = 0.0" in driver_source
    assert "self.phase_started[ready & accepted] = step" in driver_source
    assert '"used_to_gate_execution": True' in driver_source


def test_exact_truth_is_separated_into_oracle_labels_and_diagnostics() -> None:
    tree = _tree(PERCEPTION)
    estimator = _class(tree, "ModuleStateEstimator")
    methods_using_truth = {
        method.name
        for method in estimator.body
        if isinstance(method, ast.FunctionDef) and "module_pose_label" in ast.unparse(method)
    }
    assert methods_using_truth == {"_estimate_oracle", "diagnostic_position_error_m"}

    resolve = ast.unparse(
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_perception_mode")
    )
    assert "oracle_blend not in (0.0, 1.0)" in resolve
    assert "blind and oracle perception controls are mutually exclusive" in resolve


def test_runtime_configuration_audit_is_mandatory() -> None:
    perception_tree = _tree(PERCEPTION)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "audit_vision_deployment_observations"
        for node in perception_tree.body
    )

    vision_tree = _tree(VISION_CONFIG)
    groups = _class(vision_tree, "PerceivedWorkflowObsCfg")
    groups_post_init = ast.unparse(_method(groups, "__post_init__"))
    assert "mdp.audit_vision_deployment_observations(self)" in groups_post_init

    workflow = _class(vision_tree, "ZeroGBladeGrappleVisionWorkflowEnvCfg")
    post_init = ast.unparse(_method(workflow, "__post_init__"))
    assert "mdp.audit_vision_deployment_observations(self.observations)" in post_init
    assert "perception_mode: str = mdp.PERCEPTION_DEPLOYMENT" in VISION_CONFIG.read_text(encoding="utf-8")


def test_collector_holds_authored_pose_and_gates_before_writing() -> None:
    """A camera frame may never be paired with a module that drifted meanwhile."""

    source = COLLECTOR.read_text(encoding="utf-8")
    assert "env_cfg.scene.spare_blade.spawn.rigid_props.kinematic_enabled = True" in source
    assert "preserved_root_pose=authored_root_pose" in source
    assert '"collection_pose_hold": np.asarray("kinematic_reasserted_each_control_step")' in source
    assert '"frame_label_sync_gate_passed": np.asarray(True)' in source
    assert "if not drift_gate_passed:" in source
    assert source.index("if not drift_gate_passed:") < source.index("np.savez_compressed(args.output")
