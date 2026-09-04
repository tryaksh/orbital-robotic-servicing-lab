from scripts.project_insertion_checkpoint import (
    DROP_TERMS,
    SOURCE_LAYOUT,
    TARGET_LAYOUT,
    layout_width,
    retained_feature_indices,
)


def test_projection_drops_only_absolute_robot_posture() -> None:
    assert {"joint_pos", "end_effector"} == DROP_TERMS
    assert layout_width(SOURCE_LAYOUT) == 45
    assert layout_width(TARGET_LAYOUT) == 32
    assert tuple(name for name, _ in TARGET_LAYOUT) == (
        "joint_vel",
        "grip_error",
        "gripper_state",
        "blade_velocity",
        "previous_action",
        "blade_goal_error",
    )


def test_projection_indices_match_the_runtime_observation_order() -> None:
    assert retained_feature_indices() == (*range(6, 12), *range(19, 45))
