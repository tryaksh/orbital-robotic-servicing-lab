"""The insertion has to be able to start where the chain hands it the module.

It could not. The skill reset at one module pose per bay, the certified staging
pose, 167 mm from the seated plane, and the chain hands it the module at the
mouth, 529 mm out -- so every state the chain produces was 362 mm outside the
distribution the policy trained on. The checkpoint the chain carries certifies
at 0.00%, and that is the reason.

These hold the bank that replaces it: that it spans the stroke, that it agrees
with the poses this project already solved with a simulator in the loop, and
that every station is a pose the arm can actually hold.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zero_g_blade_swap.grapple_geometry import (  # noqa: E402
    GRAPPLE_PIN_GRIP_OFFSET,
    TRANSIT_CLEAR_BLADE_CENTRE_X,
)

# Loaded by path rather than by package, because importing the task package
# pulls in gymnasium and Isaac Lab and this file only needs two tuples.
_BANK_PATH = (
    PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "insert_reset_bank.py"
)
_BANK: dict[str, object] = {}
exec(compile(_BANK_PATH.read_text(encoding="utf-8"), str(_BANK_PATH), "exec"), _BANK)  # noqa: S102
INSERT_STROKE_ARM_JOINT_POS = _BANK["INSERT_STROKE_ARM_JOINT_POS"]
INSERT_STROKE_BLADE_POSE = _BANK["INSERT_STROKE_BLADE_POSE"]

ASSETS = PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py"


def _literal(name: str, path: Path = ASSETS) -> object:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(name)


def test_the_bank_spans_the_stroke_the_chain_actually_drives() -> None:
    """Shallow end at the mouth, deep end at the pose the old reset used.

    One row, not two. The chain seats into the second bay with the robot parked
    opposite it on the rail, and parked there the arm's configuration is the one
    it has at bay 1 to 0.0000 mrad. A second row for the *same* bay reached from
    the *first* bay's base differs by 505 mrad and is a pose the chain never
    presents, so it is not a second bay -- it is a distraction.
    """

    assert len(INSERT_STROKE_ARM_JOINT_POS) == len(INSERT_STROKE_BLADE_POSE) == 1
    for row in INSERT_STROKE_BLADE_POSE:
        centres = [pose[0] for pose in row]
        assert centres == sorted(centres), centres
        assert abs(centres[0] - TRANSIT_CLEAR_BLADE_CENTRE_X) < 1.0e-3, centres[0]
        assert abs(centres[-1] - 0.5829) < 1.0e-3, centres[-1]
        # The span is what was missing, not the resolution: one pose cannot be
        # a distribution however much joint noise is added around it.
        assert centres[-1] - centres[0] > 0.40, centres


def test_every_station_is_paired_with_its_own_arm_configuration() -> None:
    for arms, blades in zip(INSERT_STROKE_ARM_JOINT_POS, INSERT_STROKE_BLADE_POSE, strict=True):
        assert len(arms) == len(blades)
        for joints in arms:
            assert len(joints) == 6


def test_the_deepest_station_reproduces_the_pose_the_simulator_solved() -> None:
    """The cross-check that says the bank describes this workcell.

    ``GRAPPLE_HEAD_ON_ARM_JOINT_POS[2]`` was solved by servoing the task's own
    differential IK in Isaac Sim; the bank's deepest bay-1 station is the same
    pose solved in closed form on the CPU. If those two disagree, one of them is
    describing a different robot.
    """

    calibrated = _literal("GRAPPLE_HEAD_ON_ARM_JOINT_POS")[2]
    solved = INSERT_STROKE_ARM_JOINT_POS[0][-1]
    # Identical even though the bay is different, because the base moved with it.
    worst = max(abs(a - b) for a, b in zip(calibrated, solved, strict=True))
    assert worst < 1.0e-4, (worst, calibrated, solved)


def test_the_only_row_is_the_bay_the_chain_seats_into() -> None:
    second = _literal("SECOND_SLOT_CENTER_Y")
    assert all(abs(pose[1] - second) < 1.0e-6 for pose in INSERT_STROKE_BLADE_POSE[0])

    # And the task parks the robot opposite it, which is what makes those joint
    # angles the same ones bay 1 uses.
    two_slot = (
        PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "two_slot_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "self.scene.robot.init_state.pos = (" in two_slot
    assert "GRAPPLE_ROBOT_ROOT_POS[1] + SECOND_SLOT_CENTER_Y," in two_slot


def test_both_bays_seat_where_the_release_interlock_permits() -> None:
    """Bay 1's insertion goal was 74 mm past where this interface may let go.

    ``SERVICE_DESTINATION_SEATED_X`` is derived from the latch's own geometry:
    an engaged jaw enters the slot mouth once the module centre passes a certain
    depth, so that is the deepest a robot may drive this module and then release
    it. The second bay has used it since the relocation was built. The first bay
    was left on ``BLADE_INSERTED_POS``, which is where a module *starts*
    installed in a task that spawned it there -- six times the insertion's own
    12 mm axial tolerance away, on a goal the chain never asks for.
    """

    from zero_g_blade_swap import service_latch
    from zero_g_blade_swap.grapple_geometry import BLADE_LENGTH_M, SLOT_MOUTH_X

    seated = round(
        service_latch.release_before_blade_centre_x_m(
            SLOT_MOUTH_X, 0.5 * BLADE_LENGTH_M, service_latch.AXIAL_SEEK_MAX_M
        )
        - 0.005
        - 0.012,
        6,
    )
    nominal = _literal("BLADE_INSERTED_POS")
    assert float(nominal[0]) - seated > 0.070, (nominal[0], seated)

    assets = ASSETS.read_text(encoding="utf-8")
    for name in ("FIRST_SLOT_INSERTED_POS", "SECOND_SLOT_INSERTED_POS"):
        block = assets.split(f"{name} = (", 1)[1].split(")", 1)[0]
        assert "SERVICE_DESTINATION_SEATED_X" in block, (name, block)

    two_slot = (
        PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "two_slot_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "goal_pos_by_stage=(FIRST_SLOT_INSERTED_POS, SECOND_SLOT_INSERTED_POS)" in two_slot


def test_the_pin_grip_offset_is_what_the_bank_places_the_module_from() -> None:
    """The bank's module centres and its arm poses have to mean the same thing."""

    # Bay 1's deepest station: the module centre is the grip offset ahead of the
    # tool the calibrated joints put there. ``INSERTION_STAGING_BLADE_POS`` is
    # the pose that pairing was originally recorded as.
    staging = _literal("INSERTION_STAGING_BLADE_POS")
    assert abs(INSERT_STROKE_BLADE_POSE[0][-1][0] - float(staging[0])) < 1.0e-3
    assert GRAPPLE_PIN_GRIP_OFFSET[1] == 0.0 and GRAPPLE_PIN_GRIP_OFFSET[2] == 0.0
