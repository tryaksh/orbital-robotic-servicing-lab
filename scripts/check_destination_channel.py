"""What channel does each entry point actually put the module in?

``check_workcell_geometry.py`` answers this from the asset module's literals and
needs no simulator. That is the right way round, and it has one blind spot: the
destination bay is not what the literals say. ``configure_service_destination``
moves the guides outboard, drops the floor and raises the lips by
``service_destination_channel_relief_m`` at *configuration* time, so the channel
a run is measured in depends on which entry point built the config and how many
times it called that method.

**This script reads the built configuration rather than the source.** One app
launch, no environment instantiated, no policy: it reproduces each entry point's
own call sequence -- the certification's ``play.py``, the lock-on diagnostic's
``play.py --latch_enabled``, ``train.py --robustness_level``, and the chain's
``run_workflow_demo.py`` -- and reports the channel each one ends up with.

Two things it is here to catch, both of which decide what a rack change has to
be before any GPU time is spent on one:

* **whether the relief is applied once.** ``configure_service_destination`` is
  called from ``configure_robustness``, which ``__post_init__`` already ran, so
  any caller that calls ``configure_robustness`` a second time adds the relief a
  second time. The guide positions are the record of how many times.
* **which axis holds a resting module.** ``2c/L`` on the lateral clearance is a
  yaw limit and on the vertical clearance a pitch limit, and with the relief
  applied the two are 20 mrad apart. Naming the wrong one points a rack change
  at the wrong constant.

Run it::

    C:/isaac-sim/python.bat scripts/check_destination_channel.py
    C:/isaac-sim/python.bat scripts/check_destination_channel.py \
        --report evidence/destination_channel_geometry.json
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, default=None, help="Write the result as evidence JSON.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import zero_g_blade_swap.tasks  # noqa: F401  -- registers the tasks
from isaaclab_tasks.utils import parse_env_cfg

from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    GRIP_MAX_TRANSVERSE_M,
    SLOT_FLOOR_TOP_Z,
    SLOT_LIP_BOTTOM_Z,
)
from zero_g_blade_swap.tasks.blade_swap.assets import (
    BLADE_SIZE,
    GUIDE_CENTER_OFFSET_Y,
    SECOND_SLOT_CENTER_Y,
    SECOND_SLOT_CFG,
    SECOND_SLOT_LEFT_GUIDE_CFG,
    SECOND_SLOT_UPPER_LEFT_LIP_CFG,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import INSERTION_ORIENTATION_TOLERANCE_RAD

#: Thickness of a side guide along y, from ``_slot_guide_cfg``. The face a module
#: runs against is half of this inboard of the body centre, and reading the
#: centre as the face is the mistake that turns a 0.75 mm channel into a 9.75 mm
#: one.
GUIDE_THICKNESS_Y_M = 0.018

SKILL_TASK = "Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0"
SKILL_PLAY_TASK = "Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-Play-v0"
CHAIN_TASK = "Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0"

#: ``--destination_channel_relief_m`` in the shipped chain preset, and the
#: default the insert skill task carries.
CHAIN_RELIEF_M = 0.0046125

#: The authored poses, before any relief. Read once from the module-level asset
#: configs, which every ``configclass`` instance deep-copies, so these stay
#: pristine however many times a config mutates its own copy.
NOMINAL_GUIDE_Y_M = float(SECOND_SLOT_LEFT_GUIDE_CFG.init_state.pos[1])
NOMINAL_FLOOR_Z_M = float(SECOND_SLOT_CFG.init_state.pos[2])
NOMINAL_LIP_Z_M = float(SECOND_SLOT_UPPER_LEFT_LIP_CFG.init_state.pos[2])


def _channel(env_cfg) -> dict[str, object]:
    """Measure the destination bay from the configuration that was built."""

    scene = env_cfg.scene
    left_y = float(scene.blade_slot_two_left_guide.init_state.pos[1])
    right_y = float(scene.blade_slot_two_right_guide.init_state.pos[1])
    floor_z = float(scene.blade_slot_two.init_state.pos[2])
    lip_z = float(scene.blade_slot_two_upper_left_lip.init_state.pos[2])

    lateral_relief = left_y - NOMINAL_GUIDE_Y_M
    floor_relief = NOMINAL_FLOOR_Z_M - floor_z
    lip_relief = lip_z - NOMINAL_LIP_Z_M

    # Clearances per side, from the surfaces rather than the body centres.
    guide_inner_face = 0.5 * abs(left_y - right_y) - 0.5 * GUIDE_THICKNESS_Y_M
    lateral = guide_inner_face - 0.5 * float(BLADE_SIZE[1])
    vertical = 0.5 * ((SLOT_LIP_BOTTOM_Z - SLOT_FLOOR_TOP_Z) - float(BLADE_SIZE[2]))
    vertical += 0.5 * (floor_relief + lip_relief)

    yaw = 2.0 * lateral / BLADE_LENGTH_M
    pitch = 2.0 * vertical / BLADE_LENGTH_M
    corner = float((lateral**2 + vertical**2) ** 0.5)
    return {
        "guide_body_centre_y_m": [round(left_y, 6), round(right_y, 6)],
        "floor_centre_z_m": round(floor_z, 6),
        "upper_lip_centre_z_m": round(lip_z, 6),
        "relief_applied_lateral_m": round(lateral_relief, 6),
        "relief_applied_floor_m": round(floor_relief, 6),
        "relief_applied_lip_m": round(lip_relief, 6),
        "relief_applications": round(lateral_relief / CHAIN_RELIEF_M, 4),
        "lateral_clearance_per_side_m": round(lateral, 6),
        "vertical_clearance_per_side_m": round(vertical, 6),
        "resting_yaw_rad": round(yaw, 6),
        "resting_pitch_rad": round(pitch, 6),
        "tightest_resting_attitude_rad": round(min(yaw, pitch), 6),
        "worst_resting_attitude_rad": round(float((yaw**2 + pitch**2) ** 0.5), 6),
        "channel_corner_m": round(corner, 6),
        "pads_can_follow_the_corner": bool(corner <= GRIP_MAX_TRANSVERSE_M + 1.0e-9),
        "yaw_inside_the_success_criterion": bool(yaw <= INSERTION_ORIENTATION_TOLERANCE_RAD),
        "pitch_inside_the_success_criterion": bool(pitch <= INSERTION_ORIENTATION_TOLERANCE_RAD),
    }


def _skill_certification():
    """``certify_grapple_skills.sh``: play.py with no level and no latch flags."""

    return parse_env_cfg(SKILL_PLAY_TASK, device="cpu", num_envs=1)


def _skill_lock_diagnostic():
    """``play.py --latch_enabled --latch_mating_compliance``.

    That path calls ``configure_robustness`` a second time to reinstall the
    latch event, and ``configure_robustness`` is what calls
    ``configure_service_destination``.
    """

    env_cfg = parse_env_cfg(SKILL_PLAY_TASK, device="cpu", num_envs=1)
    env_cfg.latch_enabled = True
    env_cfg.latch_joint_mode = "fixed"
    env_cfg.scene.replicate_physics = False
    env_cfg.scene.clone_in_fabric = False
    env_cfg.configure_robustness(int(env_cfg.robustness_level))
    return env_cfg


def _skill_training():
    """``train.py --robustness_level 0``, which every insert run has passed."""

    env_cfg = parse_env_cfg(SKILL_TASK, device="cpu", num_envs=1)
    env_cfg.configure_robustness(0)
    return env_cfg


def _chain():
    """``run_workflow_demo.py --destination_channel_relief_m``."""

    env_cfg = parse_env_cfg(CHAIN_TASK, device="cpu", num_envs=1)
    env_cfg.service_destination_channel_relief_m = CHAIN_RELIEF_M
    env_cfg.configure_service_destination()
    return env_cfg


ARMS = (
    ("chain", "run_workflow_demo.py --destination_channel_relief_m 0.0046125", _chain),
    ("skill_certification", "certify_grapple_skills.sh -> play.py, no latch flags", _skill_certification),
    ("skill_lock_diagnostic", "play.py --latch_enabled --latch_mating_compliance", _skill_lock_diagnostic),
    ("skill_training", "train.py --robustness_level 0", _skill_training),
)


def main() -> int:
    rows = []
    for name, how, build in ARMS:
        measured = _channel(build())
        measured["arm"] = name
        measured["entry_point"] = how
        rows.append(measured)

    report = {
        "title": "What channel each entry point actually builds for the destination bay",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "configure_service_destination mutates the destination bay in place and is called "
            "from configure_robustness, which __post_init__ already ran. Does every entry point "
            "put the module in the same channel, and which axis holds a resting module?"
        ),
        "law": "a module fully inside a channel with c per side wedges at 2c/L",
        "module_length_m": BLADE_LENGTH_M,
        "module_section_m": [float(BLADE_SIZE[1]), float(BLADE_SIZE[2])],
        "success_orientation_tolerance_rad": INSERTION_ORIENTATION_TOLERANCE_RAD,
        "nominal_guide_body_centre_y_m": round(GUIDE_CENTER_OFFSET_Y + SECOND_SLOT_CENTER_Y, 6),
        "arms": rows,
        "relief_applied_once_everywhere": bool(
            all(abs(float(row["relief_applications"]) - 1.0) < 1.0e-6 for row in rows)
        ),
        "every_arm_builds_the_same_channel": bool(
            len(
                {
                    (row["lateral_clearance_per_side_m"], row["vertical_clearance_per_side_m"])
                    for row in rows
                }
            )
            == 1
        ),
    }

    print()
    print(
        f"{'arm':<24}{'relief x':>9}{'lateral mm':>12}{'vertical mm':>12}"
        f"{'yaw mrad':>10}{'pitch mrad':>12}"
    )
    for row in rows:
        print(
            f"{row['arm']:<24}"
            f"{float(row['relief_applications']):>9.2f}"
            f"{float(row['lateral_clearance_per_side_m']) * 1000:>12.3f}"
            f"{float(row['vertical_clearance_per_side_m']) * 1000:>12.3f}"
            f"{float(row['resting_yaw_rad']) * 1000:>10.2f}"
            f"{float(row['resting_pitch_rad']) * 1000:>12.2f}"
        )
    print()
    print(
        f"the seated success criterion is {INSERTION_ORIENTATION_TOLERANCE_RAD * 1000:.2f} mrad; "
        f"a resting module meets it only where both columns are below that"
    )
    print(f"relief applied exactly once everywhere: {report['relief_applied_once_everywhere']}")
    print(f"every arm builds the same channel:      {report['every_arm_builds_the_same_channel']}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
