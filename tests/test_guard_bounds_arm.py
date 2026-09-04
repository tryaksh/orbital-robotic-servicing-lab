"""The guard-bounds A/B has to be one change, and the default has to be the shipped one.

FIDUCIAL_GUARDED_ORIENTATION_TOLERANCE_RAD is 15 mrad and its own comment derives
it from "above the certified RGB-D p95 errors". That is a bound on whether the
estimate is trustworthy, and the guarded advance uses it to decide whether the
module may enter the bay. Those are different questions, and on the first pooled
RGB-D cohort three of eight environments arrived with millimetre lateral error
and 18 to 34 mrad of attitude, held for one to two and a half thousand steps, and
never advanced against a flare that catches 73.9 mrad.

Replacing it outright would move a published path. `--fiducial_guard_bounds`
makes it an arm instead, and these assertions are what keep it an arm: the
default reproduces the shipped behaviour, the alternative changes only the
admissibility test, and the detection interlock is outside the switch in both.

Source-level, so it runs in CI with no GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_workflow_demo.py"


def _driver() -> str:
    return DRIVER.read_text(encoding="utf-8")


def test_the_default_arm_is_the_shipped_one() -> None:
    source = _driver()
    declaration = source.split("'--fiducial_guard_bounds'", 1)[1].split("parser.add_argument", 1)[0]
    assert "choices=('estimator', 'lead_in')" in declaration
    assert "default='estimator'" in declaration


def test_only_the_admissibility_test_is_inside_the_switch() -> None:
    """The detection interlock must apply on both arms, or the A/B is two changes."""

    source = _driver()
    guard = source.split("if estimator is not None and estimator.backend == \"fiducial_pnp\":", 1)[1]
    guard = guard.split("clear_to_advance", 1)[0]
    switched = guard.split("if args.fiducial_guard_bounds == 'estimator':", 1)[1]
    assert "FIDUCIAL_GUARDED_LATERAL_TOLERANCE_M" in switched
    assert "FIDUCIAL_GUARDED_ORIENTATION_TOLERANCE_RAD" in switched
    # sensor_ready is assigned after the switch closes, so it is common to both.
    assert re.search(r"^\s{12}sensor_ready = estimator\.fiducial_current_detection", switched, flags=re.MULTILINE)


def test_the_alternative_arm_falls_back_to_the_entry_flare_catch() -> None:
    """With the switch off, the tolerances are the ones the state path uses."""

    source = _driver()
    preamble = source.split("lateral_tolerance = GUARDED_INSERT_LATERAL_TOLERANCE_M", 1)
    assert len(preamble) == 2, "the guarded advance no longer defaults to the entry flare catch"
    assert "orientation_tolerance = GUARDED_INSERT_ORIENTATION_TOLERANCE_RAD" in preamble[1].split("\n", 2)[1]


def test_the_report_says_which_arm_ran() -> None:
    """A self-describing report is what makes two pooled rates comparable."""

    source = _driver()
    assert 'applied_guarded_tolerance_source = "deployed_rgbd_estimator_bounds"' in source
    assert '"entry_flare_catch_on_the_deployed_estimate"' in source
    assert 'and args.fiducial_guard_bounds == "estimator"' in source


def test_the_oracle_and_the_fiducial_backend_refuse_to_run_together() -> None:
    """A silent deadlock that cost a control run is now an argument error."""

    source = _driver()
    assert 'if args.oracle and args.perception_backend == "fiducial_pnp":' in source
    guard = source.split('if args.oracle and args.perception_backend == "fiducial_pnp":', 1)[1]
    assert "parser.error(" in guard.split("if args.rack_retention", 1)[0]
    assert "--perception_backend pose_head --oracle" in guard.split("if args.rack_retention", 1)[0]


def test_the_module_velocity_source_defaults_to_the_shipped_camera_path() -> None:
    """Every published RGB-D number was measured differencing camera poses."""

    source = _driver()
    declaration = source.split("'--module_velocity_source'", 1)[1].split("parser.add_argument", 1)[0]
    assert "choices=('camera', 'kinematics')" in declaration
    assert "default='camera'" in declaration


def test_the_kinematic_velocity_never_reads_the_module() -> None:
    """It reads the robot, which is encoder information a real servicer has.

    Reading the module's own body velocity here would be a privileged channel
    wearing a deployment label, which is the failure the vision-deployment audit
    exists to stop.
    """

    estimator = (ROOT / "src/zero_g_blade_swap/tasks/blade_swap/mdp/perception.py").read_text(encoding="utf-8")
    body = estimator.split("def _kinematic_velocity", 1)[1].split("def _wrist_body_id", 1)[0]
    assert 'self._env.scene["robot"]' in body
    for forbidden in ("spare_blade", "attached_blade", "module_pose_label"):
        assert forbidden not in body, f"the kinematic velocity reads {forbidden}"
    # Zero before capture, the wrist's velocity after it, and nothing in between.
    assert "self._module_tool_attached" in body
    assert "torch.zeros_like" in body
