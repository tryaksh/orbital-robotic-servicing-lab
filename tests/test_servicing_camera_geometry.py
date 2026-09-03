'''Defend the fixed front-side camera intervention without Isaac Sim.'''

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_servicing_camera_geometry import check  # noqa: E402
from zero_g_blade_swap.servicing_camera import (  # noqa: E402
    CAMERA_FOCAL_LENGTH_MM,
    CAMERA_HEIGHT_PX,
    CAMERA_HORIZONTAL_APERTURE_MM,
    CAMERA_WIDTH_PX,
)


def test_front_side_camera_covers_the_complete_sampled_flush_tag() -> None:
    result = check()
    assert result['status'] == 'passed'
    projection = result['projection']
    assert projection['minimum_quiet_zone_frame_margin_px'] > 0.0
    assert projection['minimum_marker_cell_px'] >= projection['minimum_marker_cell_requirement_px']
    assert projection['maximum_incidence_deg'] < 40.0
    assert projection['optical_axis_target_alignment'] > 1.0 - 1.0e-9
    assert projection['minimum_rear_gripper_sightline_clearance_m'] > 0.0
    assert all(result['source_bindings'].values())


def test_resolution_change_preserves_optics_and_flush_datum() -> None:
    assert (CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX) == (640, 640)
    assert CAMERA_FOCAL_LENGTH_MM == 45.0
    assert CAMERA_HORIZONTAL_APERTURE_MM == 30.0
    datum = check()['flush_datum']
    assert datum['tag_size_m'] == 0.090
    assert datum['quiet_zone_size_m'] == 0.120
    # Flush on the top face, both plates on the same plane. Where along the
    # module they sit is the datum-layout arm, not the optics.
    assert datum['centre_in_module_m'][2] == 0.011
    assert datum['offsets_x_m'] == datum['shipped_offsets_x_m']
    assert datum['evaluates_the_shipped_layout']


def test_one_usable_plate_covers_every_sampled_pose() -> None:
    """The gate is on the datum set, because the estimator reads either plate."""

    coverage = check()['coverage']
    assert coverage['poses_with_at_least_one_usable_datum'] == coverage['sampled_poses']
    per_datum = coverage['poses_covered_per_datum']
    assert len(per_datum) == 2
    # Neither plate covers the envelope on its own; that is why there are two.
    assert all(0 < count < coverage['sampled_poses'] for count in per_datum.values())


def test_the_superseded_single_centred_datum_is_still_replayable() -> None:
    """The losing arm stays evaluable, so the change can be attributed."""

    result = check(datum_offsets_m=(0.0,))
    assert result['flush_datum']['offsets_x_m'] == [0.0]
    assert not result['flush_datum']['evaluates_the_shipped_layout']
    assert result['coverage']['poses_with_at_least_one_usable_datum'] == 64
