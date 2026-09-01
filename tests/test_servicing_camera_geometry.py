'''Defend the fixed overhead camera intervention without Isaac Sim.'''

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


def test_overhead_camera_covers_the_complete_sampled_flush_tag() -> None:
    result = check()
    assert result['status'] == 'passed'
    projection = result['projection']
    assert projection['minimum_quiet_zone_frame_margin_px'] > 0.0
    assert projection['minimum_marker_edge_px'] > 24.0
    assert projection['maximum_incidence_deg'] < 40.0
    assert all(result['source_bindings'].values())


def test_only_camera_placement_and_aim_changed() -> None:
    assert (CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX) == (384, 384)
    assert CAMERA_FOCAL_LENGTH_MM == 45.0
    assert CAMERA_HORIZONTAL_APERTURE_MM == 30.0
    datum = check()['flush_datum']
    assert datum['tag_size_m'] == 0.090
    assert datum['quiet_zone_size_m'] == 0.120
    assert datum['centre_in_module_m'] == [0.0, 0.0, 0.011]
