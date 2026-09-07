from dataclasses import replace

import pytest

from assembly_recovery.geometry import PegGeometry


def geometry():
    return PegGeometry(0.007986, 0.0081, 0.05, 0.025, 0.017608, 0.05, 0.04)


def test_clearance_is_radial_and_withdrawal_clears_the_hole():
    value = geometry()
    assert value.radial_clearance_m == pytest.approx(0.000057)
    assert value.maximum_withdrawn_base_clearance_m == pytest.approx(0.017608)
    assert value.seated_fingertip_above_hole_top_m == pytest.approx(0.007392)


@pytest.mark.parametrize("changes", [
    {"peg_diameter_m": 0.009}, {"hole_height_m": 0},
    {"position_action_bound_m": 0.02}, {"peg_height_m": float("nan")},
])
def test_unphysical_or_unrecoverable_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(geometry(), **changes)
