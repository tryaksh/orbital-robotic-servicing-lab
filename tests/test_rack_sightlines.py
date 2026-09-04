'''Hold the derived rack sight lines, and what they say about the datum layout.

The strict RGB-D chain lost both fixed views late in guarded insertion.  These
tests hold the geometric reason -- the destination bay's own vertical lead-in --
so a later camera or datum change cannot quietly re-open the blind band, and so
the finding survives without the recorded run's gitignored artifacts.
'''

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_rack_sightlines import check  # noqa: E402
from check_servicing_camera_geometry import MIN_MARKER_CELL_PX  # noqa: E402


def test_the_derivation_reproduces_the_recorded_loss_depth() -> None:
    """The recorded advance stopped inside the derived occlusion onset window."""

    validation = check(datum_offsets_m=(0.0,))['self_validation']
    assert validation['agrees']
    assert (
        validation['derived_marker_first_touched_m']
        <= validation['recorded_terminal_axial_target_m']
        <= validation['derived_marker_border_row_consumed_m']
    )


def test_a_single_centred_datum_is_blind_inside_the_lead_in() -> None:
    """One datum at the module centre cannot be seen through the upper ramp."""

    result = check(datum_offsets_m=(0.0,))
    assert result['status'] == 'failed'
    assert result['blind_stroke_length_m'] > 0.10
    culprits = result['per_datum'][0]['cameras']['primary']['quiet_zone'][
        'first_occluder_at_module_centre_x_m'
    ]
    assert next(iter(culprits)) == 'destination_entry_upper_ramp'


def test_the_shipped_datum_layout_leaves_no_blind_stroke() -> None:
    """Whatever the scene ships, the seating stroke has to be covered end to end."""

    result = check()
    assert result['status'] == 'passed', result['blind_stroke_any_camera_m']
    assert result['blind_stroke_length_m'] == 0.0
    # Not merely covered by pooling the two views: the primary camera alone
    # never loses every datum, so the second camera is margin, not the answer.
    assert result['blind_stroke_primary_camera_only_m'] == []
    gates = result['unchanged_gates']
    assert gates['minimum_quiet_zone_frame_margin_px'] > 0.0
    assert gates['minimum_marker_cell_px'] >= MIN_MARKER_CELL_PX
    assert all(result['source_bindings'].values())


def test_the_datum_pair_straddles_the_derived_lead_in_shadow() -> None:
    """The separation is the shadow it has to cross, not a chosen number."""

    single = check(datum_offsets_m=(0.0,))
    shadow = single['per_datum'][0]['cameras']['primary']['quiet_zone']['occluded']
    shadow_length = shadow['to_module_centre_x_m'] - shadow['from_module_centre_x_m']
    offsets = check()['datum']['shipped_offsets_x_m']
    assert max(offsets) - min(offsets) > shadow_length
