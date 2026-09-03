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
