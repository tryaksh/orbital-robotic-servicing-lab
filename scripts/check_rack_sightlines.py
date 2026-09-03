'''Which rack part stands in the camera's line to the flush datum, and where.

CPU-only, no simulator.  ``check_servicing_camera_geometry.py`` proves the datum
projects inside the frame at a resolvable size over the *transit* envelope.  It
never covered the seating stroke, and the seating stroke is where the strict
RGB-D chain lost both views: that script sweeps module centres from
``TRANSIT_CLEAR_BLADE_CENTRE_X - 0.08`` to 0.73 and asks only whether the datum
is *in frame*, never whether anything is in front of it.

This script asks the other question.  Every rack-side part is an oriented box;
the datum is its marker outline and its quiet zone; visibility is exact
segment-versus-box intersection from each configured camera.  The answer is an
interval of module-centre depth, per camera, per datum, and it is the
requirement a datum layout has to satisfy rather than a number to tune.

**The result the shipped configuration gets is that a single centred flush datum
cannot survive the destination bay's own vertical lead-in.**  The upper entry
ramp is an 80 x 60 x 18 mm plate at 12 degrees, centred on the bay centre line,
hanging about 25 mm above the module's top face.  It is a roof: a camera that
clears it has to look under an 82 mm span through 25 mm of headroom, which is
26 degrees off vertical at the near edge and worse across the plate, and a
marker cell resolved at that incidence is below what the estimator needs.  Both
configured cameras sit within ten degrees of vertical, which is why moving the
second one 370 mm along x moved the loss depth by 6.5 mm.

The script therefore also sizes the fix it implies: given a set of datum offsets
along the module, does any depth of the seating stroke leave no datum readable
from any camera?  ``--datum_offsets_m`` evaluates a candidate layout before the
scene is changed.

It validates itself against the recorded strict run before it reports.  That run
stopped advancing with its axial target at a depth that has to lie between the
depth at which the ramp first touches the marker outline and the depth at which
it has eaten the marker's outer border row, because a marker missing a whole
border cell is not decodable and one missing none of it is.

Run it::

    python scripts/check_rack_sightlines.py
    python scripts/check_rack_sightlines.py --report evidence/<new-name>.json
'''

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_servicing_camera_geometry import MIN_MARKER_CELL_PX
from zero_g_blade_swap import service_latch
from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    BLADE_THICKNESS_M,
    BLADE_WIDTH_M,
    SLOT_ENTRY_RAMP_DEG,
    SLOT_ENTRY_RAMP_LENGTH_M,
    SLOT_ENTRY_RAMP_THICKNESS_M,
    SLOT_ENTRY_RAMP_WIDTH_M,
    SLOT_FLOOR_TOP_Z,
    SLOT_LIP_BOTTOM_Z,
    SLOT_MOUTH_X,
    TRANSIT_CLEAR_BLADE_CENTRE_X,
)
from zero_g_blade_swap.servicing_camera import (
    CAMERA_FOCAL_LENGTH_MM,
    CAMERA_HEIGHT_PX,
    CAMERA_HORIZONTAL_APERTURE_MM,
    CAMERA_POSITION_M,
    CAMERA_QUATERNION_WXYZ_ROS,
    CAMERA_WIDTH_PX,
    INSERT_CAMERA_POSITION_M,
    INSERT_CAMERA_QUATERNION_WXYZ_ROS,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'src/zero_g_blade_swap/tasks/blade_swap/assets.py'
FIDUCIAL = ROOT / 'src/zero_g_blade_swap/fiducial.py'
ENV_CFG = ROOT / 'src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py'
CHANNEL = ROOT / 'evidence/destination_channel_geometry.json'
STRICT_RUN = ROOT / 'evidence/rgbd_strict_rack_retention_dual_camera_full_seed6070.json'
#: The datum layout that recorded run carried: one flush plate on the module
#: centre.  The self-validation is against what happened, so this is a fact
#: about that run and does not move when the shipped layout does.
RECORDED_RUN_DATUM_OFFSETS_X_M = (0.0,)

#: Cuboid sizes written inline in the asset factories rather than as
#: module-level constants.  ``source_bindings`` asserts each one is still the
#: literal the factory carries, so this file cannot describe a rack the scene
#: does not build.
GUIDE_SIZE_M = (0.60, 0.018, 0.050)
LIP_SIZE_M = (0.45, 0.020, 0.010)
FLOOR_SIZE_M = (0.60, 0.20, 0.014)
FLARE_SIZE_M = (0.080, 0.018, 0.050)
RACK_SIZE_M = (0.035, 0.72, 1.15)
RACK_POSITION_M = (1.005, 0.0, 0.76)
#: The slot floor's own centre, which is authored 24.5 mm below the seated
#: module rather than derived from it.
FLOOR_CENTRE_Z_M = 0.6955

#: Depth resolution of the sweep.  Half the guarded advance's 1 mm axial step,
#: so a boundary this reports is finer than a control step the chain can take.
SWEEP_STEP_M = 0.0005
#: Samples along each datum edge.  The occluder is a plate whose shadow crosses
#: the datum as a band, so edges decide readability, not corners.
EDGE_SAMPLES = 15


def _literal(name: str, path: Path) -> object:
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        targets: tuple = ()
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise KeyError(name)


def _quaternion_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rotation_about_y(radians: float) -> np.ndarray:
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _rotation_about_z(radians: float) -> np.ndarray:
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


class Box:
    """One oriented cuboid occluder in world coordinates."""

    def __init__(self, name: str, centre, size, rotation: np.ndarray | None = None) -> None:
        self.name = name
        self.centre = np.asarray(centre, dtype=np.float64)
        self.half = 0.5 * np.asarray(size, dtype=np.float64)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)

    def blocks(self, point: np.ndarray, camera: np.ndarray) -> bool:
        """Exact slab test on the segment from ``point`` to ``camera``."""

        origin = self.rotation.T @ (point - self.centre)
        direction = self.rotation.T @ (camera - point)
        near, far = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) < 1.0e-12:
                if abs(origin[axis]) > self.half[axis]:
                    return False
                continue
            first = (-self.half[axis] - origin[axis]) / direction[axis]
            second = (self.half[axis] - origin[axis]) / direction[axis]
            if first > second:
                first, second = second, first
            near, far = max(near, first), min(far, second)
            if near > far:
                return False
        return True


def _bay_boxes(bay_y: float, prefix: str, *, relieved: dict | None, lead_ins: bool) -> list[Box]:
    """Every part of one bay, as the scene builds it.

    ``relieved`` carries the destination bay's *built* surfaces, read from
    ``evidence/destination_channel_geometry.json`` rather than from the source,
    because ``configure_service_destination`` moves the guides, floor and lips
    at configuration time.  The lead-ins deliberately stay at the nominal
    surfaces, so they are built from the literals either way.
    """

    guide_offset = float(_literal('GUIDE_CENTER_OFFSET_Y', ASSETS))
    lip_half_width = float(_literal('SLOT_UPPER_LIP_HALF_WIDTH_Y', ASSETS))
    lip_centre_z = float(_literal('SLOT_UPPER_LIP_CENTER_Z', ASSETS))
    inserted = tuple(float(value) for value in _literal('BLADE_INSERTED_POS', ASSETS))
    flare_centre_x = float(_literal('_FLARE_CENTER_X', ASSETS))
    flare_outside = float(_literal('FLARE_CENTRE_OUTSIDE_RAIL_FACE_M', ASSETS))
    flare_deg = float(_literal('SLOT_ENTRY_FLARE_DEG', ASSETS))
    flare_centre_y = round(guide_offset - 0.5 * FLARE_SIZE_M[1] + flare_outside, 6)

    guide_y = (guide_offset, -guide_offset)
    floor_z = FLOOR_CENTRE_Z_M
    lip_z = lip_centre_z
    if relieved is not None:
        guide_y = (
            relieved['guide_body_centre_y_m'][0] - bay_y,
            relieved['guide_body_centre_y_m'][1] - bay_y,
        )
        floor_z = relieved['floor_centre_z_m']
        lip_z = relieved['upper_lip_centre_z_m']

    boxes = [
        Box(prefix + '_floor', (inserted[0], bay_y, floor_z), FLOOR_SIZE_M),
        Box(prefix + '_left_guide', (inserted[0], bay_y + guide_y[0], inserted[2]), GUIDE_SIZE_M),
        Box(prefix + '_right_guide', (inserted[0], bay_y + guide_y[1], inserted[2]), GUIDE_SIZE_M),
        Box(prefix + '_upper_left_lip', (0.825, bay_y + lip_half_width, lip_z), LIP_SIZE_M),
        Box(prefix + '_upper_right_lip', (0.825, bay_y - lip_half_width, lip_z), LIP_SIZE_M),
        Box(
            prefix + '_entry_left_flare',
            (flare_centre_x, bay_y + flare_centre_y, inserted[2]),
            FLARE_SIZE_M,
            _rotation_about_z(-math.radians(flare_deg)),
        ),
        Box(
            prefix + '_entry_right_flare',
            (flare_centre_x, bay_y - flare_centre_y, inserted[2]),
            FLARE_SIZE_M,
            _rotation_about_z(math.radians(flare_deg)),
        ),
    ]
    if lead_ins:
        ramp_surface_offset = flare_centre_y - (guide_offset - 0.009)
        ramp_size = (SLOT_ENTRY_RAMP_LENGTH_M, SLOT_ENTRY_RAMP_WIDTH_M, SLOT_ENTRY_RAMP_THICKNESS_M)
        boxes.extend(
            (
                Box(
                    prefix + '_entry_upper_ramp',
                    (flare_centre_x, bay_y, SLOT_LIP_BOTTOM_Z + ramp_surface_offset),
                    ramp_size,
                    _rotation_about_y(math.radians(SLOT_ENTRY_RAMP_DEG)),
                ),
                Box(
                    prefix + '_entry_lower_ramp',
                    (flare_centre_x, bay_y, SLOT_FLOOR_TOP_Z - ramp_surface_offset),
                    ramp_size,
                    _rotation_about_y(-math.radians(SLOT_ENTRY_RAMP_DEG)),
                ),
            )
        )
    return boxes


def _datum_edge_points(offset_x: float, half: float, tag_z: float) -> np.ndarray:
    """Points along the four edges of a square datum, in the module frame."""

    span = np.linspace(-half, half, EDGE_SAMPLES)
    points = [(offset_x + edge, along, tag_z) for edge in (-half, half) for along in span]
    points += [(offset_x + along, edge, tag_z) for edge in (-half, half) for along in span]
    return np.asarray(points, dtype=np.float64)


def _readability(
    points_module: np.ndarray,
    depths: np.ndarray,
    lateral: tuple[float, float],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    occluders: list[Box],
    focal_px: float,
    principal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Per-depth occlusion, frame containment and pixel depth for one outline.

    Returns ``(blocked, in_frame, first_occluder_depth_by_part)`` plus, through
    ``pixel_depths``, the camera-frame range each sample sits at, which is what
    the cell-resolution gate is computed from.
    """

    offsets = np.zeros((depths.size, 1, 3))
    offsets[:, 0, 0] = depths
    offsets[:, 0, 1] = lateral[0]
    offsets[:, 0, 2] = lateral[1]
    world = points_module[None, :, :] + offsets
    flat = world.reshape(-1, 3)

    blocked_flat = np.zeros(flat.shape[0], dtype=bool)
    culprits: dict[str, float] = {}
    for box in occluders:
        origin = (flat - box.centre) @ box.rotation
        direction = (camera_position - flat) @ box.rotation
        near = np.zeros(flat.shape[0])
        far = np.ones(flat.shape[0])
        alive = np.ones(flat.shape[0], dtype=bool)
        for axis in range(3):
            component = direction[:, axis]
            parallel = np.abs(component) < 1.0e-12
            alive &= ~(parallel & (np.abs(origin[:, axis]) > box.half[axis]))
            safe = np.where(parallel, 1.0, component)
            first = (-box.half[axis] - origin[:, axis]) / safe
            second = (box.half[axis] - origin[:, axis]) / safe
            low = np.where(parallel, 0.0, np.minimum(first, second))
            high = np.where(parallel, 1.0, np.maximum(first, second))
            near = np.maximum(near, low)
            far = np.minimum(far, high)
            alive &= near <= far
        hit = alive.reshape(depths.size, -1).any(axis=1)
        if hit.any():
            culprits.setdefault(box.name, float(depths[hit][0]))
        blocked_flat |= alive
    blocked = blocked_flat.reshape(depths.size, -1).any(axis=1)

    local = (world - camera_position) @ camera_rotation
    pixels = focal_px * local[:, :, :2] / local[:, :, 2:3] + principal
    in_frame = (
        (pixels[:, :, 0] >= 0.0).all(axis=1)
        & (pixels[:, :, 0] <= CAMERA_WIDTH_PX - 1).all(axis=1)
        & (pixels[:, :, 1] >= 0.0).all(axis=1)
        & (pixels[:, :, 1] <= CAMERA_HEIGHT_PX - 1).all(axis=1)
    )
    return blocked, in_frame, dict(sorted(culprits.items(), key=lambda item: item[1]))


def _worst_pixel_depth(
    points_module: np.ndarray,
    depths: np.ndarray,
    lateral: tuple[float, float],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> np.ndarray:
    offsets = np.zeros((depths.size, 1, 3))
    offsets[:, 0, 0] = depths
    offsets[:, 0, 1] = lateral[0]
    offsets[:, 0, 2] = lateral[1]
    world = points_module[None, :, :] + offsets
    local = (world - camera_position) @ camera_rotation
    return local[:, :, 2].max(axis=1)


def _span(depths: np.ndarray, flags: np.ndarray) -> dict[str, object]:
    selected = depths[flags]
    return {
        'from_module_centre_x_m': float(selected[0]) if selected.size else None,
        'to_module_centre_x_m': float(selected[-1]) if selected.size else None,
        'fraction_of_stroke': float(flags.mean()),
    }


def check(datum_offsets_m: tuple[float, ...] | None = None) -> dict[str, object]:
    tag_size = float(_literal('FIDUCIAL_TAG_SIZE_M', FIDUCIAL))
    quiet_size = float(_literal('FIDUCIAL_QUIET_ZONE_SIZE_M', FIDUCIAL))
    tag_centre = tuple(float(value) for value in _literal('FIDUCIAL_TAG_CENTER_M', FIDUCIAL))
    bits = _literal('FIDUCIAL_TAG_BITS', FIDUCIAL)
    cell_m = tag_size / len(bits)
    try:
        shipped_offsets = tuple(
            float(value) for value in _literal('FIDUCIAL_DATUM_OFFSETS_X_M', FIDUCIAL)
        )
    except KeyError:
        # The single-datum arm this check was written to diagnose.
        shipped_offsets = (tag_centre[0],)
    if datum_offsets_m is None:
        datum_offsets_m = shipped_offsets

    channel = json.loads(CHANNEL.read_text(encoding='utf-8'))
    chain_arm = next(arm for arm in channel['arms'] if arm['arm'] == 'chain')
    second_bay_y = float(_literal('SECOND_SLOT_CENTER_Y', ASSETS))

    occluders = _bay_boxes(second_bay_y, 'destination', relieved=chain_arm, lead_ins=True)
    occluders += _bay_boxes(0.0, 'source', relieved=None, lead_ins=False)
    occluders.append(Box('rack_backplane', RACK_POSITION_M, RACK_SIZE_M))

    cameras = {
        'primary': (
            np.asarray(CAMERA_POSITION_M, dtype=np.float64),
            _quaternion_matrix(CAMERA_QUATERNION_WXYZ_ROS),
        ),
        'insert': (
            np.asarray(INSERT_CAMERA_POSITION_M, dtype=np.float64),
            _quaternion_matrix(INSERT_CAMERA_QUATERNION_WXYZ_ROS),
        ),
    }
    focal_px = CAMERA_WIDTH_PX * CAMERA_FOCAL_LENGTH_MM / CAMERA_HORIZONTAL_APERTURE_MM
    principal = np.asarray(((CAMERA_WIDTH_PX - 1) / 2, (CAMERA_HEIGHT_PX - 1) / 2))

    # Derived exactly as ``assets.SERVICE_DESTINATION_SEATED_X`` derives it: the
    # depth at which an engaged jaw would enter the mouth, less the release
    # margin and the seated lag.  Recomputed rather than read, because that
    # constant is an expression and a copy of it could drift.
    seated_x = round(
        service_latch.release_before_blade_centre_x_m(
            SLOT_MOUTH_X, 0.5 * BLADE_LENGTH_M, service_latch.AXIAL_SEEK_MAX_M
        )
        - float(_literal('SERVICE_RELEASE_MARGIN_M', ASSETS))
        - float(_literal('SERVICE_SEATED_LAG_M', ASSETS)),
        6,
    )
    inserted_z = float(_literal('BLADE_INSERTED_POS', ASSETS)[2])
    lateral = (second_bay_y, inserted_z)
    depths = np.arange(TRANSIT_CLEAR_BLADE_CENTRE_X, seated_x + SWEEP_STEP_M, SWEEP_STEP_M)

    outlines = {
        'marker': 0.5 * tag_size,
        'marker_border_row_consumed': 0.5 * tag_size - cell_m,
        'quiet_zone': 0.5 * quiet_size,
    }

    per_datum: list[dict[str, object]] = []
    readable_anywhere = np.zeros(depths.shape, dtype=bool)
    readable_primary = np.zeros(depths.shape, dtype=bool)
    worst_margin_px = float('inf')
    worst_cell_px = float('inf')
    for offset in datum_offsets_m:
        entry: dict[str, object] = {'module_frame_offset_x_m': offset, 'cameras': {}}
        readable: dict[str, np.ndarray] = {}
        for camera_name, (camera_position, camera_rotation) in cameras.items():
            spans: dict[str, object] = {}
            for outline_name, half in outlines.items():
                points = _datum_edge_points(offset, half, tag_centre[2])
                blocked, in_frame, culprits = _readability(
                    points, depths, lateral, camera_position, camera_rotation, occluders, focal_px, principal
                )
                spans[outline_name] = {
                    'occluded': _span(depths, blocked),
                    'first_occluder_at_module_centre_x_m': culprits,
                }
                if outline_name == 'quiet_zone':
                    marker_points = _datum_edge_points(offset, 0.5 * tag_size, tag_centre[2])
                    pixel_depth = _worst_pixel_depth(
                        marker_points, depths, lateral, camera_position, camera_rotation
                    )
                    cell_px = focal_px * cell_m / pixel_depth
                    resolves = cell_px >= MIN_MARKER_CELL_PX
                    usable = ~blocked & in_frame & resolves
                    readable[camera_name] = usable
                    spans['out_of_frame'] = _span(depths, ~in_frame)
                    spans['below_cell_resolution'] = _span(depths, ~resolves)
                    spans['readable'] = _span(depths, usable)
                    if usable.any():
                        worst_cell_px = min(worst_cell_px, float(cell_px[usable].min()))
                        worst_margin_px = min(
                            worst_margin_px,
                            _frame_margin_px(
                                points, depths[usable], lateral, camera_position, camera_rotation,
                                focal_px, principal,
                            ),
                        )
            entry['cameras'][camera_name] = spans
        entry['readable_fraction_of_stroke_any_camera'] = float(
            (readable['primary'] | readable['insert']).mean()
        )
        per_datum.append(entry)
        readable_anywhere |= readable['primary'] | readable['insert']
        readable_primary |= readable['primary']

    blind_any = depths[~readable_anywhere]
    blind_primary = depths[~readable_primary]

    strict = json.loads(STRICT_RUN.read_text(encoding='utf-8'))
    recorded_stop_x = float(strict['guarded_insertion']['terminal_axial_target_m'][0])
    recorded_detections = strict['perception']['detector_availability']
    recorded = _recorded_bounds(
        occluders,
        cameras,
        depths,
        lateral,
        RECORDED_RUN_DATUM_OFFSETS_X_M,
        tag_centre[2],
        outlines,
        focal_px,
        principal,
    )
    touch = recorded['marker']
    consumed = recorded['marker_border_row_consumed']
    validated = touch is not None and consumed is not None and touch <= recorded_stop_x <= consumed

    source = ASSETS.read_text(encoding='utf-8')
    bindings = {
        'guide_size_literal': 'size=(length, 0.018, 0.050)' in source,
        'lip_size_literal': 'size=(0.45, 0.020, 0.010)' in source,
        'floor_size_literal': 'size=(0.60, 0.20, 0.014)' in source,
        'floor_centre_literal': 'pos=(BLADE_INSERTED_POS[0], 0.0, 0.6955)' in source,
        'flare_size_literal': 'size=(0.080, 0.018, 0.050)' in source,
        'rack_size_literal': 'size=(0.035, 0.72, 1.15)' in source,
        'rack_position_literal': 'pos=(1.005, 0.0, 0.76)' in source,
        'lead_ins_are_not_relieved': (
            'The lead-ins stay where they are' in ENV_CFG.read_text(encoding='utf-8')
        ),
        'module_section_unchanged': (BLADE_LENGTH_M, BLADE_WIDTH_M, BLADE_THICKNESS_M)
        == (0.45, 0.130, 0.020),
    }

    passed = validated and blind_any.size == 0 and all(bindings.values())
    return {
        'status': 'passed' if passed else 'failed',
        'title': 'What stands in the servicing cameras line to the flush datum along the seating stroke',
        'evidence_type': 'geometric_derivation_no_simulator',
        'generated_utc': datetime.now(UTC).isoformat(),
        'question': (
            'check_servicing_camera_geometry.py proves the datum is in frame and resolvable over the '
            'transit envelope. It never asked what is in front of it during insertion. This does.'
        ),
        'stroke': {
            'from_module_centre_x_m': float(depths[0]),
            'to_module_centre_x_m': seated_x,
            'derivation': 'transit-clear plane to the derived destination seated plane',
            'slot_mouth_x_m': SLOT_MOUTH_X,
            'step_m': SWEEP_STEP_M,
        },
        'datum': {
            'marker_size_m': tag_size,
            'quiet_zone_size_m': quiet_size,
            'cell_m': cell_m,
            'plane_in_module_m': tag_centre[2],
            'offsets_x_m': list(datum_offsets_m),
            'shipped_offsets_x_m': list(shipped_offsets),
            'evaluates_the_shipped_layout': tuple(datum_offsets_m) == tuple(shipped_offsets),
        },
        'per_datum': per_datum,
        'blind_stroke_any_camera_m': [float(blind_any[0]), float(blind_any[-1])] if blind_any.size else [],
        'blind_stroke_length_m': float(blind_any.size * SWEEP_STEP_M),
        'blind_stroke_primary_camera_only_m': (
            [float(blind_primary[0]), float(blind_primary[-1])] if blind_primary.size else []
        ),
        'unchanged_gates': {
            'minimum_quiet_zone_frame_margin_px': None if worst_margin_px == float('inf') else worst_margin_px,
            'minimum_marker_cell_px': None if worst_cell_px == float('inf') else worst_cell_px,
            'minimum_marker_cell_requirement_px': MIN_MARKER_CELL_PX,
            'requirement_source': 'check_servicing_camera_geometry.MIN_MARKER_CELL_PX',
            'measured_over': 'only the depths at which a datum is reported readable',
        },
        'self_validation': {
            'recorded_run': STRICT_RUN.name,
            'recorded_run_datum_offsets_x_m': list(RECORDED_RUN_DATUM_OFFSETS_X_M),
            'recorded_terminal_axial_target_m': recorded_stop_x,
            'recorded_detector_failures': recorded_detections['failures'],
            'recorded_max_consecutive_failures': recorded_detections['max_consecutive_failures'],
            'derived_marker_first_touched_m': touch,
            'derived_marker_border_row_consumed_m': consumed,
            'agrees': validated,
            'reads': (
                'the recorded advance had to stop between the depth at which the lead-in first covers any '
                'of that run marker outline in every view and the depth at which it has covered a whole '
                'border cell in every view'
            ),
        },
        'source_bindings': bindings,
        'scope_and_limitations': [
            'Occluders are the parts own cuboids; fillets, fasteners and cable runs are not modelled.',
            'Visibility is line of sight only. It does not predict decoding, exposure or motion blur.',
            'The module is swept at the seated lateral and vertical pose; the guarded envelope moves the '
            'datum by less than the plate clearances that decide the answer.',
            'The robot is not an occluder here: check_servicing_camera_geometry.py owns the rear-gripper '
            'sight line, and the gripper trails the module rear face by 83 mm along the whole stroke.',
        ],
    }


def _frame_margin_px(
    points_module: np.ndarray,
    depths: np.ndarray,
    lateral: tuple[float, float],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    focal_px: float,
    principal: np.ndarray,
) -> float:
    offsets = np.zeros((depths.size, 1, 3))
    offsets[:, 0, 0] = depths
    offsets[:, 0, 1] = lateral[0]
    offsets[:, 0, 2] = lateral[1]
    world = points_module[None, :, :] + offsets
    local = (world - camera_position) @ camera_rotation
    pixels = focal_px * local[:, :, :2] / local[:, :, 2:3] + principal
    return float(
        min(
            pixels[:, :, 0].min(),
            CAMERA_WIDTH_PX - 1 - pixels[:, :, 0].max(),
            pixels[:, :, 1].min(),
            CAMERA_HEIGHT_PX - 1 - pixels[:, :, 1].max(),
        )
    )


def _recorded_bounds(
    occluders: list[Box],
    cameras: dict[str, tuple[np.ndarray, np.ndarray]],
    depths: np.ndarray,
    lateral: tuple[float, float],
    offsets: tuple[float, ...],
    tag_z: float,
    outlines: dict[str, float],
    focal_px: float,
    principal: np.ndarray,
) -> dict[str, float | None]:
    """Depth at which the recorded run's datum set is first blocked in *every* view.

    The estimator takes the first camera that decodes, so a detection survives
    until the last view loses it, and with more than one datum until the last
    datum is lost.  The bound is therefore a maximum over both, and it is
    undefined if any view never loses the datum at all.
    """

    result: dict[str, float | None] = {}
    for outline_name, half in outlines.items():
        per_view: list[float | None] = []
        for camera_position, camera_rotation in cameras.values():
            for offset in offsets:
                points = _datum_edge_points(offset, half, tag_z)
                blocked, _in_frame, _culprits = _readability(
                    points, depths, lateral, camera_position, camera_rotation, occluders, focal_px, principal
                )
                selected = depths[blocked]
                per_view.append(float(selected[0]) if selected.size else None)
        result[outline_name] = None if any(value is None for value in per_view) else max(per_view)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument(
        '--datum_offsets_m',
        type=float,
        nargs='+',
        default=None,
        help='Module-frame x offsets of the datum set to evaluate. Defaults to the shipped datum.',
    )
    args = parser.parse_args()
    result = check(tuple(args.datum_offsets_m) if args.datum_offsets_m else None)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
