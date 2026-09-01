'''Prove the rack-side retention geometry and its simulator binding without Isaac Sim.

The pawl dimensions come from the module, lead-in, and existing service-latch
interface.  This check also reads the shipped scene and workflow sources before
reporting, so a passing calculation cannot silently describe geometry or an
engagement condition that the simulator no longer uses.
'''

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from zero_g_blade_swap import rack_retention as retention
from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    BLADE_THICKNESS_M,
    BLADE_WIDTH_M,
    SLOT_ENTRY_RAMP_CATCH_M,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'src/zero_g_blade_swap/tasks/blade_swap/assets.py'
DRIVER = ROOT / 'scripts/run_workflow_demo.py'


def check() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def record(name: str, margin_m: float, requirement: str) -> None:
        checks.append(
            {
                'check': name,
                'requirement': requirement,
                'margin_m': round(margin_m, 9),
                'passed': margin_m >= -1.0e-12,
            }
        )

    record(
        'open_pawls_clear_complete_lead_in',
        retention.PAWL_OPEN_INNER_HALF_GAP_M
        - 0.5 * BLADE_WIDTH_M
        - SLOT_ENTRY_RAMP_CATCH_M,
        'the open inner face is outside the module plus the complete lead-in catch',
    )
    record(
        'closed_pawls_overlap_module_rear',
        0.5 * BLADE_WIDTH_M - retention.PAWL_CLOSED_INNER_HALF_GAP_M,
        'each closed tip overlaps the module rear face by a positive distance',
    )
    seated = (0.0, 0.0, 0.0)
    boxes = retention.pawl_tip_boxes(seated)
    rear_face_x = seated[0] - 0.5 * BLADE_LENGTH_M
    forward_face_x = max(centre[0] + 0.5 * size[0] for _, centre, size in boxes)
    record(
        'closed_pawls_do_not_push_the_module',
        rear_face_x - forward_face_x,
        'closed tips remain behind the seated module instead of correcting its pose',
    )
    record(
        'rack_joint_rating_covers_required_axial_load',
        (retention.RATED_FORCE_N - retention.REQUIRED_AXIAL_CAPACITY_N) / 1000.0,
        'the disclosed joint rating is no lower than the derived axial load requirement',
    )

    assets = ASSETS.read_text(encoding='utf-8')
    driver = DRIVER.read_text(encoding='utf-8')
    scene_blade_size: tuple[float, ...] | None = None
    for node in ast.parse(assets).body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == 'BLADE_SIZE' for target in node.targets):
            scene_blade_size = tuple(float(value) for value in ast.literal_eval(node.value))
            break
    bindings = {
        'scene_module_geometry_matches_checked_dimensions': scene_blade_size
        == (BLADE_LENGTH_M, BLADE_WIDTH_M, BLADE_THICKNESS_M),
        'scene_uses_checked_tip_boxes': 'rack_retention.pawl_tip_boxes(cfg.seated_module_position)' in assets,
        'scene_uses_checked_open_stroke': 'rack_retention.pawl_translation(engaged=False, sign=sign)' in assets,
        'joint_is_rack_to_module': (
            "body0_relative_path: str = 'Rack'" in assets
            and "body1_relative_path: str = 'SpareBlade'" in assets
        ),
        'joint_starts_disabled': 'enabled: bool = False' in assets,
        'workflow_engages_on_unchanged_predicate': 'self.rack_retention.engage(fired, step)' in driver,
        'workflow_captures_live_relative_pose': (
            'blade.data.root_pos_w - rack.data.root_pos_w' in driver
            and 'joint.GetLocalPos0Attr().Set' in driver
        ),
    }
    passed = all(row['passed'] for row in checks) and all(bindings.values())
    return {
        'status': 'passed' if passed else 'failed',
        'title': 'Rack-side passive retention geometry and simulator binding',
        'evidence_type': 'geometric_derivation_no_simulator',
        'geometry': {
            'simulator_module_size_m': scene_blade_size,
            'module_width_m': BLADE_WIDTH_M,
            'pawl_open_inner_half_gap_m': retention.PAWL_OPEN_INNER_HALF_GAP_M,
            'pawl_closed_inner_half_gap_m': retention.PAWL_CLOSED_INNER_HALF_GAP_M,
            'pawl_close_stroke_m': retention.PAWL_CLOSE_STROKE_M,
            'pawl_overlap_m': retention.PAWL_OVERLAP_M,
            'pawl_face_clearance_m': retention.PAWL_FACE_CLEARANCE_M,
            'rated_force_n': retention.RATED_FORCE_N,
            'rated_torque_nm': retention.RATED_TORQUE_NM,
        },
        'simulator_configuration_binding': bindings,
        'checks': checks,
        'scope_and_limitations': [
            'The visible pawls disclose the rack-owned load path; contact on their visual boxes is not simulated.',
            'Load transfer is modelled by the reported break-rated Rack-to-SpareBlade fixed joint.',
            'Simulation only; no rack retention hardware has been built or loaded.',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    result = check()
    print(json.dumps(result, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
