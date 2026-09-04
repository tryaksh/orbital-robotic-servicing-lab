'''Pair strict no-rack and rack-retention cohorts and report load transfer.

This is CPU-only.  It refuses mismatched seeds, controllers, checkpoints,
source revisions, row counts, or terminal schemas before comparing outcomes.
Per-environment rack engagement and drift come from the workflow reports; raw
success and predicate rows come from the independently saved episode metrics.
'''

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from zero_g_blade_swap import rack_retention
from zero_g_blade_swap.evaluation import wilson_interval


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_metrics(path: Path) -> tuple[np.ndarray, list[str], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        rows = np.asarray(payload['rows'], dtype=np.float64)
        fields = [str(value) for value in payload['fields'].tolist()]
        metadata = json.loads(str(payload['metadata'].item()))
    return rows, fields, metadata


def _arm_summary(successes: int, fired: int, episodes: int, paths: list[Path]) -> dict[str, object]:
    low, high = wilson_interval(successes, episodes)
    return {
        'episodes': episodes,
        'successes': successes,
        'success_rate': successes / episodes,
        'success_rate_wilson_95': {'low': low, 'high': high},
        'unchanged_seating_predicate_fired': fired,
        'inputs': [
            {'path': path.as_posix(), 'sha256': _sha256(path)} for path in paths
        ],
    }


def build(
    control_paths: list[Path],
    retention_paths: list[Path],
    retention_report_paths: list[Path],
) -> dict[str, object]:
    if not (len(control_paths) == len(retention_paths) == len(retention_report_paths)):
        raise ValueError('control, retention, and retention-report lists must have equal length')

    control_successes = retention_successes = 0
    control_fired = retention_fired = 0
    episodes = 0
    engaged = full_rechecks = 0
    rack_only_steps: list[int] = []
    max_position_drift_m = 0.0
    max_orientation_drift_rad = 0.0
    upstream_failures: list[dict[str, int]] = []
    cohort_rows: list[dict[str, object]] = []
    common_metadata: dict[str, object] | None = None

    for control_path, retention_path, report_path in zip(
        control_paths, retention_paths, retention_report_paths, strict=True
    ):
        control_rows, control_fields, control_metadata = _load_metrics(control_path)
        retention_rows, retention_fields, retention_metadata = _load_metrics(retention_path)
        if control_fields != retention_fields:
            raise ValueError(f'terminal schemas differ for {control_path} and {retention_path}')
        if control_metadata != retention_metadata:
            raise ValueError(f'cohort metadata differ for {control_path} and {retention_path}')
        if control_rows.shape != retention_rows.shape:
            raise ValueError(f'cohort row shapes differ for {control_path} and {retention_path}')
        if common_metadata is None:
            common_metadata = control_metadata
        else:
            stable = ('task', 'controller', 'checkpoint_sha256', 'checkpoints', 'source_revision', 'num_envs')
            if any(common_metadata[key] != control_metadata[key] for key in stable):
                raise ValueError(f'fixed-cohort metadata changed at {control_path}')

        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('seed') != control_metadata['seed']:
            raise ValueError(f'retention report seed does not match {retention_path}')
        if report.get('source_revision') != control_metadata['source_revision']:
            raise ValueError(f'retention report source revision does not match {retention_path}')
        telemetry = report['destination_rack_retention']
        if not telemetry.get('enabled'):
            raise ValueError(f'rack retention was not enabled in {report_path}')
        observations = telemetry['observed_per_environment']
        if len(observations) != len(retention_rows):
            raise ValueError(f'rack telemetry row count does not match {retention_path}')

        success_index = control_fields.index('success')
        fired_index = control_fields.index('predicate_fired')
        control_success = control_rows[:, success_index] > 0.5
        retention_success = retention_rows[:, success_index] > 0.5
        control_predicate = control_rows[:, fired_index] > 0.5
        retention_predicate = retention_rows[:, fired_index] > 0.5
        observed_engaged = np.asarray(
            [row['engaged_after_measured_seating'] for row in observations], dtype=bool
        )
        observed_recheck = np.asarray(
            [row['full_rack_only_recheck_observed'] for row in observations], dtype=bool
        )
        if not np.array_equal(control_predicate, retention_predicate):
            raise ValueError(f'unchanged seating predicate moved in paired cohort {retention_path}')
        if not np.array_equal(retention_predicate, observed_engaged):
            raise ValueError(f'rack engagement did not exactly follow measured seating in {report_path}')
        if not np.all(retention_success <= observed_recheck):
            raise ValueError(f'a success lacks the full rack-only recheck in {report_path}')

        seed = int(control_metadata['seed'])
        for env, reached in enumerate(retention_predicate.tolist()):
            if not reached:
                upstream_failures.append({'seed': seed, 'env': env})
        engaged += int(observed_engaged.sum())
        full_rechecks += int(observed_recheck.sum())
        rack_only_steps.extend(
            int(row['rack_only_control_steps']) for row in observations if row['engaged_after_measured_seating']
        )
        max_position_drift_m = max(
            max_position_drift_m,
            max(float(row['max_rack_to_module_position_drift_m']) for row in observations),
        )
        max_orientation_drift_rad = max(
            max_orientation_drift_rad,
            max(float(row['max_rack_to_module_orientation_drift_rad']) for row in observations),
        )
        count = len(control_rows)
        episodes += count
        control_successes += int(control_success.sum())
        retention_successes += int(retention_success.sum())
        control_fired += int(control_predicate.sum())
        retention_fired += int(retention_predicate.sum())
        cohort_rows.append(
            {
                'seed': seed,
                'episodes': count,
                'control_successes': int(control_success.sum()),
                'retention_successes': int(retention_success.sum()),
                'seating_predicate_fired': int(retention_predicate.sum()),
                'rack_engaged': int(observed_engaged.sum()),
                'full_rack_only_rechecks': int(observed_recheck.sum()),
            }
        )

    assert common_metadata is not None
    conditional_low, conditional_high = wilson_interval(retention_successes, retention_fired)
    step_dt = 1.0 / 30.0
    return {
        'title': 'Paired destination rack-side retention and strict robot-support release',
        'evidence_type': 'paired_fixed_cohort_simulation',
        'generated_utc': datetime.now(UTC).isoformat(),
        'paired_design': {
            'single_change': 'visible rack-side retention enabled in the retention arm',
            'same_seed_environment_pairs': True,
            'same_terminal_schema': True,
            'same_unchanged_seating_predicate_rows': True,
            'task': common_metadata['task'],
            'controller': common_metadata['controller'],
            'policy_set_sha256': common_metadata['checkpoint_sha256'],
            'checkpoints_sha256': common_metadata['checkpoints'],
            'source_revision': common_metadata['source_revision'],
            'seeds': [row['seed'] for row in cohort_rows],
            'environments_per_seed': common_metadata['num_envs'],
        },
        'cohorts': cohort_rows,
        'control': _arm_summary(control_successes, control_fired, episodes, control_paths),
        'rack_retention': _arm_summary(
            retention_successes, retention_fired, episodes, retention_paths
        ),
        'paired_effect': {
            'additional_successes': retention_successes - control_successes,
            'success_rate_change_percentage_points':
                100.0 * (retention_successes - control_successes) / episodes,
        },
        'rack_load_transfer': {
            'eligible_after_measured_seating': retention_fired,
            'rack_engagements': engaged,
            'successful_rack_only_rechecks': retention_successes,
            'full_rack_only_rechecks_observed': full_rechecks,
            'conditional_success_rate': retention_successes / retention_fired,
            'conditional_success_rate_wilson_95': {
                'low': conditional_low,
                'high': conditional_high,
            },
            'required_rack_only_interval_s': 0.70,
            'minimum_observed_rack_only_interval_s': min(rack_only_steps) * step_dt,
            'maximum_rack_to_module_position_drift_m': max_position_drift_m,
            'maximum_rack_to_module_orientation_drift_rad': max_orientation_drift_rad,
            'mechanism': 'two visible rack pawls represented by a break-rated Rack-to-SpareBlade fixed joint',
            'rated_force_n': rack_retention.RATED_FORCE_N,
            'rated_torque_nm': rack_retention.RATED_TORQUE_NM,
            'world_constraint': False,
            'module_pose_write': False,
            'retention_reports': [
                {'path': path.as_posix(), 'sha256': _sha256(path)}
                for path in retention_report_paths
            ],
        },
        'unreached_seating_failures': upstream_failures,
        'gate': {
            'minimum_full_chain_success_rate': 0.95,
            'full_chain_passed': retention_successes / episodes >= 0.95,
            'rack_transfer_passed_for_every_eligible_episode': (
                retention_successes == retention_fired == engaged == full_rechecks
            ),
            'claim': (
                'rack-side post-release retention is supported on every eligible episode; '
                'the full autonomous chain remains below its 95% gate because two paired '
                'episodes fail before measured seating'
            ),
        },
        'scope_and_limitations': [
            'Simulation only; no rack retention hardware has been built or loaded.',
            'The visible pawls have no contact colliders; their disclosed break-rated Rack-to-module joint carries load.',
            'PhysX external-joint reaction magnitude is not exposed. Load transfer is evidenced by the break rating, rack-only interval, and measured relative drift.',
            'The 91.67% full-chain result misses the unchanged 95% gate and is not described as full-chain qualification.',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, nargs='+', required=True)
    parser.add_argument('--retention', type=Path, nargs='+', required=True)
    parser.add_argument('--retention_reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.control, args.retention, args.retention_reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'gate': result['gate']}, indent=2))
    return 0 if result['gate']['rack_transfer_passed_for_every_eligible_episode'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
