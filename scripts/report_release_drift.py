#!/usr/bin/env python3
"""Where in the sequence does the error that fails the task actually appear?

Every analysis in this repository until now read the *terminal* pose and asked
which upstream variable explained it. None of them asked when the error arrived.
The settle trace answers that directly, because it records lateral error on both
sides of the hand release, and the answer is not where any of the previous
framings looked.

The module is placed accurately. Across 186 episodes of the nominal chain the
lateral error while the robot still holds it is **inside the 2.5 mm criterion in
every single episode**. The error that fails the task appears afterwards, during
the unsupported settling window, and it appears by translation: the module
leaves the gripper carrying a residual velocity of about 2.4 mm/s and drifts for
the rest of the window with nothing to stop it. `rack_retention_engaged` is
false for all 186 episodes in this configuration, so nothing does.

That makes the failure ballistic rather than statistical. Drift predicted as
velocity x free-window duration tracks observed drift at rho = +0.84, and
selecting episodes by release velocity alone recovers the entire pass rate: at
2.0 mm/s every admitted episode passes.

**What this script does and does not show.** It shows, from recorded episodes,
that release velocity determines the outcome. It does *not* show that waiting
for a lower release velocity would work, because selecting episodes that already
had a low velocity is not the same intervention as holding on until the velocity
drops. The held-window velocity minimum is reported for exactly that reason: it
says how often a better release moment was available to take.

    python scripts/report_release_drift.py --report evidence/release_drift_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from handoff_qualification.records import DEFAULT_LATERAL_CRITERION_M  # noqa: E402
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

#: Control rate, for turning a free-window length into seconds.
CONTROL_HZ = 30.0
#: Release-velocity thresholds to report, in metres per second.
THRESHOLDS_MPS = (0.010, 0.006, 0.005, 0.004, 0.003, 0.0025, 0.002, 0.0015, 0.001)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator else 0.0


def episodes(trace_path: Path, outcome_path: Path) -> list[dict]:
    trace = np.load(trace_path, allow_pickle=True)
    fields = [str(name) for name in trace["settle_fields"]]
    settle = trace["settle"]
    outcome = np.load(outcome_path, allow_pickle=True)
    outcome_fields = [str(name) for name in outcome["fields"]]
    rows = outcome["rows"].astype(float)

    def column(block: np.ndarray, name: str) -> np.ndarray:
        return block[:, fields.index(name)]

    out = []
    for env in np.unique(column(settle, "env")).astype(int):
        block = settle[column(settle, "env") == env]
        # Sort by the global step. `steps_since_done` restarts between the
        # held and released windows, so sorting on it interleaves the two and
        # produces an alternating series that looks like a trend and is not.
        block = block[np.argsort(column(block, "step"))]
        released = column(block, "hand_released") > 0.5
        held, free = block[~released], block[released]
        if len(free) < 2 or len(held) < 1 or env >= len(rows):
            continue
        free_lateral = column(free, "lateral_error_m")
        out.append(
            {
                "env": int(env),
                "lateral_while_held_last_m": float(column(held, "lateral_error_m")[-1]),
                "lateral_while_held_max_m": float(column(held, "lateral_error_m").max()),
                "lateral_at_release_m": float(free_lateral[0]),
                "lateral_final_m": float(free_lateral[-1]),
                "drift_m": float(free_lateral[-1] - free_lateral[0]),
                "velocity_at_release_mps": float(column(free, "blade_linear_velocity_mps")[0]),
                "velocity_while_held_min_mps": float(
                    column(held, "blade_linear_velocity_mps").min()
                ),
                "free_steps": int(len(free)),
                "rack_retention_engaged": bool(column(free, "rack_retention_engaged").max() > 0.5),
                "terminal_lateral_m": float(rows[env, outcome_fields.index("lateral_error_m")]),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cohort_dir", type=Path, default=ROOT / "artifacts/traced_nominal"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[4070, 5070, 6070])
    parser.add_argument("--criterion_m", type=float, default=DEFAULT_LATERAL_CRITERION_M)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    records: list[dict] = []
    for seed in args.seeds:
        directory = args.cohort_dir / f"seed{seed}"
        trace, outcome = directory / "nominal_trace.npz", directory / "nominal.npz"
        if not (trace.exists() and outcome.exists()):
            print(f"missing {directory}, skipping")
            continue
        records.extend(episodes(trace, outcome))

    if not records:
        print("no episode carries a settle trace")
        return 1

    n = len(records)
    held_last = np.array([r["lateral_while_held_last_m"] for r in records])
    held_max = np.array([r["lateral_while_held_max_m"] for r in records])
    terminal = np.array([r["terminal_lateral_m"] for r in records])
    drift = np.array([r["drift_m"] for r in records])
    velocity = np.array([r["velocity_at_release_mps"] for r in records])
    held_min_velocity = np.array([r["velocity_while_held_min_mps"] for r in records])
    free_steps = np.array([r["free_steps"] for r in records])
    retention = np.array([r["rack_retention_engaged"] for r in records])

    predicted = velocity * (free_steps / CONTROL_HZ)
    rho = _spearman(predicted, drift)

    print(f"{n} episodes with a settle trace, criterion {args.criterion_m * 1000:.2f} mm\n")
    print("Where the error is, at each stage:")
    for label, values in (
        ("while held (last)", held_last),
        ("while held (worst)", held_max),
        ("at release", np.array([r["lateral_at_release_m"] for r in records])),
        ("terminal", terminal),
    ):
        inside = (values < args.criterion_m).mean()
        print(
            f"  {label:>20}  median {np.median(values) * 1000:6.3f} mm   "
            f"max {values.max() * 1000:7.3f} mm   inside criterion {inside:.4f}"
        )

    print()
    print(f"rack retention engaged during the free window: {int(retention.sum())}/{n}")
    print(f"drift, ballistic prediction against observed: rho = {rho:+.3f}")
    print(
        f"velocity at release {np.median(velocity) * 1000:.3f} mm/s median; "
        f"lowest reached while held {np.median(held_min_velocity) * 1000:.3f} mm/s median"
    )
    better = int((held_min_velocity < velocity).sum())
    print(f"episodes where a slower release moment was available while held: {better}/{n}")

    print()
    print("Selecting episodes by release velocity (a selection, not an intervention):")
    print(f"{'threshold mm/s':>15} {'admitted':>12} {'pass rate':>12}")
    gate = []
    for threshold in THRESHOLDS_MPS:
        mask = velocity <= threshold
        if not mask.any():
            continue
        rate = float((terminal[mask] < args.criterion_m).mean())
        gate.append(
            {
                "threshold_mps": threshold,
                "admitted": int(mask.sum()),
                "of": n,
                "pass_rate_among_admitted": rate,
            }
        )
        print(f"{threshold * 1000:>15.1f} {int(mask.sum()):>7}/{n:<4} {rate:>12.4f}")

    document = {
        "title": "The error that fails the task appears after the hand lets go",
        "evidence_type": "second_reading_of_recorded_traces",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Terminal pose analyses cannot say when the failing error arrived. The "
            "settle trace records lateral error on both sides of the hand release. "
            "Where does it appear?"
        ),
        "criterion_m": args.criterion_m,
        "source_revision": git_source_revision(ROOT),
        "episodes": n,
        "inside_criterion_while_held": float((held_last < args.criterion_m).mean()),
        "inside_criterion_while_held_worst_sample": float(
            (held_max < args.criterion_m).mean()
        ),
        "inside_criterion_terminal": float((terminal < args.criterion_m).mean()),
        "median_drift_m": float(np.median(drift)),
        "ballistic_prediction_spearman": rho,
        "median_velocity_at_release_mps": float(np.median(velocity)),
        "median_velocity_minimum_while_held_mps": float(np.median(held_min_velocity)),
        "episodes_with_a_slower_release_moment_available": better,
        "rack_retention_engaged_count": int(retention.sum()),
        "release_velocity_selection": gate,
        "per_episode": records,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "This is a second reading of existing episodes, not a new measurement.",
            "The velocity table is a SELECTION over episodes that already had a given "
            "release velocity. It is not evidence that waiting for a lower velocity "
            "would produce the same pass rate, because holding on longer is a "
            "different intervention and changes what the module does. Only a run with "
            "a velocity-conditioned release can show that.",
            "This configuration never engages rack retention. A configuration that "
            "does may fail differently, and the strict-chain cohort that scores 91.67% "
            "is exactly such a configuration.",
            "The held-window minimum says a slower release moment existed, not that a "
            "controller could have identified it online.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
