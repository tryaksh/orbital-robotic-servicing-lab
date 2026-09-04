"""Measure the attitude the transit actually hands the insertion over at.

**`DELIVERED_ATTITUDE_RAD = 0.046` is the most load-bearing number in this
project and no run has ever produced it.** It sets the clearance window's lower
bound at 10.350 mm, it is what puts this arm above the ~40 mrad threshold where
passive entry dies, and it is the "four times past the threshold" the boundary
claim is built on. Its docstring in `scripts/check_workcell_geometry.py` says it
is "reported in every robot-carried report as ``handoff_attitude_rad``". That
field does not exist: it has never been written by `run_workflow_demo.py`, it
appears in no report in `evidence/` or `artifacts/`, and searching the whole
history finds it only in the two docstrings that cite it.

Meanwhile the quantity *is* recorded, under another name and at every step.
`module_attitude_rad` in the transit trace is the axis-angle magnitude between
the module's world orientation and `RELOCATION_INSERT_STAGING_ROT`, which is the
insert task's own full-distance reset attitude -- the seated attitude. Its
comment in `run_workflow_demo.py` says what it is for: "how far off square the
module is". Read at the last transit sample, on episodes that went on to reach
the insert phase, that is the delivered attitude by the definition section 6.2
of the interface specification requires -- measured where the channel is not yet
touching the module.

So this needs no GPU. It reads recorded traces.

**What section 6.2 already warned about.** That section retracted a 63 to
67 mrad "delivered attitude" because it had been read off runs whose channel was
already open to 14-16 mm per side, where the number a module reports is `2c/L`,
the tilt *the channel* permits, not the tilt the arm produced. Every row of
`evidence/robot_carried_seating_sweep.json` lands within 13% of `2c/L`. The
constant this script measures against is 0.046, and section 9.9 of the same
specification records "attitude at the end" of a compliant mating stroke as
45.9 mrad -- taken on the 450x160x35 module, inside a channel with 15.75 mm of
lateral and 8.00 mm of vertical clearance per side. Whether 0.046 is that number
is not established here and this script does not assert it. What is established
is the measurement below, and that it does not agree.

Usage:

    python scripts/measure_delivered_attitude.py
    python scripts/measure_delivered_attitude.py --report evidence/<name>.json

Refuses to pool traces from different commits or from a dirty worktree, for the
same reason `aggregate_evaluation.py` does.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402
from zero_g_blade_swap.servicing_design import (  # noqa: E402
    ManipulatorPerformance,
    interface_regime,
    lateral_clearance_window,
)

#: The default cohort: the guarded arm of the hand-off study, three held-out
#: seeds at 32 environments on one clean commit. It is the only recorded set of
#: transit traces that is a cohort rather than a probe -- `artifacts/robotcarried`
#: has sixty more and they are single-environment runs with no source binding.
DEFAULT_TRACES = tuple(
    f"artifacts/conditioned-insertion/v1/chain_handoff_seed{seed}_guarded_trace.npz"
    for seed in (4070, 5070, 6070)
)

#: The phase index at which the transit has handed over. `PHASE_NAMES` in
#: `run_workflow_demo.py` is ("capture", "seat", "extract", "transit", "insert",
#: "done"), so an episode that reached 4 got as far as the insertion.
INSERT_PHASE_INDEX = 4

MODULE_LENGTH_M = 0.450
SEATING_STROKE_M = 0.529
#: The as-built destination channel, unrelieved. The regime question is about the
#: bay the rack derives, not about the 4.6125 mm of relief laid over it.
AS_BUILT_CLEARANCE_PER_SIDE_M = 0.011065
SEATING_TOLERANCE_RAD = 0.0523599
PAD_HALF_BEARING_OFFSET_M = 0.015
#: What `check_workcell_geometry.py` currently asserts, for comparison only.
ASSERTED_DELIVERED_ATTITUDE_RAD = 0.046


def handover_attitudes(trace_path: Path) -> dict[str, Any]:
    """Return the per-environment attitude at the last transit sample.

    Restricted to environments whose episode reached the insert phase: an
    environment that timed out in transit never handed anything over, and its
    last transit sample is a failure state rather than a hand-off.
    """

    trace = np.load(trace_path, allow_pickle=True)
    if "transit_fields" not in trace:
        raise ValueError(f"{trace_path} has no transit block")
    fields = [str(name) for name in trace["transit_fields"]]
    for required in ("module_attitude_rad", "env", "step"):
        if required not in fields:
            raise ValueError(f"{trace_path} transit block has no {required!r}")
    rows = trace["transit"].astype(float)
    attitude, env, step = (fields.index(name) for name in ("module_attitude_rad", "env", "step"))

    episodes = trace_path.with_name(trace_path.name.replace("_trace.npz", ".npz"))
    if not episodes.is_file():
        raise FileNotFoundError(f"no episode metrics beside {trace_path}")
    metrics = np.load(episodes, allow_pickle=True)
    metric_fields = [str(name) for name in metrics["fields"]]
    reached = metrics["rows"].astype(float)[:, metric_fields.index("reached_phase")]
    # The npz carries its metadata as a JSON string, exactly as aggregate_evaluation.py reads it.
    metadata = json.loads(str(metrics["metadata"].item()))

    per_env: dict[int, float] = {}
    for index in np.unique(rows[:, env]):
        block = rows[rows[:, env] == index]
        per_env[int(index)] = float(block[np.argsort(block[:, step])][-1, attitude])

    kept = {
        index: value
        for index, value in per_env.items()
        if index < len(reached) and reached[index] >= INSERT_PHASE_INDEX
    }
    return {
        "trace": trace_path.as_posix(),
        "seed": int(metadata.get("seed", -1)),
        "task": str(metadata.get("task", "")),
        "source_revision": metadata.get("source_revision"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "environments_traced": len(per_env),
        "environments_that_handed_over": len(kept),
        "attitudes_rad": [kept[index] for index in sorted(kept)],
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "median_rad": float(np.median(array)),
        "mean_rad": float(array.mean()),
        "p90_rad": float(np.percentile(array, 90)),
        "p95_rad": float(np.percentile(array, 95)),
        "max_rad": float(array.max()),
        "min_rad": float(array.min()),
    }


def implied_requirement(attitude_rad: float) -> dict[str, Any]:
    """What a channel would have to be, if the arm delivered this attitude."""

    performance = ManipulatorPerformance(
        delivered_attitude_rad=attitude_rad,
        seating_tolerance_rad=SEATING_TOLERANCE_RAD,
        pad_half_bearing_offset_m=PAD_HALF_BEARING_OFFSET_M,
    )
    window = lateral_clearance_window(performance, MODULE_LENGTH_M)
    regime = interface_regime(
        performance,
        module_length_m=MODULE_LENGTH_M,
        seating_stroke_m=SEATING_STROKE_M,
        clearance_per_side_m=AS_BUILT_CLEARANCE_PER_SIDE_M,
    )
    return {
        "delivered_attitude_rad": attitude_rad,
        "lateral_clearance_window_m": window,
        "interface_regime": regime["regime"],
    }


def build_report(traces: list[dict[str, Any]]) -> dict[str, Any]:
    revisions = [trace["source_revision"] for trace in traces]
    if any(not isinstance(revision, dict) for revision in revisions):
        raise ValueError("a trace carries no source revision; it cannot be pooled")
    unusable = [
        revision
        for revision in revisions
        if not revision.get("available") or not revision.get("commit") or revision.get("dirty") is not False
    ]
    if unusable:
        raise ValueError("traces must come from available, clean source revisions")
    commits = {str(revision["commit"]) for revision in revisions}
    if len(commits) != 1:
        raise ValueError(f"traces came from different source commits: {sorted(commits)}")
    tasks = sorted({trace["task"] for trace in traces})
    if len(tasks) != 1:
        raise ValueError(f"traces came from different tasks: {tasks}")

    pooled: list[float] = []
    for trace in traces:
        pooled.extend(trace["attitudes_rad"])
    if not pooled:
        raise ValueError("no environment in any trace reached the insert phase")

    statistics = summarize(pooled)
    asserted = implied_requirement(ASSERTED_DELIVERED_ATTITUDE_RAD)
    return {
        "title": "The attitude the transit hands the insertion over at, from recorded traces",
        "evidence_type": "measurement_from_recorded_traces",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "check_workcell_geometry.py asserts a delivered attitude of 46 mrad and cites a report "
            "field that has never been written. What do the recorded transit traces say the module's "
            "attitude actually is when the transit hands over?"
        ),
        "measurement": {
            "field": "module_attitude_rad, last transit sample",
            "definition": (
                "The axis-angle magnitude between the module's world orientation and "
                "RELOCATION_INSERT_STAGING_ROT, which is the insert task's full-distance reset "
                "attitude. Recorded every second control step during transit by run_workflow_demo.py."
            ),
            "restricted_to": "environments whose episode reached the insert phase",
            "task": tasks[0],
            "source_commit": sorted(commits)[0],
            "seeds": sorted(trace["seed"] for trace in traces),
            "per_seed": [
                {
                    "seed": trace["seed"],
                    "environments_traced": trace["environments_traced"],
                    "environments_that_handed_over": trace["environments_that_handed_over"],
                    **summarize(trace["attitudes_rad"]),
                }
                for trace in traces
            ],
            "pooled": statistics,
        },
        "what_the_measurement_implies": {
            "at_the_median": implied_requirement(statistics["median_rad"]),
            "at_p95": implied_requirement(statistics["p95_rad"]),
            "at_the_worst_environment": implied_requirement(statistics["max_rad"]),
        },
        "what_the_asserted_constant_implies": asserted,
        "asserted_delivered_attitude_rad": ASSERTED_DELIVERED_ATTITUDE_RAD,
        "ratio_asserted_over_measured_median": ASSERTED_DELIVERED_ATTITUDE_RAD / statistics["median_rad"],
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            (
                "Measured on the exact-state chain (Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0) "
                "with the scripted guarded advance. It does not describe the camera-driven chain, "
                "where every module-state channel is estimated, and a camera-driven delivered "
                "attitude has not been measured."
            ),
            (
                "The last transit sample is at most two control steps before the hand-off, not the "
                "hand-off instant itself."
            ),
            (
                "This is a measurement of what the transit delivers. It is not a re-derivation of the "
                "rack: GUIDE_CENTER_OFFSET_Y, the clearance window and every boundary result "
                "downstream still stand on DELIVERED_ATTITUDE_RAD = 0.046, and moving that constant "
                "moves all of them. This report is the evidence for deciding whether to."
            ),
            (
                "It does not establish where 0.046 came from. It establishes that no report in this "
                "repository contains the field its docstring cites, and that the recorded traces "
                "disagree with it."
            ),
        ],
        "source_revision": git_source_revision(ROOT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--traces",
        nargs="+",
        type=Path,
        default=[ROOT / path for path in DEFAULT_TRACES],
        help="Transit trace archives to pool. Default: the guarded hand-off cohort, three seeds.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write the result as evidence JSON.")
    arguments = parser.parse_args()

    missing = [path for path in arguments.traces if not path.is_file()]
    if missing:
        print("MISSING traces:")
        for path in missing:
            print(f"  {path}")
        print("\nThese live under artifacts/, which is gitignored, so a clean checkout has none of them.")
        return 66

    traces = [handover_attitudes(path) for path in arguments.traces]
    report = build_report(traces)
    pooled = report["measurement"]["pooled"]

    print(report["title"])
    print(f"  task    {report['measurement']['task']}")
    print(f"  commit  {report['measurement']['source_commit'][:12]}")
    print()
    print("  seed   handed over   median      p95      max")
    for row in report["measurement"]["per_seed"]:
        print(
            f"  {row['seed']:<6d} {row['environments_that_handed_over']:>3d}/{row['environments_traced']:<3d}"
            f"      {row['median_rad'] * 1000:7.2f}  {row['p95_rad'] * 1000:7.2f}  {row['max_rad'] * 1000:7.2f} mrad"
        )
    print(
        f"  pooled {pooled['count']:>7d}      {pooled['median_rad'] * 1000:7.2f}"
        f"  {pooled['p95_rad'] * 1000:7.2f}  {pooled['max_rad'] * 1000:7.2f} mrad"
    )
    print()
    print(f"  asserted by check_workcell_geometry.py: {ASSERTED_DELIVERED_ATTITUDE_RAD * 1000:.1f} mrad")
    print(f"  ratio to the measured median:           {report['ratio_asserted_over_measured_median']:.2f}x")
    print()
    print("  what each implies for the interface")
    for label, key in (
        ("measured median", "at_the_median"),
        ("measured p95", "at_p95"),
        ("worst environment", "at_the_worst_environment"),
    ):
        block = report["what_the_measurement_implies"][key]
        window = block["lateral_clearance_window_m"]
        print(
            f"    {label:18s} {block['delivered_attitude_rad'] * 1000:6.2f} mrad  "
            f"clearance >= {window['lower_bound_m'] * 1000:6.3f} mm   regime {block['interface_regime']}"
        )
    block = report["what_the_asserted_constant_implies"]
    window = block["lateral_clearance_window_m"]
    print(
        f"    {'asserted constant':18s} {block['delivered_attitude_rad'] * 1000:6.2f} mrad  "
        f"clearance >= {window['lower_bound_m'] * 1000:6.3f} mm   regime {block['interface_regime']}"
    )

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
