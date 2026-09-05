#!/usr/bin/env python3
"""Did turning the trace on change the run it was measuring?

`--handoff_trace` samples state during the episode. If that sampling perturbs
anything -- timing, ordering, the RNG stream -- then the pre-handoff state it
records belongs to a different run than the outcomes already published, and
joining the two would be a quiet fabrication.

So the traced cohort is compared against the untraced one it was meant to
reproduce, episode by episode, at the same seeds. Three things must hold:

* the same number of episodes pass;
* the same *individual* episodes pass -- a matching rate with different episodes
  passing would mean the run changed and the aggregate happened to survive;
* the terminal residuals agree to within a tolerance the caller states.

The third is the strict one. This project does not assume bitwise determinism --
NVIDIA documents that it is not guaranteed across environment counts and
hardware -- so the tolerance is an argument and the measured agreement is
reported rather than asserted.

    python scripts/check_trace_is_non_perturbing.py \\
        --original artifacts/robustness64_corrected/nominal.npz \\
        --traced artifacts/traced_nominal/seed4070/nominal.npz \\
        --report evidence/trace_non_perturbing_v1.json
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

from handoff_qualification import load_cohort  # noqa: E402
from handoff_qualification.records import DEFAULT_LATERAL_CRITERION_M  # noqa: E402


def compare(original: Path, traced: Path, criterion_m: float, tolerance_m: float) -> dict:
    left = load_cohort(original)
    right = load_cohort(traced)
    if len(left) != len(right):
        return {
            "original": original.as_posix(),
            "traced": traced.as_posix(),
            "agrees": False,
            "why": f"episode counts differ: {len(left)} against {len(right)}",
        }

    left_residual = np.array([r.residual_lateral_m for r in left])
    right_residual = np.array([r.residual_lateral_m for r in right])
    left_pass = left_residual < criterion_m
    right_pass = right_residual < criterion_m

    difference = np.abs(left_residual - right_residual)
    identical_episodes = int((left_pass == right_pass).sum())
    within = bool(difference.max() <= tolerance_m)

    return {
        "original": original.as_posix(),
        "traced": traced.as_posix(),
        "episodes": len(left),
        "original_successes": int(left_pass.sum()),
        "traced_successes": int(right_pass.sum()),
        "episodes_agreeing": identical_episodes,
        "max_residual_difference_m": float(difference.max()),
        "median_residual_difference_m": float(np.median(difference)),
        "tolerance_m": tolerance_m,
        "agrees": bool(
            identical_episodes == len(left)
            and int(left_pass.sum()) == int(right_pass.sum())
            and within
        ),
        "traced_carries_pre_handoff_rows": int(sum(1 for r in right if r.incoming)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--original", type=Path, action="append", required=True)
    parser.add_argument("--traced", type=Path, action="append", required=True)
    parser.add_argument("--criterion_m", type=float, default=DEFAULT_LATERAL_CRITERION_M)
    parser.add_argument(
        "--tolerance_m",
        type=float,
        default=1e-9,
        help="how far a residual may move and still count as the same run",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if len(args.original) != len(args.traced):
        print("pass one --traced for every --original")
        return 1

    pairs = [
        compare(original, traced, args.criterion_m, args.tolerance_m)
        for original, traced in zip(args.original, args.traced, strict=True)
    ]

    all_agree = all(pair["agrees"] for pair in pairs)
    for pair in pairs:
        if "why" in pair:
            print(f"{Path(pair['traced']).parent.name}: REFUSED -- {pair['why']}")
            continue
        print(
            f"{Path(pair['traced']).parent.name}: "
            f"{pair['original_successes']}/{pair['episodes']} against "
            f"{pair['traced_successes']}/{pair['episodes']}, "
            f"{pair['episodes_agreeing']}/{pair['episodes']} episodes agree, "
            f"max residual difference {pair['max_residual_difference_m'] * 1000:.6f} mm"
        )
        print(f"    pre-handoff rows recorded: {pair['traced_carries_pre_handoff_rows']}")

    print()
    print(
        "The trace is non-perturbing; its pre-handoff state may be joined to "
        "outcomes already published."
        if all_agree
        else "The traced run is NOT the same run. Do not join its state to published outcomes."
    )

    document = {
        "title": "Whether recording the handoff trace changes the run",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "The cohort whose outcomes vary was run without --handoff_trace. Before its "
            "pre-handoff state can be joined to those outcomes, the traced re-run has to "
            "be the same run."
        ),
        "criterion_m": args.criterion_m,
        "tolerance_m": args.tolerance_m,
        "pairs": pairs,
        "all_agree": all_agree,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "Agreement is measured on this machine, at this commit, at these seeds, at "
            "the same environment count. NVIDIA documents that simulator results are "
            "not guaranteed reproducible across environment counts or hardware, so "
            "nothing here claims determinism beyond what was measured.",
            "This checks that the trace does not perturb the run. It does not check "
            "that the recorded fields mean what their names say.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0 if all_agree else 2


if __name__ == "__main__":
    raise SystemExit(main())
