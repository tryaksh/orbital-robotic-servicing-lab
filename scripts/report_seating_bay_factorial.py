"""Which matters more, the seating controller or the bay it seats into.

**Two factors, four cells, one checkpoint set, and the answer is the bay.** Every
arm below is the same capture and extraction weights, the same force-feedback
insert weights loaded, the same task, the same three held-out seeds and the same
eight environments. Two flags change: `--insert_controller` and
`--destination_channel_relief_m`.

    -----------------------------------------------------------------
                              scripted guarded    learned force
                              advance             seating policy
    channel throat 11.065 mm  the library's prescription
    channel throat 15.678 mm  the bay every published chain number comes from
    -----------------------------------------------------------------

The 15.678 mm throat is 3.897 mm past the design library's own upper bound, and
that is not an accident of configuration: the source bay is built to the
prescription and the destination is relieved past it, which is what buys entry
admissibility by spending the seating bound.

**Every comparison here is paired and is read paired.** The cells share seeds,
checkpoints and deterministic resets, so episode *i* of one arm starts from the
state episode *i* of the other did -- which is what makes McNemar's exact test on
the discordant episodes the right reading. Two overlapping Wilson intervals report
"inconclusive" on data that answers the question. Both readings are printed,
because "how far apart are these arms" and "what is this arm's rate" are different
questions.

**A null with many discordant pairs is a result.** At the prescribed throat the two
controllers differ by one episode in ninety-six and thirty-one episodes change
outcome between them. That is not an absence of evidence; it is evidence that the
seating controller is not the variable.

CPU only. Reads `.npz` episode archives and the certifications written from them.

Usage::

    python scripts/report_seating_bay_factorial.py --report evidence/seating_bay_factorial_v1.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CARRIED = ROOT / "artifacts" / "robotcarried"
# `scripts/` is a directory rather than a package on the path, so a bare run needs
# the repository root on `sys.path` before the sibling import. `pyproject.toml`
# gives pytest the same thing through `pythonpath`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_paired_arms import mcnemar_exact, wilson_interval  # noqa: E402

#: Each cell: the archive stem, and what the two flags were set to.
CELLS: dict[str, dict[str, Any]] = {
    # The prescribed throat was run at both cohort sizes. Both are kept: the
    # 8-environment pair is what compares against the relieved cells episode for
    # episode, and the 32-environment pair is the one with enough discordant
    # episodes to make its null informative.
    "prescribed_guarded_8env": {
        "archives": "certify_insert_v33force_c11065_chain_guarded_seed*.npz",
        "throat_mm": 11.065,
        "relief_m": 0.0,
        "controller": "scripted guarded advance",
        "environments_per_seed": 8,
    },
    "prescribed_policy_8env": {
        "archives": "certify_insert_v33force_c11065_chain_policy_seed*.npz",
        "throat_mm": 11.065,
        "relief_m": 0.0,
        "controller": "learned force-feedback seating",
        "environments_per_seed": 8,
    },
    "prescribed_guarded_32env": {
        "archives": "certify_insert_v33force_c11065_chain_guarded_n96_seed*.npz",
        "throat_mm": 11.065,
        "relief_m": 0.0,
        "controller": "scripted guarded advance",
        "environments_per_seed": 32,
    },
    "prescribed_policy_32env": {
        "archives": "certify_insert_v33force_c11065_chain_policy_n96_seed*.npz",
        "throat_mm": 11.065,
        "relief_m": 0.0,
        "controller": "learned force-feedback seating",
        "environments_per_seed": 32,
    },
    "relieved_guarded_8env": {
        "archives": "certify_insert_v33force_relieved_chain_guarded_seed*.npz",
        "throat_mm": 15.678,
        "relief_m": 0.0046125,
        "controller": "scripted guarded advance",
        "environments_per_seed": 8,
    },
    "relieved_policy_8env": {
        "archives": "certify_insert_v33force_chain_policy_seed*.npz",
        "throat_mm": 15.678,
        "relief_m": 0.0046125,
        "controller": "learned force-feedback seating",
        "environments_per_seed": 8,
    },
}

#: Only cells with the same environment count are paired against each other. A
#: 32-environment cohort and an 8-environment one are not the same episodes, so
#: pairing them would be the mistake this whole file is about.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    (
        "prescribed_guarded_8env",
        "relieved_guarded_8env",
        "the bay, scripted controller held fixed",
    ),
    (
        "prescribed_policy_8env",
        "relieved_policy_8env",
        "the bay, learned controller held fixed",
    ),
    (
        "relieved_policy_8env",
        "relieved_guarded_8env",
        "the controller, in the bay the chain actually runs",
    ),
    (
        "prescribed_policy_8env",
        "prescribed_guarded_8env",
        "the controller, at the prescribed throat, 24 episodes",
    ),
    (
        "prescribed_policy_32env",
        "prescribed_guarded_32env",
        "the controller, at the prescribed throat, 96 episodes",
    ),
)


def _success(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(str(CARRIED / pattern)))
    if not files:
        raise SystemExit(f"no episode archives matched {pattern!r} under {CARRIED}")
    rows: list[np.ndarray] = []
    for name in files:
        archive = np.load(name)
        fields = [str(value) for value in archive["fields"]]
        table = np.asarray(archive["rows"], dtype=np.float64)
        rows.append(table[:, fields.index("success")] > 0.5)
    return np.concatenate(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None, help="Write the factorial as evidence JSON.")
    args = parser.parse_args()

    outcomes = {name: _success(cell["archives"]) for name, cell in CELLS.items()}

    cells: dict[str, Any] = {}
    print(f"{'cell':22s} {'throat':>8s} {'controller':32s} {'n':>4s} {'ok':>4s} {'rate':>8s}  Wilson 95%")
    for name, cell in CELLS.items():
        ok = int(outcomes[name].sum())
        n = int(outcomes[name].size)
        low, high = wilson_interval(ok, n)
        cells[name] = {
            "channel_throat_per_side_mm": cell["throat_mm"],
            "destination_channel_relief_m": cell["relief_m"],
            "seating_controller": cell["controller"],
            "environments_per_seed": cell["environments_per_seed"],
            "episodes": n,
            "successes": ok,
            "success_rate": round(ok / n, 6),
            "success_rate_wilson_95": {"low": round(low, 6), "high": round(high, 6)},
        }
        print(
            f"{name:22s} {cell['throat_mm']:7.3f}  {cell['controller']:32s} {n:4d} {ok:4d} "
            f"{ok / n:7.2%}  [{low:.3f}, {high:.3f}]"
        )

    print()
    comparisons: list[dict[str, Any]] = []
    for baseline, treatment, question in COMPARISONS:
        left, right = outcomes[baseline], outcomes[treatment]
        if left.size != right.size:
            raise SystemExit(
                f"{baseline} has {left.size} episodes and {treatment} has {right.size}; these are "
                "not the same cohort and must not be paired"
            )
        gained = int((~left & right).sum())
        lost = int((left & ~right).sum())
        mcnemar = mcnemar_exact(gained, lost)
        # **Directional, and named for the direction it tests.** A p-value that does
        # not know which arm won is worse than no p-value, and this project published
        # one once: `rack_prescription_paired_n192` reported 2.95e-06 for 28 gained
        # against 74 lost, which reads as a decisive improvement and is a decisive
        # loss. Two-sided is the default to quote; `improvement_p` is only meaningful
        # where the direction was named in advance, and one of the cells below is a
        # loss whose improvement_p is 1.0.
        improvement = float(mcnemar["improvement_p"])
        deterioration = float(mcnemar["deterioration_p"])
        two_sided = float(mcnemar["two_sided_p"])
        comparisons.append(
            {
                "what_changes": question,
                "baseline": baseline,
                "treatment": treatment,
                "episodes": int(left.size),
                "baseline_successes": int(left.sum()),
                "treatment_successes": int(right.sum()),
                "points": round(100.0 * (right.mean() - left.mean()), 3),
                "discordant_gained": gained,
                "discordant_lost": lost,
                "direction": mcnemar["direction"],
                "mcnemar_two_sided_p": round(two_sided, 6),
                "mcnemar_improvement_p": round(improvement, 6),
                "mcnemar_deterioration_p": round(deterioration, 6),
            }
        )
        print(
            f"{question:52s} {int(left.sum()):3d} -> {int(right.sum()):3d} of {left.size:3d}"
            f"   gained {gained:3d} lost {lost:3d}   {mcnemar['direction']:6s}"
            f"   two-sided p = {two_sided:.5f}   improvement p = {improvement:.5f}"
        )

    report = {
        "title": "Which matters more, the seating controller or the bay it seats into",
        "evidence_type": "paired_simulation_arms",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Two factors -- the seating controller and the destination channel's throat -- crossed on "
            "one checkpoint set, the same task, the same three held-out seeds and the same "
            "environments. Which one moves the chain?"
        ),
        "held_fixed": {
            "task": "Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflowForce-v0",
            "held_out_evaluation_seeds": [4070, 5070, 6070],
            "capture_checkpoint_sha256": "DB5B2887C8BB70D92D604220EFC6BC747695C8E18A2FB7F405F047D132CD975E",
            "extract_checkpoint_sha256": "ADC247AB0E9403761A3108228019AE7BE4DC408155B414B6BF0226102E5CCA54",
            "insert_checkpoint_sha256": "86599FC204C2FEE8A36E077BBA8F6C12AFDFB9C878D650E743C82FB074DADFBD",
            "insert_checkpoint": "grapple_insert_l0_seed70_v33force epoch 3000",
            "note": (
                "The insert weights are loaded in every cell and stepped only where the controller "
                "is the policy, which is how every guarded arm in this repository is run: the "
                "policy-set hash is then identical across the pair and the difference cannot be the "
                "checkpoint."
            ),
        },
        "cells": cells,
        "paired_comparisons": comparisons,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "The pairing assumes episode i of one arm started from the state episode i of the other "
            "did, which follows from identical seeds, identical checkpoints and deterministic "
            "resets. The tool cannot check it, so the claim of a fixed cohort travels with the "
            "number. Cells at different environment counts are never paired.",
            "Neither controller clears the unchanged 95% chain gate in any cell, and no gate moved.",
            "The 11.065 mm cells do not enable the destination rack's pawls, because "
            "run_robot_carried.sh certify does not; nor do the 15.678 mm cells. The published "
            "22/24 does, and is a different arm.",
            "A null between the two controllers at the prescribed throat is reported as an "
            "informative null because thirty-one of ninety-six episodes change outcome and they "
            "split evenly. It is not a claim that the two controllers are identical.",
            "Every p-value is named for the direction it tests. `improvement_p` is the probability "
            "of gaining at least as many episodes as were gained; for a comparison the treatment "
            "LOST it is near 1.0, and `deterioration_p` is the one to read. Two-sided is the "
            "default. A p-value that does not know which arm won is worse than no p-value, and this "
            "project published one before the implementation was corrected.",
            "The bay effect on the learned controller is the one cell pair with no significance at "
            "n = 24, and its direction is opposite to the scripted controller's. Direction, not "
            "rate.",
        ],
    }

    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
