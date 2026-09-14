"""Which condition the capture and extraction skills fail on, episode by episode.

Both skills sit just under the unchanged 95% gate -- capture at 86.90% and
extraction at 87.64% -- and both overlap the certificates they were meant to
beat, so neither retrain is an improvement. "Misses the gate by eight points" is
where that has stood. It is a weaker thing to be able to say than "misses the
gate because", and the episodes needed to say the second have been on disk the
whole time.

Each success predicate is a conjunction, and most of its terms are recorded per
episode. So a failure can be attributed rather than counted: take
the episodes the certification counted as failures and ask, for each, which
terms were false when the clock ran out.

**What this is and is not.** ``_freeze`` in ``run_workflow_demo.py`` stores an
episode's row at the moment of judgement, so these are the values the outcome was
decided on. That makes this a legitimate partition of the failures and an
illegitimate prediction of them: it says which condition was false, not which
condition would have gone false from some earlier state. Reading a
judgement-time value as a hand-over value is what retracted
``criterion_retention_v1.json``, and the distinction is the same one here.

Nothing is restated. The capture bounds are read out of the module that enforces
them, the two settling limits with them -- they are *derived* there, from the
capture tolerances over the chain's settling window, and that derivation is why
the criterion is consistent with the chain rather than convenient for the skill.
The extraction grip term calls ``grapple_geometry.grip_offset_admissible``, which
is the predicate ``pin_grip_intact`` runs, on the pin-axis columns the archive
already carries.

Three terms are not recoverable and are not attributed. The grip drive torque --
the signal that says the pads are *loaded* rather than merely placed -- is not an
episode column, and neither is the 0.30 s hold; both belong to each predicate.
Extraction's "clear of the slot" is not one either, because the axial column
measures the insertion goal rather than the extracted plane. An episode inside
every recorded bound is therefore reported as ``inside_every_recorded_bound``
rather than assigned to a term this reading cannot see.

CPU only. Reads ``.npz`` episode archives and the source; imports no simulator.

Usage::

    python scripts/report_skill_gate_attrition.py \\
        --capture "artifacts/certify_skills_c11065/grasp_v7m130_c11065_s*_seed*.npz" \\
        --extract "artifacts/certify_skills_c11065/extract_v18pin_c11065_s*_seed*.npz" \\
        --report evidence/skill_gate_attrition_v1.json
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from zero_g_blade_swap.grapple_geometry import (
    GRIP_MAX_APPROACH_BACKOUT_M,
    GRIP_MAX_APPROACH_FEED_M,
    GRIP_MAX_TRANSVERSE_M,
)

ROOT = Path(__file__).resolve().parents[1]
GRAPPLE = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "grapple.py"
STAGE_NAMES = {0: "near_stage_0", 1: "medium_stage_1", 2: "full_stage_2"}
#: ``evaluation.TERMINATION_REASONS`` order, which is what an archive stores.
REASONS = (
    "non_finite",
    "mount_unstable",
    "insertion_failed",
    "insertion_success",
    "time_out",
    "uncategorized",
    "excessive_contact_force",
    "capture_success",
    "capture_failed",
    "extraction_success",
    "extraction_failed",
)


def _module_scalars(path: Path) -> dict[str, float]:
    """Every module-level scalar the file assigns, literal or arithmetic.

    ``ast.literal_eval`` handles ``CAPTURE_POSITION_TOLERANCE_M = 0.020`` and not
    ``EXTRACTION_LINEAR_VELOCITY_LIMIT = 0.5 * CAPTURE_POSITION_TOLERANCE_M /
    WORKFLOW_SETTLE_S``, and the derived form is the one that matters most here.
    So names already resolved are substituted and the arithmetic is evaluated
    over them, one pass, in file order.
    """

    values: dict[str, float] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def evaluate(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            for operator, apply in (
                (ast.Mult, lambda a, b: a * b),
                (ast.Div, lambda a, b: a / b),
                (ast.Add, lambda a, b: a + b),
                (ast.Sub, lambda a, b: a - b),
            ):
                if isinstance(node.op, operator):
                    return apply(left, right)
        raise ValueError("not a scalar expression")

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = evaluate(node.value)
            except (ValueError, KeyError, ZeroDivisionError, TypeError):
                continue
    return values


def _load(patterns: list[str]) -> tuple[np.ndarray, dict[str, int], list[str]]:
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        raise SystemExit(f"no episode archives matched {patterns!r}")
    rows: list[np.ndarray] = []
    fields: list[str] | None = None
    for name in files:
        archive = np.load(name)
        names = [str(value) for value in archive["fields"]]
        if fields is not None and names != fields:
            raise SystemExit(f"{name} records different columns from the archives before it")
        fields = names
        rows.append(np.asarray(archive["rows"], dtype=np.float64))
    assert fields is not None
    return np.concatenate(rows), {name: index for index, name in enumerate(fields)}, files


def _reason_counts(codes: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code in sorted(set(codes.tolist())):
        index = int(code)
        name = REASONS[index] if 0 <= index < len(REASONS) else f"code_{index}"
        counts[name] = int((codes == code).sum())
    return counts


def _describe(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "p50": round(float(np.median(values)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "max": round(float(values.max()), 6),
    }


#: One term of a success predicate: a mask over rows saying where it is broken,
#: a one-line statement of the bound, and the column to quote when it is.
Term = tuple[Callable[[np.ndarray, dict[str, int]], np.ndarray], str, str | None]


def _over(column: str, bound: float) -> Term:
    def broken(rows: np.ndarray, index: dict[str, int]) -> np.ndarray:
        return rows[:, index[column]] > bound + 1.0e-9

    return broken, f"{column} <= {bound:.6g}", column


def _pin_grip_lost() -> Term:
    """The extraction predicate's grip term, as the pin's own geometry states it."""

    def broken(rows: np.ndarray, index: dict[str, int]) -> np.ndarray:
        closing = rows[:, index["grip_offset_closing_axis_m"]]
        third = rows[:, index["grip_offset_third_axis_m"]]
        approach = rows[:, index["grip_offset_approach_axis_m"]]
        transverse = np.hypot(closing, third)
        return ~(
            (transverse <= GRIP_MAX_TRANSVERSE_M + 1.0e-9)
            & (approach >= GRIP_MAX_APPROACH_FEED_M - 1.0e-9)
            & (approach <= GRIP_MAX_APPROACH_BACKOUT_M + 1.0e-9)
        )

    statement = (
        f"transverse <= {GRIP_MAX_TRANSVERSE_M:.6g} and "
        f"{GRIP_MAX_APPROACH_FEED_M:.6g} <= approach <= {GRIP_MAX_APPROACH_BACKOUT_M:.6g}, "
        "on the pin's own axes"
    )
    return broken, statement, None


def _attribute(rows: np.ndarray, index: dict[str, int], terms: dict[str, Term]) -> dict[str, Any]:
    """Partition failures by which of the predicate's recorded terms are false.

    A failing episode can break several terms at once, and for extraction most
    do, so three readings are reported and only the second and third are
    exclusive: how often each term is broken at all, how often it is the only one
    broken, and the exact combination of broken terms, counted.
    """

    success = rows[:, index["success"]] > 0.5
    failed = rows[~success]
    broken = {name: term[0](failed, index) for name, term in terms.items()}
    count = np.zeros(len(failed), dtype=int)
    for mask in broken.values():
        count += mask.astype(int)

    per_term: dict[str, Any] = {}
    for name, (_, statement, column) in terms.items():
        mask = broken[name]
        entry: dict[str, Any] = {
            "bound": statement,
            "broken_by": int(mask.sum()),
            "the_only_term_broken": int((mask & (count == 1)).sum()),
        }
        if column is not None:
            entry["on_the_failures_that_break_it"] = _describe(failed[mask, index[column]])
            entry["on_the_successes"] = _describe(rows[success][:, index[column]])
        per_term[name] = entry

    combinations: dict[str, int] = {}
    names = list(terms)
    for row in range(len(failed)):
        label = " + ".join(name for name in names if broken[name][row]) or "inside every recorded bound"
        combinations[label] = combinations.get(label, 0) + 1

    return {
        "episodes": int(len(rows)),
        "successes": int(success.sum()),
        "failures": int(len(failed)),
        "terms": per_term,
        "exact_combination_of_broken_terms": dict(
            sorted(combinations.items(), key=lambda item: -item[1])
        ),
        # The predicate has terms no archive records -- the grip drive torque and
        # the 0.30 s hold -- so an episode inside every bound above is not
        # unexplained. It is attributed to those.
        "inside_every_recorded_bound": int((count == 0).sum()),
        "break_more_than_one_term": int((count > 1).sum()),
    }


def _by_stage(rows: np.ndarray, index: dict[str, int], terms: dict[str, Term]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    stages = rows[:, index["curriculum_stage"]]
    for stage in sorted(set(stages.tolist())):
        name = STAGE_NAMES.get(int(stage), f"stage_{int(stage)}")
        out[name] = _attribute(rows[stages == stage], index, terms)
    return out


def _skill(label: str, patterns: list[str], terms: dict[str, Term], gate: float) -> dict[str, Any]:
    rows, index, files = _load(patterns)
    pooled = _attribute(rows, index, terms)
    rate = pooled["successes"] / pooled["episodes"]
    return {
        "skill": label,
        "archives": [Path(name).name for name in files],
        "gate": gate,
        "success_rate": round(rate, 6),
        "points_below_the_gate": round(100.0 * (gate - rate), 3),
        "termination_reasons": _reason_counts(rows[:, index["termination_reason"]]),
        "pooled": pooled,
        "by_curriculum_stage": _by_stage(rows, index, terms),
    }


def _print(block: dict[str, Any]) -> None:
    pooled = block["pooled"]
    print(
        f"== {block['skill']}: {pooled['successes']}/{pooled['episodes']} = "
        f"{block['success_rate']:.4f}, {block['points_below_the_gate']:.2f} points below "
        f"{block['gate']:.2f} -- {pooled['failures']} failures"
    )
    for name, term in pooled["terms"].items():
        print(
            f"     {name:18s} broken by {term['broken_by']:5d}"
            f"   only term broken {term['the_only_term_broken']:5d}"
            f"   [{term['bound']}]"
        )
    print("   the exact combinations:")
    for label, total in pooled["exact_combination_of_broken_terms"].items():
        print(f"     {total:5d}  {100.0 * total / max(1, pooled['failures']):5.1f}%  {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", nargs="+", required=True, help="Capture certification .npz archives or globs.")
    parser.add_argument("--extract", nargs="+", required=True, help="Extraction certification .npz archives or globs.")
    parser.add_argument("--gate", type=float, default=0.95, help="The unchanged promotion gate both skills are held to.")
    parser.add_argument("--report", type=Path, default=None, help="Write the partition as evidence JSON.")
    args = parser.parse_args()

    source = _module_scalars(GRAPPLE)
    # Capture: the pads loaded on the pin, inside the grip position tolerance the
    # *chain* demands, at the capture attitude, held for 0.30 s. The position
    # bound is ``WORKFLOW_HANDOVER_GRIP_M`` and not ``capture_established``'s
    # looser 20 mm, because ``capture_success_mask`` passes the tighter one -- a
    # skill certified more loosely than the workflow it belongs to passes alone
    # and stalls in the chain.
    capture_terms: dict[str, Term] = {
        "grip_position": _over("tool_to_handle_error_m", source["WORKFLOW_HANDOVER_GRIP_M"]),
        "grip_attitude": _over("tool_to_handle_orientation_rad", source["CAPTURE_ATTITUDE_TOLERANCE_RAD"]),
    }
    # Extraction: clear of the slot, still gripped *on the pin*, and settled.
    extract_terms: dict[str, Term] = {
        "pin_grip_lost": _pin_grip_lost(),
        "grip_attitude": _over("tool_to_handle_orientation_rad", source["CAPTURE_ATTITUDE_TOLERANCE_RAD"]),
        "linear_settling": _over("blade_linear_velocity_mps", source["EXTRACTION_LINEAR_VELOCITY_LIMIT"]),
        "angular_settling": _over("blade_angular_velocity_radps", source["EXTRACTION_ANGULAR_VELOCITY_LIMIT"]),
    }

    report = {
        "title": "Which condition the capture and extraction skills fail on",
        "evidence_type": "second_reading_of_existing_episodes",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Capture certifies at 86.90% and extraction at 87.64% against an unchanged 95% gate. "
            "Each predicate is a conjunction whose terms are episode columns, so which term is "
            "false on the episodes that fail?"
        ),
        "tolerances_read_from": [
            "src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py",
            "src/zero_g_blade_swap/grapple_geometry.py",
        ],
        "tolerances": {
            "WORKFLOW_HANDOVER_GRIP_M": source["WORKFLOW_HANDOVER_GRIP_M"],
            "CAPTURE_POSITION_TOLERANCE_M": source["CAPTURE_POSITION_TOLERANCE_M"],
            "CAPTURE_ATTITUDE_TOLERANCE_RAD": source["CAPTURE_ATTITUDE_TOLERANCE_RAD"],
            "WORKFLOW_SETTLE_S": source["WORKFLOW_SETTLE_S"],
            "EXTRACTION_LINEAR_VELOCITY_LIMIT": round(source["EXTRACTION_LINEAR_VELOCITY_LIMIT"], 6),
            "EXTRACTION_ANGULAR_VELOCITY_LIMIT": round(source["EXTRACTION_ANGULAR_VELOCITY_LIMIT"], 6),
            "GRIP_MAX_TRANSVERSE_M": GRIP_MAX_TRANSVERSE_M,
            "GRIP_MAX_APPROACH_FEED_M": GRIP_MAX_APPROACH_FEED_M,
            "GRIP_MAX_APPROACH_BACKOUT_M": GRIP_MAX_APPROACH_BACKOUT_M,
        },
        "capture": _skill("capture", args.capture, capture_terms, args.gate),
        "extraction": _skill("extraction", args.extract, extract_terms, args.gate),
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "A second reading of episodes that were already certified. No simulator was run and no "
            "rate moves: the pooled counts here reproduce the certificates they come from.",
            "Every value is the state the episode was JUDGED in, because that is when the row is "
            "frozen. This partitions the failures; it does not predict them from an earlier state, "
            "and reading it as prediction is the error that retracted criterion_retention_v1.json.",
            "Three terms are not episode columns and are not attributed. The grip drive torque -- "
            "the signal that the pads are loaded rather than placed -- and the 0.30 s hold belong to "
            "both predicates; 'clear of the slot' belongs to extraction, whose axial column measures "
            "the insertion goal rather than the extracted plane. Episodes inside every recorded bound "
            "are reported as such rather than assigned to a term this reading cannot see.",
            "A failing episode may break several terms at once, so only 'the_only_term_broken' and "
            "'exact_combination_of_broken_terms' are exclusive counts. 'broken_by' is not.",
            "Attribution is not a cause. It says which condition the episode was judged against and "
            "failed, which is a stronger statement than a rate and a weaker one than a mechanism.",
        ],
    }

    _print(report["capture"])
    print()
    _print(report["extraction"])

    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
