#!/usr/bin/env python3
"""Will my controller handle this change, where will it fail, and what should I measure?

The engineer-facing entry point. Give it the configurations you have already
run and a change you are considering, and it returns a phase-level prediction --
or says why it cannot.

Nothing about this workcell is built in. The corpus is a JSON file naming the
cohorts you have, the dimensions each one was run at, and where its episodes
live. An engineer with their own cohorts describes them the same way and gets
their own answer.

    {
      "phase_names": ["capture", "seat", "extract", "transit", "insert", "done"],
      "criterion_m": 0.0025,
      "reference": {
        "label": "nominal 130x20",
        "episodes": "artifacts/jam_mechanism/reference/seed*/nominal.npz",
        "dimensions": {"width": 0.130, "thickness": 0.020, "lateral_clearance": 0.011065}
      },
      "cohorts": [ ... same shape ... ],
      "change": {"width": 0.140, "thickness": 0.020, "lateral_clearance": 0.006065}
    }

**Read the refusals first.** The procedure declines when no cohort varied the
dimension you are proposing, and when cohorts varied it only alongside another
so the response cannot be attributed between them. The second is the common
case, because campaigns vary a part rather than a dimension, and it is the more
useful message: it names the experiment that would let the question be answered.

    python scripts/predict_configuration_change.py --corpus configs/rack_corpus.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from handoff_qualification.change_prediction import (  # noqa: E402
    Cohort,
    predict,
    sensitivity_corpus,
)
from handoff_qualification.records import load_cohort  # noqa: E402
from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402


def failing_phase(row, column, phase_names: list[str]) -> str | None:
    """Which phase an episode failed in, for episodes that did not time out too.

    A timeout names its own phase. But the failure this workcell's prescribed
    clearance produces does **not** time out: the module wedges, the insertion
    predicate never fires, and the episode ends having reached the insert phase
    with `timed_out_in_phase = -1`. Attributing on the timeout alone reports
    those twelve episodes as no phase at all, which is how the first run of this
    tool predicted "insert unchanged" against an arm that had twelve insert
    failures in it.

    So: a timeout names its phase; otherwise a failure whose success predicate
    never fired is attributed to the phase it reached, and a failure whose
    predicate *did* fire is a terminal-gate failure after seating.
    """

    timed_out = int(row[column["timed_out_in_phase"]]) if "timed_out_in_phase" in column else -1
    if 0 <= timed_out < len(phase_names):
        return phase_names[timed_out]
    if "success" in column and row[column["success"]] > 0.5:
        return None
    if "predicate_fired" in column and row[column["predicate_fired"]] <= 0.5:
        reached = int(row[column["reached_phase"]]) if "reached_phase" in column else -1
        if 0 <= reached < len(phase_names):
            return phase_names[reached]
        return None
    return "terminal_gate"


def build_cohort(spec: dict, phase_names: list[str]) -> Cohort:
    """Load one cohort's episodes and the phase each failure occurred in."""

    paths = sorted(glob.glob(str(ROOT / spec["episodes"])))
    if not paths:
        raise SystemExit(f"no episodes matched {spec['episodes']} for {spec['label']}")
    records = []
    phases: list[str | None] = []
    for path in paths:
        records.extend(load_cohort(Path(path)))
        archive = np.load(path, allow_pickle=True)
        fields = [str(n) for n in archive["fields"]]
        rows = archive["rows"].astype(float)
        column = {name: index for index, name in enumerate(fields)}
        for row in rows:
            phases.append(failing_phase(row, column, phase_names))
    return Cohort(
        label=spec["label"],
        dimensions=dict(spec["dimensions"]),
        records=records,
        timed_out_phase=phases,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.corpus.read_text(encoding="utf-8"))
    phase_names = spec.get(
        "phase_names", ["capture", "seat", "extract", "transit", "insert", "done"]
    )
    criterion = float(spec.get("criterion_m", 0.0025))
    reference = build_cohort(spec["reference"], phase_names)
    cohorts = [build_cohort(entry, phase_names) for entry in spec["cohorts"]]
    change = dict(spec["change"])

    dimensions = sorted(
        set(reference.dimensions) | {d for c in cohorts for d in c.dimensions}
    )
    corpus = sensitivity_corpus(reference, cohorts, dimensions)

    print(f"reference: {reference.label}  ({len(reference.records)} episodes)")
    print(
        f"   delivery {reference.delivery_rate():.4f}   "
        f"precision given delivery {reference.precision_given_delivery(criterion):.4f}"
    )
    print()
    print("what the corpus can speak to:")
    for dimension, info in corpus.items():
        state = "usable" if info["usable"] else "CONFOUNDED"
        alone = info["cohorts_varying_it_alone"] or "-"
        print(f"   {dimension:>20} {state:>11}   varied alone by {alone}")
        if not info["usable"] and info["cohorts_varying_it"]:
            print(f"   {'':>20} {'':>11}   only ever with {info['confounded_with']}")

    print()
    proposed = {
        d: v for d, v in change.items() if reference.dimensions.get(d) not in (None, v)
    }
    print(f"proposed change: {proposed or 'nothing moves'}")
    print()
    phases = tuple(phase_names[:-1]) + ("terminal_gate",)
    predictions = predict(reference, cohorts, change, criterion, phases)
    answered = [p for p in predictions.values() if p.answered]
    print(f"{'phase':>12} {'verdict':>12} {'rate':>8} {'reference':>10} {'span':>18}")
    for phase, prediction in predictions.items():
        if prediction.answered:
            low, high = prediction.rate_interval
            print(
                f"{phase:>12} {prediction.direction:>12} {prediction.rate:>8.4f} "
                f"{prediction.reference_rate:>10.4f} {f'[{low:.4f}, {high:.4f}]':>18}"
            )
        else:
            print(f"{phase:>12} {'refused':>12}   {prediction.refused}")

    if not answered:
        print()
        print("The procedure declines this change. That is an answer: the evidence to")
        print("support it does not exist yet, and the refusal says which experiment")
        print("would supply it.")

    document = {
        "title": "Predicted phase response to a proposed configuration change",
        "evidence_type": "prediction_from_existing_episodes",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": git_source_revision(ROOT),
        "corpus_file": args.corpus.as_posix(),
        "criterion_m": criterion,
        "reference": {
            "label": reference.label,
            "episodes": len(reference.records),
            "dimensions": reference.dimensions,
            "delivery_rate": reference.delivery_rate(),
            "precision_given_delivery": reference.precision_given_delivery(criterion),
        },
        "corpus_coverage": corpus,
        "proposed_change": change,
        "predictions": {
            phase: {
                "direction": p.direction,
                "rate": p.rate,
                "rate_interval": list(p.rate_interval) if p.rate_interval else None,
                "reference_rate": p.reference_rate,
                "refused": p.refused,
                "driven_by": list(p.driven_by),
                "confounded_with": list(p.confounded_with),
            }
            for phase, p in predictions.items()
        },
        "scope_and_limitations": [
            "A prediction, made before the proposed configuration was run. If any "
            "episode of it already exists, this is a diagnosis and must be labelled "
            "as one.",
            "The projection is linear from the cohorts that varied a dimension alone. "
            "That is the least the evidence allows and not a model of the physics; "
            "the interval is wide because the evidence is thin.",
            "The procedure is only as wide as its corpus. A dimension no cohort "
            "varied, or varied only alongside another, gets a refusal rather than a "
            "number.",
            "Phase names and the success criterion come from the corpus file. Nothing "
            "here is specific to a workcell.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
