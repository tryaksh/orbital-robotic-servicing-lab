"""Which seating controller does the chain keep, decided arithmetically.

The chain's seating phase is a scripted guarded advance. A learned insert policy
replaces it only if it wins, and "wins" has to mean something narrower than a
better headline: **the same three held-out seeds, the same rack, the same
capture and extraction checkpoints, and only the seating controller different.**
Anything else and a geometry change is being quoted as a policy change, which is
the mistake this repository has paid for most.

So this reads the two certification reports and states the outcome rather than
leaving it to be eyeballed. It refuses to compare two runs whose protocols
differ on the seeds, the environments per run, the episode count or the task,
because a comparison across those is not a comparison.

The decision it prints is the one ``docs/NOW.md`` records:

* the policy takes the seating phase only if it wins pooled **and** on every
  seed, since a controller that is better on average and worse on one seed is a
  controller with a failure mode nobody has looked at;
* otherwise the guarded advance keeps it, and both arms are published.

CPU only.

Usage::

    python scripts/report_seating_head_to_head.py \
        --guarded evidence/workflow_robot_carried_m130pin_guarded_c11065_certification.json \
        --policy  evidence/workflow_robot_carried_insert_v24rack_chain_policy_certification.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Protocol fields that must match, or the two runs are not the same experiment.
COMPARABLE = ("tasks", "held_out_evaluation_seeds", "curriculum_stages", "environments_per_run", "runs")


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing certification report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _per_seed(report: dict) -> dict[str, float]:
    return {
        str(seed): float(block["success_rate"])
        for seed, block in report.get("by_evaluation_seed", {}).items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guarded", type=Path, required=True, help="The scripted guarded advance's report.")
    parser.add_argument("--policy", type=Path, required=True, help="The learned seating policy's report.")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "evidence" / "seating_controller_head_to_head.json",
    )
    args = parser.parse_args()

    guarded, policy = _load(args.guarded), _load(args.policy)
    mismatches = [
        field
        for field in COMPARABLE
        if guarded.get("protocol", {}).get(field) != policy.get("protocol", {}).get(field)
    ]

    guarded_seeds, policy_seeds = _per_seed(guarded), _per_seed(policy)
    shared = sorted(set(guarded_seeds) & set(policy_seeds))
    guarded_pooled = float(guarded["overall"]["success_rate"])
    policy_pooled = float(policy["overall"]["success_rate"])
    wins_pooled = policy_pooled > guarded_pooled
    wins_every_seed = bool(shared) and all(policy_seeds[s] >= guarded_seeds[s] for s in shared)
    comparable = not mismatches

    decision = (
        "the learned insert policy takes the chain's seating phase"
        if comparable and wins_pooled and wins_every_seed
        else "the scripted guarded advance keeps the chain's seating phase"
    )

    report = {
        "title": "Which controller seats the module: the guarded advance or the learned policy",
        "evidence_type": "simulation_only",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "On the same rack, the same three held-out seeds and the same capture and extraction "
            "checkpoints, does the learned insert policy seat better than the scripted guarded "
            "advance?"
        ),
        "guarded_report": str(args.guarded.as_posix()),
        "policy_report": str(args.policy.as_posix()),
        "protocols_are_comparable": comparable,
        "protocol_mismatches": mismatches,
        "guarded_pooled": round(guarded_pooled, 6),
        "policy_pooled": round(policy_pooled, 6),
        "guarded_by_seed": {seed: round(rate, 6) for seed, rate in sorted(guarded_seeds.items())},
        "policy_by_seed": {seed: round(rate, 6) for seed, rate in sorted(policy_seeds.items())},
        "policy_wins_pooled": bool(wins_pooled),
        "policy_wins_every_seed": bool(wins_every_seed),
        "decision": decision,
        "decision_rule": (
            "the policy takes the seating phase only if it wins pooled AND on every shared seed; "
            "a controller better on average and worse on one seed has a failure mode nobody has "
            "looked at, and the guarded advance is the one with a published interval"
        ),
        "both_arms_are_published": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'':<12}{'pooled':>9}" + "".join(f"{seed:>9}" for seed in shared))
    print(f"{'guarded':<12}{guarded_pooled * 100:>8.2f}%" + "".join(f"{guarded_seeds[s] * 100:>8.2f}%" for s in shared))
    print(f"{'policy':<12}{policy_pooled * 100:>8.2f}%" + "".join(f"{policy_seeds[s] * 100:>8.2f}%" for s in shared))
    if mismatches:
        print(f"NOT COMPARABLE: protocols differ on {', '.join(mismatches)}")
    print(f"DECISION: {decision}")
    print(f"wrote {args.report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
