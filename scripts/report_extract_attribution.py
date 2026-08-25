"""One table: what each change to the extract task was worth, on one policy.

Four things changed under this skill in one session -- the grip criterion, the
rack's lateral clearance, the reset's bound, and then the policy. Quoting the
last number against the first would credit the policy with all four, so every
step was run on the **unchanged** checkpoint first and only the last row is a
different policy.

Reads the ``.npz`` rows the runs wrote; writes ``evidence/`` JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zero_g_blade_swap.evaluation import wilson_interval  # noqa: E402

#: label -> (path, what was different, what was the same)
LADDER = (
    (
        "as_certified",
        "artifacts/diag/x_v17_s0_new.npz",
        "artifacts/diag/x_v17_s2_new.npz",
        "extract v17m130 exactly as it was certified: isotropic 30 mm grip ball, "
        "the rack's inherited 15.750 mm lateral clearance, the unbounded joint-noise reset",
    ),
    (
        "pin_resolved_criterion",
        "artifacts/diag/crit_pin_s0.npz",
        "artifacts/diag/crit_pin_s2.npz",
        "same policy, same rack, same reset; the grip is judged on the pin's own axes "
        "instead of as a distance from its drawing pose",
    ),
    (
        "derived_rack_clearance",
        "artifacts/diag/rack_v17_s0.npz",
        "artifacts/diag/rack_v17_s2.npz",
        "same policy, same reset; the channel's lateral clearance is derived from what the "
        "pads can follow rather than inherited from a 160 mm module, 15.750 -> 12.689 mm",
    ),
    (
        "bounded_reset",
        "artifacts/extract_pin/control_v17_s0.npz",
        "artifacts/extract_pin/control_v17_s2.npz",
        "same policy; the reset's joint noise is scaled so the grip error it induces stays "
        "inside the one the chain requires before it hands a module over",
    ),
    (
        "retrained",
        "artifacts/certify_skills/extract_v18pin_s0_seed1070.npz",
        "artifacts/certify_skills/extract_v18pin_s2_seed1070.npz",
        "the policy, retrained against all of the above",
    ),
)


def _rate(path: Path) -> tuple[float, int]:
    data = np.load(path, allow_pickle=True)
    rows = data["rows"]
    fields = [str(name) for name in data["fields"]]
    success = rows[:, fields.index("success")] > 0.5
    return float(success.mean()), int(len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "evidence" / "extract_attribution.json")
    arguments = parser.parse_args()

    steps: list[dict[str, object]] = []
    print(f"{'step':>26} {'stage 0':>10} {'stage 2':>10}  what changed")
    for label, stage0, stage2, note in LADDER:
        entry: dict[str, object] = {"step": label, "what_changed": note}
        rates: dict[str, float] = {}
        for name, relative in (("stage_0", stage0), ("stage_2", stage2)):
            path = PROJECT_ROOT / relative
            if not path.exists():
                entry[name] = None
                continue
            rate, episodes = _rate(path)
            low, high = wilson_interval(int(round(rate * episodes)), episodes)
            rates[name] = rate
            entry[name] = {
                "success_rate": round(rate, 6),
                "episodes": episodes,
                "wilson_95": {"low": round(low, 6), "high": round(high, 6)},
                "rows": relative,
            }
        steps.append(entry)
        print(
            f"{label:>26} {rates.get('stage_0', float('nan')) * 100:9.2f}% "
            f"{rates.get('stage_2', float('nan')) * 100:9.2f}%  {note[:44]}"
        )

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(
            {
                "title": "What each change to the extract task was worth, before any retraining",
                "evidence_type": "simulation_only",
                "generated_utc": datetime.now(UTC).isoformat(),
                "question": (
                    "Extract certified at 74.27% and 900 epochs of fine-tuning moved it 1.4 points. "
                    "How much of what follows is a better ruler, how much is a better rack, and how "
                    "much is a better policy?"
                ),
                "method": (
                    "One held-out seed, 512 episodes a point, the same checkpoint throughout. Each row "
                    "changes one thing and keeps everything before it."
                ),
                "policy": "grapple_extract_l0_seed70_v17m130, ep 10600, unchanged in every row but the last",
                "steps": steps,
                "scope_and_limitations": (
                    "One evaluation seed and two of the three curriculum stages, because this table is "
                    "an attribution rather than a certification. The pooled three-stage, three-seed "
                    "numbers are in the grapple_extract_* reports."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
