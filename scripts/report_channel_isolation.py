"""Which observation channel costs extraction its transfer, on an unchanged checkpoint.

Thirteen of the twenty-four camera-driven chain episodes time out in extraction
and never engage the form lock. Two channels could be responsible and they are
separable without training anything and without rendering a frame:

* the **pose** channels -- the tool-to-grip error and the remaining travel --
  carry the estimator's residual;
* the **velocity** channel carries what the estimator *manufactures*. The camera
  runs at half the control rate, so the pose is a staircase and a differenced
  estimate is zero on one control step and a full jump on the next. Measured
  with the arm held still, that channel reads 17.02 mm/s at the deployed filter
  against 3.38 for the identical differencing on the simulator's own pose, on a
  quantity whose seated signal is 0.69 mm/s.

Four arms, one checkpoint, one curriculum stage, three held-out evaluation
seeds. The certified v18pin weights are unchanged in all four; only which
channels read the surrogate differs. So the difference between the arms is the
channel and cannot be a different policy.

    exact state        the published task, and the control
    pose only          pose channels noised, velocity exact
    velocity only      velocity noised, pose channels exact
    both               the full noised task the fine-tune trains on

If the two single-channel arms do not sum to the full arm's loss, the channels
interact, and that is worth knowing too -- a policy servoing on a noisy pose with
a noisy velocity is not the same problem as either alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

ARMS: tuple[tuple[str, str, str], ...] = (
    ("Extract", "exact state", "the published task; the control"),
    ("ExtractPoseNoised", "pose only", "grip error and remaining travel from the surrogate; velocity exact"),
    ("ExtractVelocityNoised", "velocity only", "velocity from the surrogate; every pose channel exact"),
    ("ExtractNoised", "both", "the full noised task the fine-tune trains on"),
)


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_dir", type=Path, default=ROOT / "evidence")
    parser.add_argument("--prefix", default="extract_channel_isolation_")
    parser.add_argument("--report", type=Path, default=ROOT / "evidence" / "extract_channel_attribution_v1.json")
    arguments = parser.parse_args()

    revision = git_source_revision(ROOT)
    if revision.get("available") and revision.get("dirty"):
        raise SystemExit("refusing to write evidence from a dirty tracked worktree; commit first")

    rows = []
    for task, label, note in ARMS:
        path = arguments.evidence_dir / f"{arguments.prefix}{task}.json"
        if not path.is_file():
            raise SystemExit(f"missing arm: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        overall = report["overall"]
        successes = int(overall["successes"])
        episodes = int(overall.get("episodes") or overall.get("trials") or 0)
        low, high = _wilson(successes, episodes)
        rows.append(
            {
                "task": task,
                "arm": label,
                "what_differs": note,
                "episodes": episodes,
                "successes": successes,
                "success_rate": round(successes / episodes, 6) if episodes else None,
                "wilson_95": {"low": round(low, 6), "high": round(high, 6)},
                "source": path.name,
            }
        )

    control = rows[0]["success_rate"]
    for row in rows:
        row["points_below_the_control"] = (
            None if control is None or row["success_rate"] is None else round(100.0 * (control - row["success_rate"]), 2)
        )

    pose = rows[1]["points_below_the_control"]
    velocity = rows[2]["points_below_the_control"]
    both = rows[3]["points_below_the_control"]
    interaction = None if None in (pose, velocity, both) else round(both - pose - velocity, 2)

    report = {
        "title": "Which observation channel costs extraction its transfer",
        "evidence_type": "channel_attribution_on_an_unchanged_checkpoint",
        "what_this_is": (
            "One checkpoint, one curriculum stage, three held-out evaluation seeds, four arms. The "
            "certified v18pin weights are unchanged in all four; only which observation channels read "
            "the training-time estimator surrogate differs, so the difference between arms is the "
            "channel and cannot be a different policy."
        ),
        "scope": [
            "Simulation only. No result here was produced on real hardware.",
            "The surrogate reproduces the estimator's certified residual, sample-and-hold and miss rate. It does not reproduce occlusion or the in-loop tail, so these are lower bounds on what the cameras cost.",
            "Curriculum stage 0 only: that is the station the chain hands extraction over at.",
        ],
        "arms": rows,
        "attribution_points": {
            "pose_channels_alone": pose,
            "velocity_channel_alone": velocity,
            "both_together": both,
            "interaction": interaction,
            "reading": (
                "A positive interaction means the two channels cost more together than apart, which is "
                "what servoing on a noisy pose with a noisy velocity would predict; a negative one means "
                "one channel's damage is already done by the other."
            ),
        },
        "source_revision": revision,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'arm':<16} {'episodes':>9} {'success':>9} {'wilson 95':>18} {'vs control':>11}")
    for row in rows:
        interval = row["wilson_95"]
        print(
            f"{row['arm']:<16} {row['episodes']:>9} {row['success_rate']:>9.4f} "
            f"  [{interval['low'] * 100:5.1f}, {interval['high'] * 100:5.1f}] "
            f"{row['points_below_the_control']:>11.2f}"
        )
    print(f"\npose alone {pose}   velocity alone {velocity}   both {both}   interaction {interaction}")
    print(f"wrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
