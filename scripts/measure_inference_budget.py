"""What the autonomy stack costs to run, per control step.

Nobody can put this on a vehicle without knowing whether it closes a 30 Hz loop
on hardware they can fly. This measures the two things that would run onboard —
the perception head and a skill policy — separately, on the GPU and on the CPU,
and states the result as a fraction of the 33.3 ms control period the whole
project is built around.

The CPU number is the interesting one. Flight processors are not RTX parts, and
a stack that needs a discrete GPU to close its loop is a different engineering
proposition from one that does not.

What this is not: an end-to-end latency figure. Sensor readout, image transport
and actuator command all cost time this does not measure, and on real hardware
they usually dominate. This is the compute budget of the learned components,
which is the part the project actually owns.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zero_g_blade_swap.pose_head import ModulePoseHead  # noqa: E402

#: The control period every policy here was trained and certified at.
CONTROL_PERIOD_MS = 1_000.0 / 30.0


class SkillPolicy(torch.nn.Module):
    """The actor shape the RL-Games checkpoints carry: three hidden layers.

    Rebuilt here rather than loaded, because the timing depends on the shape and
    not on the weights, and this script must run without RL-Games or a
    simulator.
    """

    def __init__(self, observation_dim: int = 57, action_dim: int = 7) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def _time(module: torch.nn.Module, sample: torch.Tensor, device: str, repeats: int) -> dict[str, float]:
    module = module.to(device).eval()
    sample = sample.to(device)
    synchronize = device.startswith("cuda")
    with torch.inference_mode():
        for _ in range(20):  # warm up kernels and any autotuning
            module(sample)
        if synchronize:
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            module(sample)
            if synchronize:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1_000.0)
    timings.sort()
    return {
        "mean_ms": statistics.fmean(timings),
        "p50_ms": timings[len(timings) // 2],
        "p99_ms": timings[min(len(timings) - 1, int(0.99 * len(timings)))],
        "max_ms": timings[-1],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--report", type=Path, default=Path("evidence/inference_budget.json"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    parameters = {
        "pose_head": sum(p.numel() for p in ModulePoseHead().parameters()),
        "skill_policy": sum(p.numel() for p in SkillPolicy().parameters()),
    }
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

    results: dict[str, dict[str, dict[str, float]]] = {}
    for device in devices:
        head = _time(ModulePoseHead(), torch.rand(1, 64, 64, 3), device, args.repeats)
        policy = _time(SkillPolicy(), torch.rand(1, 57), device, args.repeats)
        combined_mean = head["mean_ms"] + policy["mean_ms"]
        combined_p99 = head["p99_ms"] + policy["p99_ms"]
        results[device] = {
            "pose_head": head,
            "skill_policy": policy,
            "combined": {
                "mean_ms": combined_mean,
                "p99_ms": combined_p99,
                "fraction_of_control_period_mean": combined_mean / CONTROL_PERIOD_MS,
                "fraction_of_control_period_p99": combined_p99 / CONTROL_PERIOD_MS,
                "closes_the_loop": combined_p99 < CONTROL_PERIOD_MS,
            },
        }
        print(
            f"[INFO] {device:8s} head {head['mean_ms']:.3f} ms + policy {policy['mean_ms']:.3f} ms "
            f"= {combined_mean:.3f} ms ({100 * combined_mean / CONTROL_PERIOD_MS:.1f}% of the period)"
        )

    report = {
        "status": "measured",
        "title": "Onboard compute budget of the learned components",
        "evidence_type": "simulation_compute_characterization",
        "protocol": {
            "control_period_ms": CONTROL_PERIOD_MS,
            "control_rate_hz": 30,
            "batch_size": 1,
            "repeats": args.repeats,
            "parameters": parameters,
            "host": {"platform": platform.platform(), "processor": platform.processor()},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        },
        "by_device": results,
        "interpretation": (
            "Batch size one, because a flight system runs one arm. The CPU figure is the one that "
            "matters for a vehicle: a stack that needs a discrete GPU to close a 30 Hz loop is a "
            "different proposition from one that does not."
        ),
        "scope_and_limitations": [
            "Learned components only. Sensor readout, image transport, the IK solve and actuator "
            "command are not included and on real hardware usually dominate.",
            "Desktop hardware under a desktop OS, not a flight processor and not a real-time kernel.",
            "Inference only. No memory-bandwidth contention from anything else running onboard.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {args.report}")


if __name__ == "__main__":
    main()
