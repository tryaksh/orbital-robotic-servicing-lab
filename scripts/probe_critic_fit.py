"""Isolate value clipping on frozen on-policy observations; no simulator training."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import sha256, write_json  # noqa: E402
from assembly_recovery.study_ppo import StudyPolicy  # noqa: E402
from scripts.run_experiment import git, snapshot_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / "artifacts/assembly/uniform_completion_170_v3_r01/pilot/partial_cohort_13.pt"
    trajectory = ROOT / "artifacts/assembly/partial_checkpoint_diagnosis_v3_r02/stochastic/diagnosis/trajectory.pt"
    specification = {"rounds": 100, "mini_epochs": 4, "minibatch": 512, "learning_rate": .0001,
                     "gradient_norm": 1., "seed": 410070, "gamma": .995,
                     "arms": {"clipped_0_2": .2, "unclipped": None}, "wall_limit_s": 180,
                     "target": "Monte Carlo finite-job return-to-go on every active stochastic diagnostic row.",
                     "comparison": "Same starting critic, frozen normalized inputs, targets, minibatch orders, fresh Adam and update budget; only value clipping differs.",
                     "scope": "CPU regression diagnostic, not PPO training or an estimate of generalization; fitted models cannot initialize a policy run."}
    report = {"status": "starting", "research_result": False, "started_at_utc": datetime.now(UTC).isoformat(),
              "specification": specification, "checkpoint_sha256": sha256(checkpoint), "trajectory_sha256": sha256(trajectory),
              "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"), "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
              "source_hashes": snapshot_source(args.output / "source.zip"), "python": sys.version, "torch": torch.__version__,
              "platform": platform.platform(), "simulator_transitions": 0, "arms": {}}
    write_json(args.output / "registration.json", report)
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        data = torch.load(trajectory, map_location="cpu", weights_only=True)
        spec = loaded["spec"]
        model = StudyPolicy(spec["actor_size"], spec["critic_size"], spec["action_size"], spec["widths"])
        model.load_state_dict(loaded["model"])
        returns = torch.zeros_like(data["job_reward"])
        carry = torch.zeros(28)
        for t in reversed(range(450)):
            carry = data["job_reward"][t] + .995 * carry
            returns[t] = carry
        mask = data["active"].bool()
        inputs, targets = data["normalized_critic_before"][mask], returns[mask]
        generator = torch.Generator().manual_seed(specification["seed"])
        orders = [[torch.randperm(len(targets), generator=generator) for _ in range(4)] for _ in range(100)]
        predictions = {"targets": targets, "case_indices": torch.arange(28).expand(450, 28)[mask]}
        with torch.no_grad():
            predictions["before"] = model.value(inputs)
        report["active_rows"] = len(targets)
        report["saved_value_replay_max_error"] = float((predictions["before"] - data["critic_value"][mask]).abs().max())
        original = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith(("critic.", "value_head."))}
        for name, clip in specification["arms"].items():
            fitted = copy.deepcopy(model)
            parameters = list(fitted.critic.parameters()) + list(fitted.value_head.parameters())
            optimizer = torch.optim.Adam(parameters, lr=.0001)
            history, steps = [], 0
            for iteration, epoch_orders in enumerate(orders):
                if time.monotonic() - start > specification["wall_limit_s"]:
                    raise TimeoutError("Predeclared critic-fit wall limit exceeded")
                with torch.no_grad():
                    reference = fitted.value(inputs)
                for order in epoch_orders:
                    for offset in range(0, len(order), 512):
                        ids = order[offset:offset + 512]
                        value = fitted.value(inputs[ids])
                        squared = (value - targets[ids]).square()
                        if clip is not None:
                            clipped = reference[ids] + (value - reference[ids]).clamp(-clip, clip)
                            squared = torch.maximum(squared, (clipped - targets[ids]).square())
                        loss = squared.mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                        optimizer.step()
                        steps += 1
                with torch.no_grad():
                    prediction = fitted.value(inputs)
                    history.append({"round": iteration + 1, "rmse": float((prediction - targets).square().mean().sqrt()),
                                    "mean_prediction": float(prediction.mean())})
            predictions[name] = prediction
            report["arms"][name] = {"optimizer_steps": steps, "history": history,
                                   "actor_and_normalizers_unchanged": all(torch.equal(value, fitted.state_dict()[key]) for key, value in original.items()),
                                   "finite": all(bool(torch.isfinite(p).all()) for p in fitted.state_dict().values())}
        torch.save(predictions, args.output / "predictions.pt")
        report.update(status="completed", wall_s=time.monotonic() - start,
                      interpretation="Compare train-set fit at equal optimizer work. This isolates an optimization mechanism; it does not establish learned control or generalization.")
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", wall_s=time.monotonic() - start)
        raise
    finally:
        write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "rows": report["active_rows"], "replay_error": report["saved_value_replay_max_error"],
                      "arms": {name: {**result["history"][-1], "optimizer_steps": result["optimizer_steps"],
                                      "unchanged_actor": result["actor_and_normalizers_unchanged"]} for name, result in report["arms"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
