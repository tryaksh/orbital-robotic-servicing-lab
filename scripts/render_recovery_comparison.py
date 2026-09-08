"""Render a scientific trace figure and side-by-side recorded simulator video."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly_recovery.protocol import sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", type=int, default=2)
    args = parser.parse_args()
    args.comparison = args.comparison.resolve()
    args.output = args.output.resolve()
    comparison = json.loads(args.comparison.read_text())
    if comparison["status"] != "verified":
        raise ValueError("Only a verified matched pair can be rendered")
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v2 as imageio
    import matplotlib
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair = comparison["pairs"][args.job]
    cohorts = []
    for run in comparison["runs"]:
        directory = (ROOT / run["manifest"]).parent / "probe"
        report = json.loads((directory / "report.json").read_text())
        if report["video"]["job_index"] != args.job:
            raise ValueError("Recorded camera job differs from requested comparison")
        controls = [json.loads(row) for row in (directory / "control_samples.jsonl").read_text().splitlines()]
        physics = [json.loads(row) for row in (directory / "physics_samples.jsonl").read_text().splitlines()]
        physics = [row for row in physics if row["job_id"] == report["jobs"][args.job]["job_id"]]
        cohorts.append((directory, report, controls, physics))
    colors, labels = ["#087e8b", "#bf5039"], ["Scripted withdrawal + reinsertion", "Scripted continued insertion"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for color, label, (_, report, controls, physics) in zip(colors, labels, cohorts, strict=True):
        job = report["jobs"][args.job]
        dt = report["criteria"]["physics_dt"]
        times = np.array([row["step"] * 8 * dt for row in controls])
        active = times <= job["elapsed_s"]
        height = [(row["part_pos"][args.job][2] - report["initial_fixed_pos"][args.job][2] - report["geometry"]["hole_height_m"]) * 1000 for row in controls]
        axes[0].plot(times[active], np.asarray(height)[active], color=color, label=label)
        pt = [row["step"] * dt for row in physics]
        axes[1].plot(pt, [row["raw_force_n"] for row in physics], color=color, linewidth=0.75, alpha=0.7, label=label + ": raw wrist")
        axes[1].plot(pt, [row["peg_fixture_contact_force_n"] for row in physics], color=color, linestyle="--", linewidth=0.8, alpha=0.8, label=label + ": fixture normal")
        cumulative, values = 0.0, []
        for row, t in zip(controls, times, strict=True):
            if t <= job["elapsed_s"] + 1e-9:
                cumulative += 0.995 ** (row["step"] - 1) * row["reward"][args.job]
            values.append(cumulative)
        axes[2].plot(times, values, color=color, label=f"{label}: {cumulative:.3f}")
        axes[0].plot([job["elapsed_s"]], [np.asarray(height)[active][-1]], marker="o", color=color)
    witness = pair["recovery_witness"]
    axes[0].axhline(0, color="grey", linewidth=0.8, label="Fixture top")
    axes[0].axhline(-25, color="grey", linestyle=":", linewidth=0.8, label="Seated base target")
    axes[0].set_ylabel("Peg base above fixture top (mm)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].axhline(20, color="black", linestyle=":", linewidth=0.9, label="Raw wrist abort threshold")
    axes[1].set_ylabel("Simulated force (N)")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    axes[2].set_ylabel("Discounted job return")
    axes[2].set_xlabel("Job time (s)")
    axes[2].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.set_xlim(0, 30)
        ax.grid(alpha=0.2)
        for t in (witness.get("evaluator_first_stall_s"), 4, 7, pair["recovery"]["elapsed_s"]):
            if t is not None:
                ax.axvline(t, color="grey", linewidth=0.6, alpha=0.4)
    fig.suptitle(f"Assembly Recovery Lab | simulation | development seed 10070, case {args.job}\n"
                 "Identical initial state and approach; 30 s deadline, 0.5 s seating dwell, gamma = 0.995", fontsize=12)
    fig.savefig(args.output / "comparison.png", dpi=180)
    fig.savefig(args.output / "comparison.pdf")
    plt.close(fig)

    readers = [imageio.get_reader(str(directory / "trajectory.mp4")) for directory, _, _, _ in cohorts]
    fps = cohorts[0][1]["video"]["fps"]
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 25)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 21)
    frames = min(report["video"]["frames"] for _, report, _, _ in cohorts)
    poster_saved = False
    output_video = args.output / "paired_recovery.mp4"
    try:
        with imageio.get_writer(str(output_video), fps=fps, macro_block_size=8) as writer:
            for index in range(frames):
                canvas = Image.new("RGB", (1920, 864), (247, 248, 250))
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 10), "Assembly Recovery Lab  |  SIMULATION  |  Scripted development controllers", font=font, fill=(20, 30, 40))
                for arm, (reader, (_, report, controls, _), label) in enumerate(zip(readers, cohorts, labels, strict=True)):
                    frame = reader.get_data(index)
                    canvas.paste(Image.fromarray(frame), (960 * arm, 96))
                    job = report["jobs"][args.job]
                    elapsed = (index + 1) / fps
                    phase = controls[index]["phase"][args.job]
                    state = job["outcome"] if elapsed >= job["elapsed_s"] else phase
                    draw.text((24 + 960 * arm, 48), f"{label}  |  {elapsed:4.1f} s  |  {state}", font=small, fill=colors[arm])
                    score = report["returns"][args.job]["discounted_return"]
                    draw.text((24 + 960 * arm, 825), f"Outcome: {job['outcome']} at {job['elapsed_s']:.2f} s  |  Job return: {score:.3f}", font=small, fill=(20, 30, 40))
                writer.append_data(np.asarray(canvas))
                if not poster_saved and index >= round(pair["recovery"]["elapsed_s"] * fps):
                    canvas.save(args.output / "poster.png")
                    poster_saved = True
    finally:
        for reader in readers:
            reader.close()
    # Decode the saved comparison through its last frame; container existence is insufficient.
    with imageio.get_reader(str(output_video)) as reader:
        decoded = sum(1 for _ in reader)
    if decoded != frames:
        raise ValueError("Encoded comparison frame count mismatch")
    record = {"comparison": str(args.comparison.relative_to(ROOT)), "comparison_sha256": sha256(args.comparison),
              "job_index": args.job, "decoded_frames": decoded, "fps": fps, "duration_s": decoded / fps,
              "scope": "Side-by-side actual simulator recordings with labels; terminal jobs are held, and all post-terminal reward is excluded.",
              "artifacts": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size}
                            for p in args.output.iterdir() if p.is_file()]}
    (args.output / "visualization.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
