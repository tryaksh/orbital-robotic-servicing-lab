"""Regenerate the hand-off pose bank the insert task resets from.

Reads one or more `run_workflow_demo.py --handoff_trace` files and writes
`src/zero_g_blade_swap/tasks/blade_swap/handoff_poses.py`.

Collect the traces on training-side seeds. A reset distribution drawn from the
seeds a skill is certified on is not a held-out evaluation any more, and this
script cannot check that for you.

    python scripts/build_handoff_pose_bank.py artifacts/handoff/bank_seed*.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PHASE_NAMES = ("capture", "seat", "extract", "transit", "insert", "done")
DEFAULT_OUTPUT = Path("src/zero_g_blade_swap/tasks/blade_swap/handoff_poses.py")
#: The nominal head-on pose the deviation figures in the generated docstring are
#: measured against: assets.GRAPPLE_HEAD_ON_ARM_JOINT_POS[2].
NOMINAL = (-0.403679, -1.653908, 2.293487, 2.502023, -1.167132, -1.570785)

#: Module pose columns, in the order `reset_from_handoff_bank` writes them.
MODULE_COLUMNS = ("blade_x_m", "blade_y_m", "blade_z_m", "blade_qw", "blade_qx", "blade_qy", "blade_qz")

TEMPLATE = '''"""Arm poses the capture skill actually hands the insert skill.

Not authored. Collected with ``scripts/run_workflow_demo.py --handoff_trace`` over
{episodes} chained installations on {seeds} -- deliberately not the seeds the chain
is certified on, because a reset distribution drawn from the evaluation seeds is
not a held-out evaluation any more.

Why a bank and not a wider noise term. The state a capture hands over is a point
on a manifold: the arm sits {p50:.3f} rad from the nominal head-on pose on its worst
axis at the median, almost all of it ``wrist_1``, *and* the grip error there is
about 12.5 mm, because the capture servoed to it. Independent per-joint noise wide
enough to reach that deviation breaks the correlation -- it produces a large joint
error and a large grip error together, the fingers close on nothing, and 534 of
534 measured episodes lost the grip at reset. Sampling measured poses keeps the
correlation for free.

Regenerate with ``scripts/build_handoff_pose_bank.py``.
"""

from __future__ import annotations

#: {count} measured hand-off poses, in ARM_JOINTS order. Worst-axis deviation from
#: the nominal pose: p50 {p50:.4f}, p95 {p95:.4f}, max {maximum:.4f} rad.
INSERT_HANDOFF_ARM_POSES: tuple[tuple[float, ...], ...] = (
{body}
)

#: The module pose measured at the *same* hand-off, row for row: env-local
#: position then quaternion. Row ``i`` of this and of the arm table are one
#: state, and they must be drawn with one index. Sampling them independently is
#: what made the arm-only bank unfaithful -- the arm moved and the module did
#: not, so the grip geometry was one the chain never produces.
INSERT_HANDOFF_MODULE_POSES: tuple[tuple[float, ...], ...] = (
{module_body}
)

__all__ = ["INSERT_HANDOFF_ARM_POSES", "INSERT_HANDOFF_MODULE_POSES"]
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--to_phase", default="insert", choices=PHASE_NAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    target = PHASE_NAMES.index(args.to_phase)
    poses: list[np.ndarray] = []
    modules: list[np.ndarray] = []
    seeds: list[str] = []
    episodes = 0
    for path in args.traces:
        data = np.load(path)
        fields = [str(name) for name in data["handoff_fields"]]
        index = {name: position for position, name in enumerate(fields)}
        missing = [name for name in MODULE_COLUMNS if name not in index]
        if missing:
            raise SystemExit(
                f"{path.name} predates the module-orientation columns ({', '.join(missing)}). "
                "Re-collect the trace: an arm pose without the module pose it was reached against "
                "is not a hand-off, and a bank built from one measures 26% where the chain gets 80%."
            )
        handoff = data["handoff"]
        selected = handoff[handoff[:, index["to_phase"]] == target]
        joints = np.stack([selected[:, index[f"arm_joint_{axis}"]] for axis in range(6)], axis=-1)
        module = np.stack([selected[:, index[name]] for name in MODULE_COLUMNS], axis=-1)
        poses.append(joints)
        modules.append(module)
        episodes += int(selected.shape[0])
        stem = path.stem
        seeds.append(stem.split("seed")[-1] if "seed" in stem else stem)
        print(f"{path.name}: {joints.shape[0]} hand-offs to {args.to_phase}")

    bank = np.concatenate(poses)
    module_bank = np.concatenate(modules)
    if bank.shape[0] == 0:
        raise SystemExit(f"no hand-offs to {args.to_phase} in the given traces")
    deviation = np.abs(bank - np.asarray(NOMINAL)).max(axis=-1)
    body = "\n".join("    (" + ", ".join(f"{value:.6f}" for value in row) + ")," for row in bank)
    module_body = "\n".join(
        "    (" + ", ".join(f"{value:.6f}" for value in row) + ")," for row in module_bank
    )
    text = TEMPLATE.format(
        count=bank.shape[0],
        episodes=episodes,
        seeds="seeds " + ", ".join(sorted(set(seeds))),
        body=body,
        module_body=module_body,
        p50=float(np.percentile(deviation, 50)),
        p95=float(np.percentile(deviation, 95)),
        maximum=float(deviation.max()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.output}: {bank.shape[0]} poses, worst-axis deviation "
        f"p50={np.percentile(deviation, 50):.4f} p95={np.percentile(deviation, 95):.4f} "
        f"max={deviation.max():.4f} rad"
    )


if __name__ == "__main__":
    main()
