#!/usr/bin/env bash
# Does correcting the rack the way the tool prescribes fix the dominant failure?
#
# The design library says this destination channel's lateral clearance must sit
# between 10.350 and 11.781 mm per side, with an equal-margin design point at
# 11.066 mm. The unrelieved channel is 11.065 mm -- the design point almost
# exactly. Every chain run in this project adds
# `--destination_channel_relief_m 0.0046125`, which takes it to 15.678 mm:
# **3.897 mm above the tool's own upper bound.**
#
# The upper bound exists because a channel wider than it lets a *seated* module
# rest further off the centre line than the seating gate accepts. That is the
# chain's dominant failure: 77 of 192 nominal episodes arrive, seat and miss the
# terminal gate, and what separates them from the successes is lateral error --
# 3.348 mm against 1.395 -- not orientation, where not one of the 77 exceeds its
# tolerance.
#
# An old 16-episode pair says the mechanism moves: median lateral error is
# 0.606 mm at zero relief against 1.725 mm at the shipped relief, same seed, one
# flag. Both scored 15/16, so the rate could not show it at that sample.
#
# This measures it at the sample the rest of the boundary work uses: three
# held-out seeds, 64 environments, 6,000 steps, paired against
# `artifacts/robustness64_corrected` and the two seed directories beside it,
# which are the same configuration with the relief left in.
#
# **If the terminal-gate rate falls, the paper closes its own loop**: the tool
# derived a bound, our rack violated it, the violation predicted a specific
# failure mode, and removing the violation removed the failure. If it does not
# fall, the upper bound does not govern and that is worth as much -- it is the
# same test either way.
#
# Run from the scratchpad so the repository stays clean while runs are in
# flight; a dirty worktree is recorded in every report produced during it.
set -u
cd /d/6axis-space-robotics || exit 1
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"

say () { echo "[$(date +%H:%M:%S)] $*"; }

until grep -q "factorial done" artifacts/campaign/supervise_factorial.log 2>/dev/null; do sleep 300; done
say "factorial finished; measuring the tool's own prescription"

for seed in 4070 5070 6070; do
  say "zero relief, seed $seed"
  POINTS="nominal" EPISODES=64 ENVS=64 STEPS=6000 \
    OUT="artifacts/relief0_seed${seed}" SEED="$seed" \
    RELIEF=0.0 \
    bash scripts/sweep_chain_robustness.sh
  rc=$?
  say "  seed $seed exit=$rc"
done

say "relief prescription measured; comparing"
./.venv/Scripts/python.exe - <<'PY'
import numpy as np, pathlib, sys, importlib.util
sys.path.insert(0, ".")
from src.zero_g_blade_swap.evaluation import wilson_interval
spec = importlib.util.spec_from_file_location("m", "scripts/report_boundary_failure_modes.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def pool(paths):
    rows, fields = [], None
    for p in paths:
        f = pathlib.Path(p)
        if not f.exists():
            continue
        a = np.load(f, allow_pickle=True)
        fields = fields or [str(x) for x in a["fields"]]
        rows.append(a["rows"].astype(float))
    return (np.concatenate(rows), fields) if rows else (None, None)

arms = {
    "relief 4.6125 mm (shipped)": [
        "artifacts/robustness64_corrected/nominal.npz",
        "artifacts/robustness64_seed5070/nominal.npz",
        "artifacts/robustness64_seed6070/nominal.npz",
    ],
    "relief 0 mm (tool's design point)": [
        f"artifacts/relief0_seed{s}/nominal.npz" for s in (4070, 5070, 6070)
    ],
}
for label, paths in arms.items():
    rows, f = pool(paths)
    if rows is None:
        print(f"  {label}: no archives")
        continue
    s = int((rows[:, f.index("success")] > 0.5).sum())
    n = rows.shape[0]
    lo, hi = wilson_interval(s, n)
    dec = m.decompose(rows, f)
    modes = {k: v["count"] for k, v in dec["modes"].items()}
    lat = rows[:, f.index("lateral_error_m")] * 1000
    print(f"  {label:36s} {s:3d}/{n:<4d} {s/n:.4f} [{lo:.3f},{hi:.3f}]")
    print(f"      missed the gate {modes['missed_the_terminal_gate']:3d}   median lateral {np.median(lat):.3f} mm")
PY

say "relief study done"
