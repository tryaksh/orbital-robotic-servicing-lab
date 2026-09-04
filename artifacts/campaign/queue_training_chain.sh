#!/usr/bin/env bash
# Keep both training slots full. Each stage waits for the checkpoint that ends
# the run before it, so the GPU never idles between jobs.
set -u
cd /d/6axis-space-robotics || exit 1
ROOT="logs/rl_games/zero_g_blade_insertion_contact"

# Slot A: when the noised extract fine-tune reaches its target, take the slot
# for the second grasp training seed. Grasp is the shorter run and T3 needs
# three seeds before any skill number can carry a spread.
until ls "$ROOT/grapple_extract_l0_seed70_v19noised/nn/"*ep_14600_*.pth >/dev/null 2>&1; do sleep 120; done
echo "[$(date +%H:%M:%S)] noised extract finished; starting grasp seed 71"
"C:/isaac-sim/python.bat" scripts/train.py --headless \
    --task Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 \
    --num_envs 512 --seed 71 --robustness_level 0 --max_iterations 3100 \
    --run_name grapple_grasp_l0_seed71_v8scratch \
    > artifacts/campaign/train_grasp_seed71_v8scratch.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] grasp seed 71 exit=$rc"

echo "[$(date +%H:%M:%S)] starting grasp seed 72"
"C:/isaac-sim/python.bat" scripts/train.py --headless \
    --task Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 \
    --num_envs 512 --seed 72 --robustness_level 0 --max_iterations 3100 \
    --run_name grapple_grasp_l0_seed72_v8scratch \
    > artifacts/campaign/train_grasp_seed72_v8scratch.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] grasp seed 72 exit=$rc"
