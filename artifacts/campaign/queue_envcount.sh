#!/usr/bin/env bash
# Separate the environment count from the episode length, once and cheaply.
#
# Terminal lateral error is near-deterministic at 32 environments -- among the
# survivors of the published cohorts it is 1.69 to 1.85 mm with a standard
# deviation of 0.054 mm -- and genuinely spread at 64, where the same point runs
# 0.05 to 2.5 mm. The 2.5 mm gate sits between them, which is why nominal is
# 97.9% in one and 54.7% in the other. Two things differ and only one is the
# environment count: the sweep also runs STEPS=6000 against the certification's
# 1900, and episode_length_s is max(phase budgets, steps/30 + 2).
#
# Two runs, one variable each, against the two points already measured.
set -u
cd /d/6axis-space-robotics || exit 1
while [ ! -f "artifacts/robustness64_seed6070/section_140x26.npz" ]; do sleep 120; done
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
echo "[$(date +%H:%M:%S)] 64 environments at the certification's episode length"
POINTS="nominal" EPISODES=64 ENVS=64 STEPS=1900 OUT=artifacts/envcount_64x1900 SEED=4070 \
  bash scripts/sweep_chain_robustness.sh
echo "[$(date +%H:%M:%S)] 32 environments at the sweep's episode length"
POINTS="nominal" EPISODES=32 ENVS=32 STEPS=6000 OUT=artifacts/envcount_32x6000 SEED=4070 \
  bash scripts/sweep_chain_robustness.sh
echo "[$(date +%H:%M:%S)] env-count probe done"
