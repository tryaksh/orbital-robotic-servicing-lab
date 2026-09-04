#!/usr/bin/env bash
set -u
cd /d/6axis-space-robotics || exit 1
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
echo "[$(date +%H:%M:%S)] clearance points re-measured with the mouth moving with the walls"
POINTS="rack_lat_6mm rack_lat_16mm" EPISODES=64 ENVS=64 STEPS=6000 \
  OUT=artifacts/robustness64_channel SEED=4070 \
  SWEEP_EXTRA="--rack_clearance_scope channel" \
  bash scripts/sweep_chain_robustness.sh
rc=$?
echo "[$(date +%H:%M:%S)] done exit=$rc"
