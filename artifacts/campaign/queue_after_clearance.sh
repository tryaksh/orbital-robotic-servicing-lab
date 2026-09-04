#!/usr/bin/env bash
# Wait for the clearance re-measurement, then ladder the rail's stopping error.
set -u
cd /d/6axis-space-robotics || exit 1
while [ ! -f artifacts/robustness64_channel/rack_lat_16mm.npz ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] clearance points landed; starting the base_y ladder"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
POINTS="base_y_+1mm base_y_+2mm base_y_+4mm base_y_+6mm" BASE_Y_MM="1 2 4 6" \
  EPISODES=64 ENVS=64 STEPS=6000 OUT=artifacts/robustness64_baseladder SEED=4070 \
  bash scripts/sweep_chain_robustness.sh
rc=$?
echo "[$(date +%H:%M:%S)] ladder done exit=$rc"
