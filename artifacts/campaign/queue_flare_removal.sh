#!/usr/bin/env bash
# Which mechanism squares the module during the stroke?
#
# Two points, both at 6 mm of lateral clearance per side with the mouth moving
# with the walls, differing only in whether the bays have their lateral entry
# flares. If the no-flare arm still seats, the guarded advance is doing the
# squaring and the flare is not load-bearing for the entry criterion. If it
# jams, the flare is, and the closed form's lower bound belongs on the flare's
# catch rather than on the channel.
set -u
cd /d/6axis-space-robotics || exit 1
while [ ! -f artifacts/envcount_32x6000/nominal.npz ]; do sleep 120; done
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
echo "[$(date +%H:%M:%S)] 6 mm per side, flares removed"
POINTS="rack_lat_6mm" EPISODES=64 ENVS=64 STEPS=6000 \
  OUT=artifacts/robustness64_noflare SEED=4070 \
  SWEEP_EXTRA="--rack_clearance_scope channel --remove_entry_flares" \
  bash scripts/sweep_chain_robustness.sh
echo "[$(date +%H:%M:%S)] and the nominal channel with the flares removed, for scale"
POINTS="nominal" EPISODES=64 ENVS=64 STEPS=6000 \
  OUT=artifacts/robustness64_noflare_nominal SEED=4070 \
  SWEEP_EXTRA="--remove_entry_flares" \
  bash scripts/sweep_chain_robustness.sh
echo "[$(date +%H:%M:%S)] flare removal done"
