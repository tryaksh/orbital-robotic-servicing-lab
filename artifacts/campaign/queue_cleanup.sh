#!/usr/bin/env bash
# After the master queue: the ladder rung the solved-IK check killed, and the
# +8 mm rung, so the rail curve has five points between 0 and 10 mm.
set -u
cd /d/6axis-space-robotics || exit 1
# One evaluation process at a time: two training slots are permanently
# occupied and an Isaac process is about nine gigabytes of thirty-two, so a
# fourth exhausts the machine. The chain is master -> cohorts -> skill cert
# -> here -> traced rung -> degradation curve.
# Behind the datum-pair perception certification, which closes a gap in a
# published claim -- no fiducial certificate exists for the datum layout the
# chain actually carries -- where this fills in two ladder rungs.
# Behind the grasp seed spread, which is a named reviewer demand (T3), where
# this fills in two ladder rungs.
until grep -q "grasp seed spread done" artifacts/campaign/grasp_seed_spread.log 2>/dev/null; do sleep 180; done
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
echo "[$(date +%H:%M:%S)] re-running the ladder rungs that are missing"
POINTS="base_y_+1mm base_y_+8mm" BASE_Y_MM="1 8" EPISODES=64 ENVS=64 STEPS=6000 \
  OUT=artifacts/robustness64_baseladder SEED=4070 bash scripts/sweep_chain_robustness.sh
echo "[$(date +%H:%M:%S)] cleanup done"
