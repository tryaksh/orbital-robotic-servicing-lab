#!/usr/bin/env bash
# Raise n on the section axis, the only way that keeps the points comparable.
#
# section_120x16 is grip-inadmissible by 3.92 mm and loses the module before
# delivery in 0.156 of its episodes against nominal's 0.031 -- the direction the
# criterion predicts, and not separated at 64 episodes. Two further held-out
# seeds take the three section points to n=192 each. The environment count stays
# at 64 because this repository has an open question about whether the rate
# moves with it, and changing two things at once would answer neither.
set -u
cd /d/6axis-space-robotics || exit 1
while [ ! -f artifacts/campaign/visionprobe/probe.npz ] && [ ! -f artifacts/campaign/visionprobe/probe_report.json ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] probe landed; raising n on the section axis"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
for seed in 5070 6070; do
  POINTS="nominal section_120x16 section_140x26" EPISODES=64 ENVS=64 STEPS=6000 \
    OUT="artifacts/robustness64_seed${seed}" SEED="$seed" \
    bash scripts/sweep_chain_robustness.sh
  rc=$?
  echo "[$(date +%H:%M:%S)] seed $seed exit=$rc"
done
echo "[$(date +%H:%M:%S)] section power done"
