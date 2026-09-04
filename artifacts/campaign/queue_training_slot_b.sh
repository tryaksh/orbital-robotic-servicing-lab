#!/usr/bin/env bash
# Slot B, in the order that gets the answers soonest.
#
#   1. grasp seed 71                 -- T3 needs three seeds before any skill
#                                       number can carry a spread
#   2. the wedge-gated insert resume -- two hours, and it decides whether the
#                                       paper claims a learned seating phase or
#                                       defends the scripted one
#   3. grasp seed 72                 -- the third seed
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
until ls "$ROOT/grapple_extract_l0_seed70_v19noised/nn/"*ep_14600_*.pth >/dev/null 2>&1; do sleep 120; done

echo "[$(date +%H:%M:%S)] grasp seed 71"
"$PY" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 \
    --num_envs 512 --seed 71 --robustness_level 0 --max_iterations 3100 \
    --run_name grapple_grasp_l0_seed71_v8scratch \
    > artifacts/campaign/train_grasp_seed71_v8scratch.log 2>&1
echo "[$(date +%H:%M:%S)] grasp seed 71 exit=$?"

echo "[$(date +%H:%M:%S)] wedge-gated insert, resumed from the frozen v24rack weights"
V24="$ROOT/grapple_insert_l0_seed70_v24rack/nn/last_zero_g_blade_insertion_contact_ep_2100_rew_43.909218.pth"
"$PY" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-InsertWedgeGated-v0 \
    --num_envs 512 --seed 70 --robustness_level 0 --max_iterations 3300 \
    --checkpoint "$V24" --run_name grapple_insert_l0_seed70_v32wedgegated \
    > artifacts/campaign/train_insert_v32wedgegated.log 2>&1
echo "[$(date +%H:%M:%S)] wedge insert exit=$?"
WEDGE=$(ls -t "$ROOT/grapple_insert_l0_seed70_v32wedgegated/nn/"last_*.pth 2>/dev/null | head -1)
if [ -n "$WEDGE" ]; then
  echo "[$(date +%H:%M:%S)] verifying both halves of $WEDGE"
  CKPT="$WEDGE" TAG=insert_v32wedgegated bash scripts/verify_insert_skill.sh \
      > artifacts/campaign/verify_insert_v32.log 2>&1
  echo "[$(date +%H:%M:%S)] verify exit=$?"
  tail -20 artifacts/campaign/verify_insert_v32.log
fi

echo "[$(date +%H:%M:%S)] grasp seed 72"
"$PY" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 \
    --num_envs 512 --seed 72 --robustness_level 0 --max_iterations 3100 \
    --run_name grapple_grasp_l0_seed72_v8scratch \
    > artifacts/campaign/train_grasp_seed72_v8scratch.log 2>&1
echo "[$(date +%H:%M:%S)] grasp seed 72 exit=$?"
echo "[$(date +%H:%M:%S)] slot B done"
