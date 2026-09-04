#!/usr/bin/env bash
# Slot A tail: capture, trained against the estimator's error.
#
# Three of the eight environments in the first pooled RGB-D cohort never
# captured at all, so extraction is not the only phase paying for the transfer.
# Same construction as the extract arm: resume the certified checkpoint on the
# noised task at the same seed, one change from a published arm.
set -u
cd /d/6axis-space-robotics || exit 1
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
until grep -q "slot A done" artifacts/campaign/training_slot_a.log 2>/dev/null; do sleep 180; done
V7="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
echo "[$(date +%H:%M:%S)] noised capture fine-tune"
"C:/isaac-sim/python.bat" scripts/train.py --headless \
    --task Isaac-ZeroG-Blade-GrapplePin-GraspNoised-v0 \
    --num_envs 512 --seed 70 --robustness_level 0 --max_iterations 5100 \
    --checkpoint "$V7" --run_name grapple_grasp_l0_seed70_v9noised \
    > artifacts/campaign/train_grasp_seed70_v9noised.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] noised capture exit=$rc"
