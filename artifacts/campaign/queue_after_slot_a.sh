#!/usr/bin/env bash
# Two jobs that were launched by hand and need somewhere to live.
#
#   1. The noised capture fine-tune, started at 18:28 and stopped at 19:39 to get
#      back under the measured two-training ceiling. It had reached epoch 3,429
#      of 5,100 and resumes from its own last checkpoint, so nothing is lost.
#   2. Verification of the force-feedback seating policy, once it finishes. Both
#      halves, the way every other insert checkpoint is verified: the skill on
#      three held-out seeds, then the same weights inside the full chain against
#      the scripted guarded advance. A skill certification alone has never been
#      allowed to move the seating phase and must not start now.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"

until grep -q "slot A done" artifacts/campaign/training_slot_a.log 2>/dev/null; do sleep 300; done

RESUME=$(ls -t "$ROOT/grapple_grasp_l0_seed70_v9noised/nn/"last_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
if [ -n "$RESUME" ]; then
  echo "[$(date +%H:%M:%S)] resuming the noised capture fine-tune from $RESUME"
  "$PY" scripts/train.py --headless \
      --task Isaac-ZeroG-Blade-GrapplePin-GraspNoised-v0 \
      --num_envs 512 --seed 70 --robustness_level 0 --max_iterations 5100 \
      --checkpoint "$RESUME" --run_name grapple_grasp_l0_seed70_v9noised \
      > artifacts/campaign/train_grasp_seed70_v9noised_resumed.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] noised capture exit=$rc"
else
  echo "[$(date +%H:%M:%S)] no noised-capture checkpoint to resume; skipping"
fi

echo "[$(date +%H:%M:%S)] waiting for the force-feedback seating policy"
waited=0
until ls "$ROOT/grapple_insert_l0_seed70_v33force/nn/"*ep_3000_*.pth >/dev/null 2>&1; do
  sleep 300
  waited=$((waited + 300))
  if [ "$waited" -gt 43200 ]; then echo "[$(date +%H:%M:%S)] never finished; skipping"; break; fi
done
FORCE=$(ls -t "$ROOT/grapple_insert_l0_seed70_v33force/nn/"*ep_3000_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
if [ -n "$FORCE" ]; then
  echo "[$(date +%H:%M:%S)] verifying both halves of $FORCE"
  # The skill half runs on the force task it was trained on; the chain half is
  # what decides, and `verify_insert_skill.sh` runs the chain against the
  # scripted advance on the same three held-out seeds.
  CKPT="$FORCE" TAG=insert_v33force PLAY_TASK="Isaac-ZeroG-Blade-GrapplePin-InsertForce-Play-v0" CHAIN_TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflowForce-v0" \
    bash scripts/verify_insert_skill.sh > artifacts/campaign/verify_insert_v33force.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] verify exit=$rc"
  tail -25 artifacts/campaign/verify_insert_v33force.log
fi
echo "[$(date +%H:%M:%S)] after slot A done"
