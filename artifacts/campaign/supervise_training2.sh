#!/usr/bin/env bash
# Keep the GPU working: the trainings that close named blockers, in order of value.
#
# The first training supervisor finished at 15:38 and left the machine with one
# evaluation and no training at all. These are the three runs that turn open
# statements into closed ones.
#
#   1. The noised capture fine-tune, interrupted at epoch 3,400 of 5,100 by the
#      06:25 fork exhaustion. Finishing it turns the composition bound from
#      "granting a perfect capture, the certified skills over-predict the camera
#      chain by 72.8 points" into a real product with both skills measured.
#
#   2 and 3. The force-feedback seating policy at two more seeds. Its first seed
#      is the only one, and every published skill number in this repository
#      resting on a single seed is a named blocker. It matters more than usual
#      here: the chain arm scored 4/24 against the scripted controller's 20/24
#      and the skill half never ran, so the one thing we have is one seed of an
#      unverified policy.
#
# One at a time, beside whatever evaluation is running.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"

say () { echo "[$(date +%H:%M:%S)] $*"; }

RESUME=$(ls -t "$ROOT/grapple_grasp_l0_seed70_v9noised/nn/"last_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
if [ -n "$RESUME" ]; then
  say "STAGE 1/3  noised capture fine-tune, resuming from $(basename "$RESUME" | grep -oE 'ep_[0-9]+')"
  "$PY" scripts/train.py --headless \
      --task Isaac-ZeroG-Blade-GrapplePin-GraspNoised-v0 \
      --num_envs 512 --seed 70 --robustness_level 0 --max_iterations 5100 \
      --checkpoint "$RESUME" --run_name grapple_grasp_l0_seed70_v9noised \
      > artifacts/campaign/train_grasp_seed70_v9noised_resumed2.log 2>&1
  rc=$?
  say "noised capture exit=$rc"
else
  say "STAGE 1/3  no noised-capture checkpoint to resume; skipping"
fi

for seed in 71 72; do
  say "STAGE  force-feedback seating policy, seed $seed from scratch"
  "$PY" scripts/train.py --headless \
      --task Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0 \
      --num_envs 512 --seed "$seed" --robustness_level 0 --max_iterations 3000 \
      --run_name "grapple_insert_l0_seed${seed}_v33force" \
      > "artifacts/campaign/train_insert_seed${seed}_v33force.log" 2>&1
  rc=$?
  say "force insert seed $seed exit=$rc"
done

say "training supervisor 2 done"
