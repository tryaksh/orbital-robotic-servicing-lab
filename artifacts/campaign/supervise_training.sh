#!/usr/bin/env bash
# One sequential training supervisor. Replaces three slot queues and their waiters.
#
# The overnight run died at 06:25 with `fork: retry: Resource temporarily
# unavailable` and took three trainings with it -- capture seed 71 at 2,805 of
# 3,100 epochs, extraction seed 71 at 12,541 of 12,600, and the force-feedback
# seating policy at 1,400 of 3,000. All three resume from their own last
# checkpoint, so nothing is lost except the hours.
#
# Two things changed besides the structure.
#
# **One training at a time, beside one evaluation.** Three concurrent trainings
# aggregate more frames per second than one, which is why rule 7 says run three,
# but that measurement was taken on a machine that had not yet fallen over. Until
# the fork exhaustion is understood, throughput is worth less than finishing.
#
# **The force run's log is suppressed.** It reached 21 MB, almost entirely
# `PhysicsUSD: CreateJoint - found a joint with disjointed body transforms`,
# emitted every reset of every environment. That is a known and harmless warning
# about the release latch, and writing it a few million times is not free.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"

say () { echo "[$(date +%H:%M:%S)] $*"; }

resume () {
  run="$1"; task="$2"; seed="$3"; iters="$4"; extra="${5:-}"
  ckpt=$(ls -t "$ROOT/$run/nn/"last_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
  if [ -z "$ckpt" ]; then say "$run has no checkpoint to resume; skipping"; return 1; fi
  say "resuming $run from $(basename "$ckpt" | grep -oE 'ep_[0-9]+')"
  # shellcheck disable=SC2086
  "$PY" scripts/train.py --headless --task "$task" \
      --num_envs 512 --seed "$seed" --robustness_level 0 --max_iterations "$iters" \
      --checkpoint "$ckpt" --run_name "$run" $extra \
      > "artifacts/campaign/train_${run}_resumed.log" 2>&1
  rc=$?
  say "$run exit=$rc, highest epoch now $(ls "$ROOT/$run/nn/"*.pth 2>/dev/null | grep -oE 'ep_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)"
}

# Nearly finished, so cheapest first: sixty epochs and three hundred.
say "STAGE 1/4  extraction seed 71, 59 epochs from the end"
resume grapple_extract_l0_seed71_v18stage "Isaac-ZeroG-Blade-GrapplePin-Extract-v0" 71 12600

say "STAGE 2/4  capture seed 71, 295 epochs from the end"
resume grapple_grasp_l0_seed71_v8scratch "Isaac-ZeroG-Blade-GrapplePin-Grasp-v0" 71 3100

say "STAGE 3/4  the force-feedback seating policy, from 1,400"
resume grapple_insert_l0_seed70_v33force "Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0" 70 3000

say "STAGE 4/4  capture seed 72 from scratch, for the seed spread"
"$PY" scripts/train.py --headless \
    --task Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 \
    --num_envs 512 --seed 72 --robustness_level 0 --max_iterations 3100 \
    --run_name grapple_grasp_l0_seed72_v8scratch \
    > artifacts/campaign/train_grasp_seed72_v8scratch.log 2>&1
rc=$?
say "capture seed 72 exit=$rc"

say "training supervisor done"
