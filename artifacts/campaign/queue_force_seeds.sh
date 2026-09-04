#!/usr/bin/env bash
# Seeds and randomization for the first seating policy that can feel contact.
#
# Two things this queue exists to prevent.
#
#   1. **A headline claim from one seed.** `v33force` is seed 70 alone. Every
#      published skill number in this repository comes from one training seed and
#      that is a named blocker; a result this large must not join them. Seeds 71
#      and 72 train the identical task from scratch.
#   2. **Claiming the FORGE recipe while using none of its randomization.** The
#      grapple-pin family already implements it -- `configure_robustness` level 2
#      randomizes blade mass, level 3 adds slot and guide materials, stiction and
#      rail stiction -- and every skill this project has ever trained ran at
#      level 0. The third run is the same force task at level 3, and it is the
#      arm that answers what randomization buys here rather than asserting it.
#
# The level-3 run is a training-distribution change, not a criterion change, and
# it is published as its own arm beside the level-0 one. The level-0 policy is
# kept whatever the level-3 policy does.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"

until grep -q "slot B done" artifacts/campaign/training_slot_b.log 2>/dev/null; do sleep 300; done
echo "[$(date +%H:%M:%S)] slot B free; force-feedback seeds"

for seed in 71 72; do
  echo "[$(date +%H:%M:%S)] force insert seed $seed from scratch"
  "$PY" scripts/train.py --headless \
      --task Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0 \
      --num_envs 512 --seed "$seed" --robustness_level 0 --max_iterations 3000 \
      --run_name "grapple_insert_l0_seed${seed}_v33force" \
      > "artifacts/campaign/train_insert_seed${seed}_v33force.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] force insert seed $seed exit=$rc"
done

# Level 3 keeps six event terms the grapple-pin scene has never run with: blade
# mass, the slot and both guide materials, stiction and rail stiction. They are
# defined on RobustInsertionEventsCfg and inherited, so they exist -- but they
# were written against the single-slot contact scene and nothing has ever
# constructed them here. Two minutes at one environment decides that, rather
# than three hours.
echo "[$(date +%H:%M:%S)] smoking robustness level 3 before committing hours to it"
"$PY" scripts/train.py --headless     --task Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0     --num_envs 1 --seed 70 --robustness_level 3 --max_iterations 2     --run_name smoke_l3_forcerand     > artifacts/campaign/smoke_l3_forcerand.log 2>&1
SMOKE=$?
echo "[$(date +%H:%M:%S)] level-3 smoke exit=$SMOKE"
if [ "$SMOKE" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] level 3 does not construct on this scene; skipping the randomized arm"
  grep -aE "Error|error:|Traceback" artifacts/campaign/smoke_l3_forcerand.log | head -5
  echo "[$(date +%H:%M:%S)] force seeds done"
  exit 0
fi

echo "[$(date +%H:%M:%S)] force insert seed 70 at robustness level 3"
"$PY" scripts/train.py --headless \
    --task Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0 \
    --num_envs 512 --seed 70 --robustness_level 3 --max_iterations 3000 \
    --run_name grapple_insert_l3_seed70_v34forcerand \
    > artifacts/campaign/train_insert_seed70_v34forcerand.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] randomized force insert exit=$rc"

echo "[$(date +%H:%M:%S)] force seeds done"
