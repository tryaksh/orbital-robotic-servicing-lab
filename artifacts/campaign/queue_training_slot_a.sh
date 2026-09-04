#!/usr/bin/env bash
# Slot A, after the from-scratch grasp control: the extract seed spread.
#
# Extract's published 87.64% is not a from-scratch run -- v18pin resumed v17m130
# at epoch 10,600 for 2,000 more epochs on the corrected criterion and rack, and
# that lineage is four task corrections deep. Reproducing the whole lineage at
# three seeds is twelve hours a seed. So the spread is measured over the *final
# stage*: the same resume, the same checkpoint, the same 2,000 epochs, three
# seeds. Earlier stages are shared and the report has to say so, because a
# spread over one stage is not a spread over the procedure.
set -u
cd /d/6axis-space-robotics || exit 1
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
V17="$ROOT/grapple_extract_l0_seed70_v17m130/nn/last_zero_g_blade_insertion_contact_ep_10600_rew_168.46431.pth"
until ls "$ROOT/grapple_grasp_l0_seed70_v8scratch/nn/"*ep_3100_*.pth >/dev/null 2>&1; do sleep 120; done
echo "[$(date +%H:%M:%S)] grasp seed 70 finished; extract seed spread on the final stage"
for seed in 71 72; do
  echo "[$(date +%H:%M:%S)] extract seed $seed"
  "C:/isaac-sim/python.bat" scripts/train.py --headless \
      --task Isaac-ZeroG-Blade-GrapplePin-Extract-v0 \
      --num_envs 512 --seed "$seed" --robustness_level 0 \
      --max_iterations 12600 --checkpoint "$V17" \
      --run_name "grapple_extract_l0_seed${seed}_v18stage" \
      > "artifacts/campaign/train_extract_seed${seed}_v18stage.log" 2>&1
  echo "[$(date +%H:%M:%S)] extract seed $seed exit=$?"
done
echo "[$(date +%H:%M:%S)] slot A done"
