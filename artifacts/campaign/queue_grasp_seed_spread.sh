#!/usr/bin/env bash
# Certify the from-scratch grasp seeds, which is what T3 actually asks for.
#
# The published 86.90% comes from `v7m130`, and that is a resume lineage: v5 to
# v6w65 to v7m130, across task corrections. Reproducing that lineage three times
# is not affordable, so the spread is measured over three *from-scratch* runs at
# a matched 3,100-epoch budget -- and the seed-70 run doubles as the control that
# says how much of the published number came from the lineage rather than from
# its last stage. It reached reward 28.48 against the lineage's 30.26.
#
# Same protocol as `run_grapple_skills.sh`: three curriculum stages, three
# held-out evaluation seeds, pooled with the unchanged gate. Anything else would
# not be comparable with the number it is a spread for.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
PLAY="Isaac-ZeroG-Blade-GrapplePin-Grasp-Play-v0"

until grep -q "datum pair perception done" artifacts/campaign/datum_pair_perception.log 2>/dev/null; do sleep 180; done
mkdir -p artifacts/campaign/graspspread

for seed in 70 71 72; do
  run="grapple_grasp_l0_seed${seed}_v8scratch"
  # Wait for this seed's training to have written its final checkpoint. Seed 70
  # is already done; 71 and 72 come off the slot-B queue in turn.
  waited=0
  until ls "$ROOT/$run/nn/"*ep_3100_*.pth >/dev/null 2>&1; do
    sleep 300
    waited=$((waited + 300))
    if [ "$waited" -gt 43200 ]; then
      echo "[$(date +%H:%M:%S)] seed $seed never finished within twelve hours; skipping"
      break
    fi
  done
  CKPT=$(ls -t "$ROOT/$run/nn/"*ep_3100_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
  if [ -z "$CKPT" ]; then echo "[$(date +%H:%M:%S)] no checkpoint for seed $seed"; continue; fi
  echo "[$(date +%H:%M:%S)] certifying grasp seed $seed: $CKPT"
  rows=()
  for eval_seed in 1070 2070 3070; do
    for stage in 0 1 2; do
      out="artifacts/campaign/graspspread/seed${seed}_s${stage}_eval${eval_seed}"
      "$PY" scripts/play.py --headless --task "$PLAY" --checkpoint "$CKPT" \
          --num_envs 128 --episodes 512 --curriculum_stage "$stage" --seed "$eval_seed" \
          --episode_metrics "${out}.npz" > "${out}.log" 2>&1
      rows+=("${out}.npz")
    done
  done
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
      --output "evidence/grapple_grasp_v8scratch_seed${seed}_certification.json" \
      --title "Capture skill, trained from scratch at seed ${seed}" \
      --scope \
        "Simulation only. No result here was produced on real hardware." \
        "The grasp is physical pad-against-pin contact, not a fixed joint." \
        "Trained from scratch for 3,100 epochs. The published v7m130 checkpoint is a resume lineage across task corrections, so this is a spread over the from-scratch procedure and not over the one that produced the published number." \
      > "artifacts/campaign/graspspread/aggregate_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] seed $seed aggregate exit=$rc"
  tail -5 "artifacts/campaign/graspspread/aggregate_seed${seed}.log"
done
echo "[$(date +%H:%M:%S)] grasp seed spread done"
