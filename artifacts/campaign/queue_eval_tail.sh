#!/usr/bin/env bash
# Evaluation tail: the degradation curve nobody has ever measured (T4).
#
# Every certification in this repository is at robustness level 0, and levels
# 1 to 3 are implemented and unexercised. Re-certifying the *unchanged*
# checkpoints at 1, 2 and 3 costs evaluation time only and answers the honest
# first question -- how much does the promoted policy lose to wider reset noise,
# to randomized module mass, and to randomized slot and guide friction. Level 4
# is skipped: its base compliance is authored outside the load path, so a number
# there would imply a mount compliance that is not being simulated.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
# Behind the traced rung, which is nineteen minutes and settles a hypothesis,
# where this is three and a half hours and confirms a curve.
until grep -q "traced rung done" artifacts/campaign/trace_rung.log 2>/dev/null; do sleep 180; done
mkdir -p artifacts/campaign/robustness
declare -A CKPT=(
  [grasp]="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
  [extract]="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
)
declare -A PLAY=(
  [grasp]="Isaac-ZeroG-Blade-GrapplePin-Grasp-Play-v0"
  [extract]="Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0"
)
for skill in grasp extract; do
  for level in 1 2 3; do
    rows=()
    for seed in 1070 2070 3070; do
      for stage in 0 1 2; do
        out="artifacts/campaign/robustness/${skill}_l${level}_s${stage}_seed${seed}"
        "$PY" scripts/play.py --headless --task "${PLAY[$skill]}" --checkpoint "${CKPT[$skill]}" \
            --num_envs 128 --episodes 512 --curriculum_stage "$stage" --seed "$seed" \
            --robustness_level "$level" \
            --episode_metrics "${out}.npz" > "${out}.log" 2>&1
        rows+=("${out}.npz")
      done
    done
    echo "[$(date +%H:%M:%S)] $skill level $level evaluated"
    ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/grapple_${skill}_robustness_level${level}_certification.json" \
        --title "Unchanged ${skill} checkpoint re-certified at robustness level ${level}" \
        --scope \
          "Simulation only. No result here was produced on real hardware." \
          "The promoted checkpoint, unchanged. Only the robustness profile differs from its level-0 certificate." \
          "Level 1 widens the arm reset noise, level 2 adds randomized module mass, level 3 adds slot and guide friction and stiction. Level 4 is excluded: its base compliance is authored outside the load path." \
        > "artifacts/campaign/robustness/aggregate_${skill}_l${level}.log" 2>&1
    tail -4 "artifacts/campaign/robustness/aggregate_${skill}_l${level}.log"
  done
done
echo "[$(date +%H:%M:%S)] degradation curve done"
