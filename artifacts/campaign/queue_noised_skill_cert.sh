#!/usr/bin/env bash
# Certify the noised extract fine-tune on both tasks, the moment it finishes.
#
# Two certificates, because the interesting number is the pair. On the noised
# task it says whether the policy learned to work against the estimator's error;
# on the unchanged state task it says what that cost against the published
# 87.64%. A policy that wins the first and loses the second has traded, and the
# trade has to be published either way.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN="grapple_extract_l0_seed70_v19noised"
# Two conditions, because RAM is the binding resource: an Isaac process here
# takes roughly nine gigabytes, two training slots are permanently occupied,
# and a fourth process would exhaust the machine. So wait for the fine-tune
# to finish AND for the master evaluation queue to be out of the way.
until ls "$ROOT/$RUN/nn/"*ep_14600_*.pth >/dev/null 2>&1; do sleep 120; done
# Behind the pooled RGB-D cohorts: those are the submission gate and this is
# the supporting evidence for them, so the gate measures first.
until grep -q "RGB-D cohorts done" artifacts/campaign/rgbd_cohorts.log 2>/dev/null; do sleep 120; done
CKPT=$(ls -t "$ROOT/$RUN/nn/"*ep_14600_*.pth | head -1)
echo "[$(date +%H:%M:%S)] certifying $CKPT"
mkdir -p artifacts/campaign/noisedcert

for arm in noised clean; do
  if [ "$arm" = "noised" ]; then
    TASK="Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-Play-v0"
    TITLE="Extraction fine-tuned on the estimator's error, evaluated on it"
    NOTE="Observations carry the deployed estimator's certified residual, sample-and-hold and miss rate."
  else
    TASK="Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0"
    TITLE="Extraction fine-tuned on the estimator's error, evaluated on exact state"
    NOTE="The unchanged state task, so this is directly comparable with the published 87.64% pin certificate."
  fi
  rows=()
  for seed in 1070 2070 3070; do
    for stage in 0 1 2; do
      out="artifacts/campaign/noisedcert/${arm}_s${stage}_seed${seed}"
      "$PY" scripts/play.py --headless --task "$TASK" --checkpoint "$CKPT" \
          --num_envs 128 --episodes 1000 --curriculum_stage "$stage" --seed "$seed" \
          --episode_metrics "${out}.npz" > "${out}.log" 2>&1
      echo "[$(date +%H:%M:%S)]   $arm stage=$stage seed=$seed exit=$?"
      rows+=("${out}.npz")
    done
  done
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py \
      --episodes "${rows[@]}" \
      --output "evidence/grapple_extract_v19noised_${arm}_certification.json" \
      --title "$TITLE" \
      --scope \
        "Simulation only. No result here was produced on real hardware." \
        "The grasp is physical pad-against-pin contact, not a fixed joint." \
        "One PPO training seed, resumed from the certified v18pin checkpoint; the evaluation seeds are held out." \
        "$NOTE" \
      > "artifacts/campaign/noisedcert/aggregate_${arm}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $arm aggregate exit=$?"
  tail -5 "artifacts/campaign/noisedcert/aggregate_${arm}.log"
done
echo "[$(date +%H:%M:%S)] noised skill certification done"
