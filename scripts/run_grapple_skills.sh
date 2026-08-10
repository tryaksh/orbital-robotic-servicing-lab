#!/usr/bin/env bash
# Train and certify the three head-on grapple-pin skills, end to end.
#
# One skill at a time: this laptop has 12 GB of VRAM and a 512-environment PPO
# run plus its optimizer will not share it with another. Each skill trains, then
# is evaluated on three held-out seeds across three curriculum stages, then the
# raw per-episode rows are pooled into one gated report under evidence/.
#
# The promotion gate is the project's existing one and is not relaxed here:
# at least the stated success rate pooled and in every stage, zero instability
# terminations, and zero non-finite terminal metrics.
#
# Usage: scripts/run_grapple_skills.sh [skill ...]   (default: all three)

set -u

PYTHON="C:/isaac-sim/python.bat"
NUM_ENVS="${NUM_ENVS:-512}"
EPOCHS="${EPOCHS:-700}"
EVAL_ENVS="${EVAL_ENVS:-128}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
LOGROOT="logs/rl_games"
mkdir -p artifacts/grapple evidence

skills=("$@")
if [ ${#skills[@]} -eq 0 ]; then
  skills=(Grasp Extract Insert)
fi

experiment_for() {
  case "$1" in
    Grasp)   echo "zero_g_blade_insertion_contact" ;;
    Extract) echo "zero_g_blade_insertion_contact" ;;
    Insert)  echo "zero_g_blade_insertion_contact" ;;
  esac
}

for skill in "${skills[@]}"; do
  task="Isaac-ZeroG-Blade-GrapplePin-${skill}-v0"
  play="Isaac-ZeroG-Blade-GrapplePin-${skill}-Play-v0"
  run="grapple_$(echo "$skill" | tr '[:upper:]' '[:lower:]')_l0_seed70"

  # Smoke first. A task that cannot take a step is not worth hours of GPU, and
  # every one of these three has failed a smoke at least once already.
  echo "[$(date +%H:%M:%S)] SMOKE $skill"
  "$PYTHON" scripts/train.py --headless --task "$task" --num_envs 32 --smoke \
      > "artifacts/grapple/smoke_${skill}.log" 2>&1
  if grep -qE "^Traceback" "artifacts/grapple/smoke_${skill}.log"; then
    echo "[$(date +%H:%M:%S)] SMOKE FAILED for $skill; skipping. Last error:"
    grep -E "Error|Exception" "artifacts/grapple/smoke_${skill}.log" | tail -2
    continue
  fi
  echo "[$(date +%H:%M:%S)] smoke clean"

  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] TRAIN $skill  task=$task envs=$NUM_ENVS epochs=$EPOCHS"
  echo "=============================================================="
  "$PYTHON" scripts/train.py --headless \
      --task "$task" \
      --num_envs "$NUM_ENVS" \
      --seed 70 \
      --robustness_level 0 \
      --max_iterations "$EPOCHS" \
      --run_name "$run" \
      > "artifacts/grapple/train_${skill}.log" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S)] train exit=$status"

  # rl-games writes the newest checkpoint under <experiment>/<run>/nn.
  checkpoint=$(ls -t "$LOGROOT"/*/"$run"/nn/*.pth 2>/dev/null | head -1)
  if [ -z "$checkpoint" ]; then
    echo "[$(date +%H:%M:%S)] NO CHECKPOINT for $skill; skipping evaluation"
    continue
  fi
  echo "[$(date +%H:%M:%S)] checkpoint: $checkpoint"

  # Held-out evaluation seeds, three per skill, one run per curriculum stage.
  rows=()
  for seed in 1070 2070 3070; do
    for stage in 0 1 2; do
      out="artifacts/grapple/${skill}_s${stage}_seed${seed}"
      "$PYTHON" scripts/play.py --headless \
          --task "$play" \
          --checkpoint "$checkpoint" \
          --num_envs "$EVAL_ENVS" \
          --episodes "$EVAL_EPISODES" \
          --curriculum_stage "$stage" \
          --seed "$seed" \
          --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      echo "[$(date +%H:%M:%S)]   eval stage=$stage seed=$seed exit=$?"
      rows+=("${out}.npz")
    done
  done

  "$PYTHON" scripts/aggregate_evaluation.py \
      --episodes "${rows[@]}" \
      --output "evidence/grapple_${skill,,}_certification.json" \
      --title "Head-on grapple-pin ${skill,,} skill" \
      --scope \
        "Simulation only. No result here was produced on real hardware." \
        "The grasp is physical pad-against-pin contact, not a fixed joint." \
        "One PPO training seed. The evaluation seeds are held out, but training repeatability is untested." \
      > "artifacts/grapple/aggregate_${skill}.log" 2>&1
  echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/grapple_${skill,,}_certification.json"
done

echo "[$(date +%H:%M:%S)] ALL DONE"
