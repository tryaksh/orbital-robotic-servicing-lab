#!/usr/bin/env bash
# Train and measure insertion under a wrong pose belief, force-aware against a
# matched force-blind control.
#
# The experiment is only valid if the two arms differ in one thing: whether the
# actor observes the contact wrench. They therefore share this script, one PPO
# configuration, one training seed, and one schedule, and each is trained from
# scratch because the observation width differs and no checkpoint survives that.
#
# The schedule is 1800 PPO epochs at 512 environments, robustness level 2
# throughout, which is the same budget that produced this project's promoted
# policies and its earlier force-feedback comparison.
#
# It does NOT walk up the robustness levels the way those runs did, and the
# reason is specific to this task. Level 0 disables rail collision entirely and
# level 1 leaves 6 mm of clearance per side, so at both of them a displaced
# channel is untouchable: there is nothing to feel and nothing to push the blade
# straight. A policy trained there would spend most of its budget learning that
# contact carries no information, which is the opposite of the lesson. Only
# level 2's 0.75 mm per side both hides the displacement from every other sense
# and reveals it on contact.
#
# The curriculum that remains is IndustReal's, over the displacement itself: the
# full 0 to 4 mm range is sampled from the first step and the easy end is
# withdrawn as success improves.
#
# Two kinds of measurement come out of it:
#
#   certification  the trained displacement, three held-out seeds, judged
#                  against the promotion gate
#   envelope       displacement swept past what either policy trained on,
#                  labelled a capability measurement
#
# There is one reset distance, not three. Moving the channel downstream so its
# lead-in starts ahead of the blade leaves no room for the two nearer stages;
# see uncertain_insertion_env_cfg.py. Every episode is a full-distance approach.
#
# Usage: scripts/run_uncertain_insertion.sh [train|certify|sweep|aggregate ...]
#        (default: everything, in order)

set -u

PYTHON="C:/isaac-sim/python.bat"
NUM_ENVS="${NUM_ENVS:-512}"
SEED="${SEED:-80}"
EPOCHS="${EPOCHS:-1800}"
EVAL_ENVS="${EVAL_ENVS:-128}"
CERT_EPISODES="${CERT_EPISODES:-1000}"
SWEEP_EPISODES="${SWEEP_EPISODES:-600}"
LOGROOT="logs/rl_games/zero_g_blade_insertion_uncertain"
OUT="artifacts/uncertain"
mkdir -p "$OUT" evidence

# Belief error in millimetres. The trained ceiling is 4 mm; everything past it is
# a measurement of where the policy stops working, not a certification.
SWEEP_MM=(0 1 2 4 6 8 10)
CERT_MM=4
EVAL_SEEDS=(1080 2080 3080)

arms=("aware:Isaac-ZeroG-Blade-Insertion-Uncertain" "blind:Isaac-ZeroG-Blade-Insertion-UncertainBlind")

stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then
  stages=(train certify sweep aggregate)
fi
has() { printf '%s\n' "${stages[@]}" | grep -qx "$1"; }

checkpoint_for() {
  ls -t "$LOGROOT"/"uncertain_$1_seed$SEED"/nn/*.pth 2>/dev/null | head -1
}

# ---------------------------------------------------------------- train
if has train; then
  for arm in "${arms[@]}"; do
    name="${arm%%:*}"
    task="${arm#*:}-v0"
    run="uncertain_${name}_seed${SEED}"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] SMOKE $name"
    "$PYTHON" scripts/train.py --headless --task "$task" --num_envs 32 --robustness_level 2 --smoke \
        --run_name "smoke_$run" > "$OUT/smoke_${name}.log" 2>&1
    if grep -qE "^Traceback" "$OUT/smoke_${name}.log"; then
      echo "[$(date +%H:%M:%S)] SMOKE FAILED for $name; refusing to burn GPU on it"
      grep -E "Error|Exception" "$OUT/smoke_${name}.log" | tail -3
      continue
    fi

    echo "[$(date +%H:%M:%S)] TRAIN $name  level 2, $EPOCHS epochs, $NUM_ENVS environments"
    "$PYTHON" scripts/train.py --headless \
        --task "$task" \
        --num_envs "$NUM_ENVS" \
        --seed "$SEED" \
        --robustness_level 2 \
        --max_iterations "$EPOCHS" \
        --run_name "$run" \
        > "$OUT/train_${name}.log" 2>&1
    echo "[$(date +%H:%M:%S)]   exit=$?"
    checkpoint=$(checkpoint_for "$name")
    if [ -z "$checkpoint" ]; then
      echo "[$(date +%H:%M:%S)] NO CHECKPOINT for $name"
    else
      echo "[$(date +%H:%M:%S)]   checkpoint: $checkpoint"
    fi
  done
fi

# ------------------------------------------------------- certify and sweep
evaluate() {
  local name="$1" task="$2" checkpoint="$3" bias="$4" stage="$5" seed="$6" episodes="$7"
  local tag
  tag=$(printf '%s_b%02d_s%s_seed%s' "$name" "$bias" "$stage" "$seed")
  "$PYTHON" scripts/play.py --headless \
      --task "${task}-Play-v0" \
      --checkpoint "$checkpoint" \
      --num_envs "$EVAL_ENVS" \
      --episodes "$episodes" \
      --robustness_level 2 \
      --curriculum_stage "$stage" \
      --belief_bias_mm "$bias" \
      --seed "$seed" \
      --report "$OUT/${tag}_play.json" \
      --episode_metrics "$OUT/${tag}.npz" \
      > "$OUT/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)]   $tag exit=$?"
}

for arm in "${arms[@]}"; do
  name="${arm%%:*}"
  task="${arm#*:}"
  checkpoint=$(checkpoint_for "$name")
  if [ -z "$checkpoint" ]; then
    echo "[$(date +%H:%M:%S)] no checkpoint for $name; skipping its evaluation"
    continue
  fi

  if has certify; then
    echo "[$(date +%H:%M:%S)] CERTIFY $name at ${CERT_MM} mm on three held-out seeds"
    for seed in "${EVAL_SEEDS[@]}"; do
      evaluate "$name" "$task" "$checkpoint" "$CERT_MM" 0 "$seed" "$CERT_EPISODES"
    done
  fi

  if has sweep; then
    echo "[$(date +%H:%M:%S)] SWEEP $name across ${SWEEP_MM[*]} mm"
    for bias in "${SWEEP_MM[@]}"; do
      [ "$bias" = "$CERT_MM" ] && continue  # already measured at stage 2 above
      for seed in "${EVAL_SEEDS[@]}"; do
        evaluate "$name" "$task" "$checkpoint" "$bias" 0 "$seed" "$SWEEP_EPISODES"
      done
    done
  fi
done

# ------------------------------------------------------------- aggregate
if has aggregate; then
  for arm in "${arms[@]}"; do
    name="${arm%%:*}"
    label=$([ "$name" = "aware" ] && echo "force-aware" || echo "force-blind")

    cert_rows=("$OUT/${name}_b0${CERT_MM}"_s0_seed*.npz)
    if [ -e "${cert_rows[0]}" ]; then
      "$PYTHON" scripts/aggregate_evaluation.py \
          --episodes "${cert_rows[@]}" \
          --output "evidence/uncertain_insertion_${name}_certification.json" \
          --title "Insertion under a 4 mm pose-belief error, ${label}" \
          --minimum_stage_success_rate 0.95 \
          --scope \
            "Simulation only. No result here was produced on real hardware." \
            "The actor never observes the true blade pose; it observes a belief displaced by a bias held constant for the episode." \
            "The grasp is a PhysX fixed joint standing in for an already-secured grasp. It is not learned grasping." \
            "One PPO training seed. The evaluation seeds are held out, but training repeatability is untested." \
          > "$OUT/aggregate_${name}_certification.log" 2>&1
      echo "[$(date +%H:%M:%S)] certification ${name} exit=$? -> evidence/uncertain_insertion_${name}_certification.json"
    fi

    sweep_rows=("$OUT/${name}"_b??_s0_seed*.npz)
    if [ -e "${sweep_rows[0]}" ]; then
      "$PYTHON" scripts/aggregate_evaluation.py \
          --episodes "${sweep_rows[@]}" \
          --output "evidence/uncertain_insertion_${name}_envelope.json" \
          --title "Success against pose-belief error, ${label}" \
          --scope \
            "MEASUREMENT, not certification. Belief error is swept past the 4 mm ceiling either policy trained on." \
            "One reset distance, the full-distance approach; the moved channel leaves no room for the nearer stages." \
            "Simulation only. One PPO training seed." \
          > "$OUT/aggregate_${name}_envelope.log" 2>&1
      echo "[$(date +%H:%M:%S)] envelope ${name} exit=$? -> evidence/uncertain_insertion_${name}_envelope.json"
    fi
  done
fi

echo "[$(date +%H:%M:%S)] DONE"
