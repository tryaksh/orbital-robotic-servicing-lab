#!/usr/bin/env bash
# Measure what the certified grapple skills cost and where they stop working.
#
# Certification says a policy works inside the distribution it trained on. It
# says nothing about interaction loads, and nothing about the axes the project's
# objective actually names: payload variation and friction. All three skills
# trained at robustness level 0, which holds blade mass fixed and disables
# friction and stiction randomization, so those axes are untested.
#
# Every run here is a MEASUREMENT, not a certification. Policies trained at
# level 0 and evaluated at level 2 or 3 are being run outside their training
# distribution on purpose, exactly like the existing
# evidence/rigid_grasp_l2_envelope_* reports, and the resulting files are
# labelled so they can never be read as promotion.
#
# Usage: scripts/measure_grapple_envelope.sh [skill ...]

set -u

PYTHON="C:/isaac-sim/python.bat"
EVAL_ENVS="${EVAL_ENVS:-128}"
EPISODES="${EPISODES:-600}"
SEED="${SEED:-7070}"
STAGE="${STAGE:-2}"
mkdir -p artifacts/envelope evidence

skills=("$@")
if [ ${#skills[@]} -eq 0 ]; then
  skills=(Grasp Extract Insert)
fi

# label:extra-play-arguments
probes=(
  "contact:--contact_metrics"
  "payload:--robustness_level 2"
  "friction:--robustness_level 3"
)

for skill in "${skills[@]}"; do
  lower=$(echo "$skill" | tr '[:upper:]' '[:lower:]')
  play="Isaac-ZeroG-Blade-GrapplePin-${skill}-Play-v0"
  # Newest checkpoint across every run of this skill, v1 or v2 or v3.
  checkpoint=$(ls -t logs/rl_games/*/grapple_${lower}_l0_seed70*/nn/*.pth 2>/dev/null | head -1)
  if [ -z "$checkpoint" ]; then
    echo "[envelope] no checkpoint for $skill; skipping"
    continue
  fi
  echo "[envelope] $skill <- $checkpoint"

  for probe in "${probes[@]}"; do
    label="${probe%%:*}"
    extra="${probe#*:}"
    out="artifacts/envelope/${lower}_${label}"
    # shellcheck disable=SC2086
    "$PYTHON" scripts/play.py --headless \
        --task "$play" \
        --checkpoint "$checkpoint" \
        --num_envs "$EVAL_ENVS" \
        --episodes "$EPISODES" \
        --curriculum_stage "$STAGE" \
        --seed "$SEED" \
        --episode_metrics "${out}.npz" \
        --report "${out}_play.json" \
        $extra \
        > "${out}.log" 2>&1
    echo "[envelope]   ${label} exit=$?"
  done

  rows=()
  for probe in "${probes[@]}"; do
    rows+=("artifacts/envelope/${lower}_${probe%%:*}.npz")
  done
  "$PYTHON" scripts/aggregate_evaluation.py \
      --episodes "${rows[@]}" \
      --output "evidence/grapple_${lower}_envelope.json" \
      --title "Head-on grapple-pin ${lower}: contact load and out-of-distribution envelope" \
      --scope \
        "MEASUREMENT, not certification. The policy trained at robustness level 0 and is run here at levels 2 and 3, outside its training distribution." \
        "Level 2 randomizes blade mass over 5 to 15 kg. Level 3 adds rail friction and 10 to 120 N breakaway stiction." \
        "Contact forces are a relative damage proxy against primitive geometry with no connector or chamfer, not an absolute hardware force budget." \
        "Simulation only. One PPO training seed." \
      > "artifacts/envelope/aggregate_${lower}.log" 2>&1
  echo "[envelope] aggregate ${lower} exit=$? -> evidence/grapple_${lower}_envelope.json"
done
echo "[envelope] DONE"
