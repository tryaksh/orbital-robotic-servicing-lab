#!/usr/bin/env bash
# Certify the chained servicing workflows, not just their parts.
#
# Three skills certified separately at 95.6%, 91.5% and 30.7% composed into zero
# working workflows the first time they were chained, and none of the three
# reasons was visible in any individual certification. A skill certification is
# therefore not evidence about the chain, and until this script has run the two
# workflow videos are demonstrations with n = 1.
#
# `run_workflow_demo.py --episodes` runs the same driver headless across many
# environments and writes one row per completed workflow in the format
# `aggregate_evaluation.py` already pools, so the chain is gated exactly the way
# a single skill is: pooled success rate, Wilson 95% interval, zero instability
# terminations, zero non-finite metrics.
#
# Success is the workflow's own condition re-checked after a 0.70 s settling
# window, not the instant a predicate fires. That is strictly harder than the
# skills' own criteria and it is what makes the install number honest: the
# insertion predicate fires correctly and the pin then relaxes a couple of
# hundredths of a radian in the pads, which trips a grip check while the module
# stays exactly where it was put. Both numbers are reported.
#
# Usage: scripts/certify_workflow.sh [remove|install ...]   (default: both)

set -u

PYTHON="C:/isaac-sim/python.bat"
ENVS="${ENVS:-64}"
EPISODES="${EPISODES:-192}"
SEEDS="${SEEDS:-4070 5070 6070}"
OUT="artifacts/workflow_cert"
mkdir -p "$OUT" evidence

# These defaults must always name the **promoted** set in CLAUDE.md, and they
# have twice drifted behind it. The rule that keeps them honest: a run that
# forgets to override a checkpoint variable must still measure the policy this
# project claims, never a superseded one. Capture v3 and extract v4 sat here
# after v5 and v13unsat were promoted, so any unattended re-run of this script
# would have re-measured the old chain and filed it as the current one.
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
GRASP_CKPT="${GRASP_CKPT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth}"
EXTRACT_CKPT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v13unsat/nn/last_zero_g_blade_insertion_contact_ep_5700_rew__148.17932_.pth}"
INSERT_CKPT="${INSERT_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth}"
TAG="${TAG:-}"

workflows=("$@")
if [ ${#workflows[@]} -eq 0 ]; then
  workflows=(remove install)
fi

for checkpoint in "$GRASP_CKPT" "$EXTRACT_CKPT" "$INSERT_CKPT"; do
  if [ ! -f "$checkpoint" ]; then
    echo "MISSING checkpoint: $checkpoint"
    exit 1
  fi
done

for workflow in "${workflows[@]}"; do
  # The curriculum stage is where the module starts, so it is not a free choice.
  # Removal begins with a module fully installed in the rack, which is stage 0.
  # Installation begins with one presented at the mouth, which is stage 2.
  case "$workflow" in
    remove)  stage=0 ;;
    install) stage=2 ;;
    full)    stage=0 ;;
  esac
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] CERTIFY CHAIN: $workflow (module starts at stage $stage)"
  echo "=============================================================="
  rows=()
  for seed in $SEEDS; do
    out="$OUT/${workflow}${TAG}_seed${seed}"
    "$PYTHON" scripts/run_workflow_demo.py --headless \
        --workflow "$workflow" \
        --curriculum_stage "$stage" \
        --grasp_checkpoint "$GRASP_CKPT" \
        --extract_checkpoint "$EXTRACT_CKPT" \
        --insert_checkpoint "$INSERT_CKPT" \
        --num_envs "$ENVS" \
        --episodes "$EPISODES" \
        --seed "$seed" \
        --report "${out}_report.json" \
        --episode_metrics "${out}.npz" \
        > "${out}.log" 2>&1
    status=$?
    echo "[$(date +%H:%M:%S)]   $workflow seed=$seed exit=$status"
    if [ "$status" -ne 0 ]; then
      tail -25 "${out}.log"
      exit 1
    fi
    rows+=("${out}.npz")
  done

  "$PYTHON" scripts/aggregate_evaluation.py \
      --episodes "${rows[@]}" \
      --output "evidence/workflow_${workflow}${TAG}_certification.json" \
      --title "Chained servicing workflow: ${workflow}" \
      --scope \
        "Simulation only. No result here was produced on real hardware." \
        "The module is held by physical pad-against-pin contact throughout. No fixed joint, no software fixture." \
        "Capture and the skill that follows it are trained policies. The one-second seating pause between them is scripted, and the transit in the full round trip is scripted; both are labelled in the report." \
        "Success is the workflow's own condition re-checked after a 0.70 s settling window, not the instant a predicate fired." \
        "One PPO training seed per skill. The evaluation seeds are held out, but training repeatability is untested." \
      > "$OUT/aggregate_${workflow}${TAG}.log" 2>&1
  echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/workflow_${workflow}${TAG}_certification.json"
  tail -5 "$OUT/aggregate_${workflow}${TAG}.log"
done

echo "[$(date +%H:%M:%S)] CHAIN CERTIFICATION DONE"
