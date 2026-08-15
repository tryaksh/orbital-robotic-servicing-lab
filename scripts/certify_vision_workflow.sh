#!/usr/bin/env bash
# Certify the servicing workflow driven by a camera, against the same workflow
# driven by the simulator's own answer.
#
# This is the comparison the whole project has been missing. Every certified
# policy here reads the module's pose out of the simulator, which has been
# recorded as the central weakness since 2026-08-10. Both arms below run the
# *same* certified checkpoints, on the same scene, through the same observation
# term, with the module displaced by an amount nothing in the observation
# reveals. The only difference is where its pose comes from:
#
#   oracle  -- the simulator, via pose_head_oracle_blend = 1.0
#   camera  -- a 64x64 RGB frame through mdp.ModulePoseHead
#
# so the gap between the two numbers is the cost of perception and cannot be
# anything else. Nothing is retrained to flatter the camera.
#
# Randomized in both arms: orbital sun intensity, angle, pitch, yaw and colour
# temperature; rack albedo, metallic and roughness; camera radiation noise; and
# the module's own pose.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
TASK=Isaac-ZeroG-Blade-GrappleVision-Workflow-v0
HEAD="${HEAD:-checkpoints/module_pose_head.pth}"
ENVS="${ENVS:-32}"
EPISODES="${EPISODES:-192}"
SEEDS="${SEEDS:-4070 5070 6070}"
OUT=artifacts/vision_cert
mkdir -p "$OUT" evidence

GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth"
EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/last_zero_g_blade_insertion_contact_ep_3500_rew_148.35918.pth"
INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"

for c in "$GRASP_CKPT" "$EXTRACT_CKPT" "$INSERT_CKPT" "$HEAD"; do
  if [ ! -f "$c" ]; then echo "MISSING $c"; exit 1; fi
done

say() { echo "[$(date +%H:%M:%S)] $*"; }

for arm in oracle camera; do
  if [ "$arm" = "oracle" ]; then
    extra=(--oracle)
    title="Vision servicing workflow, ORACLE arm: the module pose read from the simulator"
  else
    extra=(--pose_head_checkpoint "$HEAD")
    title="Vision servicing workflow, CAMERA arm: the module pose regressed from 64x64 RGB"
  fi
  rows=()
  for seed in $SEEDS; do
    out="$OUT/${arm}_seed${seed}"
    say "$arm seed=$seed"
    "$PYTHON" scripts/run_workflow_demo.py --headless \
        --task "$TASK" --workflow install --curriculum_stage 2 \
        --grasp_checkpoint "$GRASP_CKPT" \
        --extract_checkpoint "$EXTRACT_CKPT" \
        --insert_checkpoint "$INSERT_CKPT" \
        --num_envs "$ENVS" --episodes "$EPISODES" --seed "$seed" \
        "${extra[@]}" \
        --report "${out}_report.json" --episode_metrics "${out}.npz" \
        > "${out}.log" 2>&1
    say "  exit=$?"
    rows+=("${out}.npz")
  done

  "$PYTHON" scripts/aggregate_evaluation.py \
      --episodes "${rows[@]}" \
      --output "evidence/vision_workflow_${arm}_certification.json" \
      --title "$title" \
      --scope \
        "Simulation only. A rendered camera, not a calibrated real one." \
        "The module is held by physical pad-against-pin contact throughout. No fixed joint." \
        "Both arms run the identical certified checkpoints through the identical observation term. Nothing was retrained for either." \
        "The module is displaced per episode by an amount no observation reveals, so the image is the only source of its pose." \
        "Orbital lighting, rack albedo, and camera noise are randomized in both arms." \
        "One PPO training seed per skill and one training run for the pose head. Evaluation seeds are held out." \
      > "$OUT/aggregate_${arm}.log" 2>&1
  say "aggregate $arm -> evidence/vision_workflow_${arm}_certification.json"
  tail -4 "$OUT/aggregate_${arm}.log"
done

say "VISION WORKFLOW CERTIFICATION DONE"
