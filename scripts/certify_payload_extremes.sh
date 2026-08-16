#!/usr/bin/env bash
# The payload envelope, at enough episodes to mean something.
#
# A first pass at 96 episodes a point gave 59/39/73/42/75/57% across 2-80 kg:
# non-monotonic with 36-point swings, which is variance rather than a mass
# effect, and publishing it as an envelope would have been an overclaim. This
# runs the two extremes at the same power as every other certification here --
# three held-out seeds, 576 workflows -- so each can be compared against the
# certified 80.38% at the nominal 10 kg.

set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth"
E="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/last_zero_g_blade_insertion_contact_ep_3500_rew_148.35918.pth"
I="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"
OUT=artifacts/payload_cert
mkdir -p "$OUT" evidence

for kg in 2 80; do
  rows=()
  for seed in 4070 5070 6070; do
    out="$OUT/kg${kg}_seed${seed}"
    echo "[$(date +%H:%M:%S)] ${kg} kg seed=${seed}"
    "$PYTHON" scripts/run_workflow_demo.py --headless \
        --task Isaac-ZeroG-Blade-GrappleVision-Workflow-v0 --workflow install --curriculum_stage 2 \
        --grasp_checkpoint "$G" --extract_checkpoint "$E" --insert_checkpoint "$I" \
        --num_envs 32 --episodes 192 --seed "$seed" \
        --pose_head_checkpoint checkpoints/module_pose_head.pth \
        --module_mass_kg "$kg" \
        --report "${out}_report.json" --episode_metrics "${out}.npz" > "${out}.log" 2>&1
    echo "[$(date +%H:%M:%S)]   exit=$?"
    rows+=("${out}.npz")
  done
  "$PYTHON" scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
      --output "evidence/vision_workflow_payload_${kg}kg_certification.json" \
      --title "Vision servicing workflow at a ${kg} kg module" \
      --scope \
        "Simulation only. Nothing retrained: the same certified policies and the same pose head." \
        "The module is held by pad-against-pin contact throughout, so inertia loads the grip directly." \
        "Compare against the certified 80.38% at the nominal 10 kg module." \
      > "$OUT/aggregate_${kg}.log" 2>&1
  echo "[$(date +%H:%M:%S)] aggregate ${kg} kg exit=$?"
done
echo "[$(date +%H:%M:%S)] PAYLOAD CERTIFICATION DONE"
