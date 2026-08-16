#!/usr/bin/env bash
# What payload can this interface actually service?
#
# The specification has to state a mass range, and the only mass sweep this
# project owns was run on the *fixed-joint* task, where the module is welded to
# the tool and mass was measured as very nearly vacuous -- 100% success from 1 to
# 50 kg. That result was correctly recorded as weakening its own claim.
#
# Held by pad-against-pin contact it should be a different story, because
# inertia levers the grip and nothing in zero gravity damps it. This sweeps the
# certified install workflow, camera in the loop, with no retraining anywhere,
# so the curve is the interface's envelope rather than a training artefact.

set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth"
E="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/last_zero_g_blade_insertion_contact_ep_3500_rew_148.35918.pth"
I="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"

for kg in 2 5 10 20 40 80; do
  echo "[$(date +%H:%M:%S)] module mass ${kg} kg"
  "$PYTHON" scripts/run_workflow_demo.py --headless \
      --task Isaac-ZeroG-Blade-GrappleVision-Workflow-v0 --workflow install --curriculum_stage 2 \
      --grasp_checkpoint "$G" --extract_checkpoint "$E" --insert_checkpoint "$I" \
      --num_envs 32 --episodes 96 --seed 4070 \
      --pose_head_checkpoint checkpoints/module_pose_head.pth \
      --module_mass_kg "$kg" \
      --report "artifacts/mass/mass_${kg}kg.json" > "artifacts/mass/mass_${kg}kg.log" 2>&1
  echo "[$(date +%H:%M:%S)]   exit=$?"
done
echo "[$(date +%H:%M:%S)] PAYLOAD SWEEP DONE"
