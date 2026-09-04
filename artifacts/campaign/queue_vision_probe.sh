#!/usr/bin/env bash
# After the base ladder, probe the RGB-D chain with whatever the noised extract
# fine-tune has reached. One seed, eight environments: this is a direction check,
# not a certificate. The published cohort is three seeds and comes later.
set -u
cd /d/6axis-space-robotics || exit 1
while [ ! -f artifacts/robustness64_baseladder/base_y_+6mm.npz ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] ladder landed; probing the RGB-D chain"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
NOISED=$(ls -t "$CKPT_ROOT/grapple_extract_l0_seed70_v19noised/nn/"last_*.pth 2>/dev/null | head -1)
if [ -z "$NOISED" ]; then echo "no noised checkpoint yet"; exit 1; fi
echo "[$(date +%H:%M:%S)] using $NOISED"
export EXTRACT_CKPT="$NOISED"
export TASK="Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
export CHAIN_EXTRA="--perception_backend fiducial_pnp --rack_retention"
export CERT_ENVS=8
mkdir -p artifacts/campaign/visionprobe
"C:/isaac-sim/python.bat" scripts/run_workflow_demo.py --headless \
    --workflow relocate --curriculum_stage 0 --task "$TASK" \
    --grasp_checkpoint "$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth" \
    --extract_checkpoint "$NOISED" \
    --insert_checkpoint "$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth" \
    --num_envs 8 --seed 4070 --steps 1900 \
    --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
    --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
    --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
    --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
    --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
    --report artifacts/campaign/visionprobe/probe_report.json \
    --episode_metrics artifacts/campaign/visionprobe/probe.npz \
    > artifacts/campaign/visionprobe/probe.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] probe exit=$rc"
