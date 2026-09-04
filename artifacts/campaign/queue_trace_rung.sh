#!/usr/bin/env bash
# One traced rung, to turn a hypothesis into a measurement.
#
# At 6 mm of rail stop error the chain loses 40 of 64 episodes, and every one of
# them is *extracted* -- past the extracted plane, still gripped at a normal
# tool-to-pin offset -- and fails the settling condition instead, carrying 16 to
# 30 mm/s against a derived 14.29 mm/s limit. The natural explanation is that a
# pull off the bay's centre line applies a moment to two flat pads on a pin,
# which is the one thing this interface cannot resist and is the project's
# founding measurement.
#
# The episode rows cannot confirm it: they record the speed and not the vector.
# `--handoff_trace` records the vector. If the residual velocity is transverse to
# the pull axis, the explanation holds; if it is along the pull, it does not and
# something else is leaving the module moving.
set -u
cd /d/6axis-space-robotics || exit 1
until grep -q "cleanup done" artifacts/campaign/cleanup.log 2>/dev/null; do sleep 120; done
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
mkdir -p artifacts/campaign/tracedrung
echo "[$(date +%H:%M:%S)] 6 mm rail stop error, with the handoff trace"
"C:/isaac-sim/python.bat" scripts/run_workflow_demo.py --headless \
    --workflow relocate --curriculum_stage 0 \
    --task Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0 \
    --grasp_checkpoint "$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth" \
    --extract_checkpoint "$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth" \
    --insert_checkpoint "$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth" \
    --num_envs 64 --episodes 64 --seed 4070 --steps 6000 \
    --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
    --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
    --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
    --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
    --robot_base_y 0.006 \
    --report artifacts/campaign/tracedrung/base_y_6mm_report.json \
    --episode_metrics artifacts/campaign/tracedrung/base_y_6mm.npz \
    --handoff_trace artifacts/campaign/tracedrung/base_y_6mm_trace.npz \
    > artifacts/campaign/tracedrung/base_y_6mm.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] traced rung exit=$rc"
echo "[$(date +%H:%M:%S)] traced rung done"
