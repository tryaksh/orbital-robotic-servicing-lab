#!/usr/bin/env bash
# The evaluation slot, after the factorial: what the seating claim rests on.
#
# Four stages, in one script as consecutive lines. The 06:25 fork exhaustion was
# caused by twelve queue scripts each parked in a sleep loop, not by the load, so
# stages that must run in order are lines here rather than processes waiting on
# each other's logs.
#
#   1  the guarded advance on the force workflow task, in the unrelieved bay
#   2  the same three checkpoints with `--insert_controller policy`, both halves
#   3  the factorial cell base_000, which a commit rejected mid-flight
#   4  the camera-driven chain on the finished noised capture
#
# **Stages 1 and 2 are one flag apart and that is the whole design.** The
# 2026-09-04 verification compared a force-feedback policy in a 15.678 mm bay
# against a guarded advance published at 11.065 mm, on a different task, and then
# died before writing the comparison. Here both arms run the same task, the same
# three checkpoints, the same three held-out seeds and the same unrelieved bay;
# the only difference is whether the seating phase steps the policy. A seating
# claim that cannot survive that design should not be made.
#
# Stage 2's skill half needs no bay setting: `play.py` has no relief flag, so the
# insert task is always the 11.065 mm design point the rack was derived at.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED="$ROOT/grapple_extract_l0_seed70_v19noised/nn/last_zero_g_blade_insertion_contact_ep_14600_rew_166.19054.pth"
FORCE="$ROOT/grapple_insert_l0_seed70_v33force/nn/last_zero_g_blade_insertion_contact_ep_3000_rew_98.33571.pth"
FORCE_TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflowForce-v0"
GUARDED="evidence/workflow_robot_carried_insert_v33force_c11065_chain_guarded_certification.json"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

say () { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -f "$FORCE" ]; then say "no force checkpoint at $FORCE; refusing to start"; exit 1; fi

# ---------------------------------------------------------------------------
say "STAGE 1/4  guarded advance on the force task, unrelieved bay, three seeds"
# ---------------------------------------------------------------------------
# The paired control. Same task, same policy set -- the force checkpoint is
# loaded and never stepped, exactly as the seating checkpoint is in every other
# guarded run -- so the policy-set hash matches stage 2's and the two reports are
# a pair rather than two measurements that happen to share seeds.
TASK="$FORCE_TASK" INSERT_CKPT="$FORCE" RELIEF=0.0 \
CERT_TAG="insert_v33force_c11065_chain_guarded" \
CERT_TITLE="Robot-carried relocation, scripted guarded advance, force task, channel throat 11.065 mm per side" \
  bash scripts/run_robot_carried.sh certify
rc=$?
say "  stage 1 exit=$rc -> $GUARDED"

# ---------------------------------------------------------------------------
say "STAGE 2/4  the force-feedback seating policy, skill half and chain half"
# ---------------------------------------------------------------------------
# `InsertForce-Play-v0` raised TypeError at construction until today, so the
# skill half of this has never run. The chain half runs against stage 1.
CKPT="$FORCE" TAG=insert_v33force_c11065 \
PLAY_TASK="Isaac-ZeroG-Blade-GrapplePin-InsertForce-Play-v0" \
CHAIN_TASK="$FORCE_TASK" \
BASELINE="$GUARDED" \
RELIEF=0.0 \
  bash scripts/verify_insert_skill.sh
rc=$?
say "  stage 2 exit=$rc"
say "  skill    evidence/grapple_insert_v33force_c11065_certification.json"
say "  chain    evidence/workflow_robot_carried_insert_v33force_c11065_chain_policy_certification.json"
say "  decision evidence/seating_controller_head_to_head.json"

# ---------------------------------------------------------------------------
say "STAGE 3/4  factorial cell base_000, re-run so its three seeds share a commit"
# ---------------------------------------------------------------------------
OUT=artifacts/campaign/factorial
mkdir -p "$OUT"
rows=()
for seed in 4070 5070 6070; do
  out="$OUT/base_000_seed${seed}"
  "$PY" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 \
      --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
      --grasp_checkpoint "$G" --extract_checkpoint "$E" --insert_checkpoint "$I" \
      --num_envs 8 --seed "$seed" --steps 1900 \
      --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
      --report "${out}_report.json" --episode_metrics "${out}.npz" \
      > "${out}.log" 2>&1
  rc=$?
  say "  base_000 seed $seed exit=$rc"
  [ -f "${out}.npz" ] && rows+=("${out}.npz")
done
if [ "${#rows[@]}" -eq 3 ]; then
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
      --output "evidence/workflow_robot_carried_vision_factorial_base_000_certification.json" \
      --title "RGB-D chain, factorial cell base_000" \
      --scope "$SCOPE_1" "$SCOPE_2" \
        "One cell of the 2x2x2 over the retrained extraction, the kinematic velocity channel and the lead-in guard bound. Every other term is the published camera-driven configuration." \
        "$SCOPE_3" \
      > "$OUT/aggregate_base_000.log" 2>&1
  rc=$?
  say "  base_000 aggregate exit=$rc"
  tail -4 "$OUT/aggregate_base_000.log"
else
  say "  base_000: only ${#rows[@]} of 3 seeds produced episodes; not aggregating"
fi

# ---------------------------------------------------------------------------
say "STAGE 4/4  the camera-driven chain on the finished noised capture"
# ---------------------------------------------------------------------------
# The published camera-driven arm is 17/24 with the *old* capture policy. This
# changes one term -- the capture checkpoint -- and nothing else, so the
# difference is attributable. The bay stays at the shipped relief on purpose: a
# capture change and a 4.6 mm geometry change must not be quoted as one number.
waited=0
until ls "$ROOT/grapple_grasp_l0_seed70_v9noised/nn/"*ep_5100_*.pth >/dev/null 2>&1; do
  sleep 300
  waited=$((waited + 300))
  if [ "$waited" -gt 14400 ]; then
    say "no ep_5100 noised capture after four hours; stopping before stage 4"
    say "evaluation supervisor 2 done"
    exit 0
  fi
done
NCAP=$(ls -t "$ROOT/grapple_grasp_l0_seed70_v9noised/nn/"*ep_5100_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
if [ -z "$NCAP" ]; then
  say "ep_5100 exists only under the double-underscore name; stopping"
  say "evaluation supervisor 2 done"
  exit 0
fi
say "  noised capture $(basename "$NCAP")"
OUT2=artifacts/campaign/noisedchain
mkdir -p "$OUT2"
rows=()
for seed in 4070 5070 6070; do
  out="$OUT2/noisedcap_seed${seed}"
  "$PY" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 \
      --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
      --grasp_checkpoint "$NCAP" --extract_checkpoint "$NOISED" --insert_checkpoint "$I" \
      --num_envs 8 --seed "$seed" --steps 1900 \
      --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
      --module_velocity_source kinematics --fiducial_guard_bounds lead_in \
      --report "${out}_report.json" --episode_metrics "${out}.npz" \
      > "${out}.log" 2>&1
  rc=$?
  say "  noised capture chain seed $seed exit=$rc"
  [ -f "${out}.npz" ] && rows+=("${out}.npz")
done
if [ "${#rows[@]}" -eq 3 ]; then
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
      --output "evidence/workflow_robot_carried_vision_noised_capture_extract_kinematic_leadin_certification.json" \
      --title "RGB-D chain, both skills trained on the estimator's error, kinematic velocity, lead-in guard" \
      --scope "$SCOPE_1" "$SCOPE_2" \
        "The published camera-driven arm with one term changed: the capture policy is the noised fine-tune rather than the nominal-state one. Extraction, the velocity channel, the guard bound and the bay are unchanged." \
        "$SCOPE_3" \
      > "$OUT2/aggregate_noisedcap.log" 2>&1
  rc=$?
  say "  noised capture chain aggregate exit=$rc"
  tail -4 "$OUT2/aggregate_noisedcap.log"
else
  say "  noised capture chain: only ${#rows[@]} of 3 seeds produced episodes; not aggregating"
fi

say "evaluation supervisor 2 done"
