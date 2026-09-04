#!/usr/bin/env bash
# One sequential evaluation supervisor. Replaces nine waiting queues.
#
# **Why this exists.** The overnight campaign ran twelve queue scripts at once,
# each parked in an `until ... sleep` loop, each with a child shell. At 06:25 the
# machine ran out of whatever Git-Bash's fork emulation needs and everything died
# together: `queue_noised_skill_cert.sh: fork: retry: Resource temporarily
# unavailable`, three trainings gone at 95%, 99.5% and 47% of their epochs, and
# not one evaluation stage started. Eighteen sleeping shells was the design
# defect. Stages that must run in order should be lines in one script, not
# separate processes waiting on each other's log files.
#
# Order is by what the paper needs, not by what was queued first. The RGB-D
# cohorts are the submission gate and spent the night blocked behind a
# certification that is not on the critical path; they go first now.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED="$ROOT/grapple_extract_l0_seed70_v19noised/nn/last_zero_g_blade_insertion_contact_ep_14600_rew_166.19054.pth"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

say () { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- the gate
# Seeds 5070 and 6070 for each arm; 4070 survived the crash on disk.
finish_arm () {
  tag="$1"; title="$2"; extract="$3"; note="$4"; shift 4
  for seed in 5070 6070; do
    out="artifacts/campaign/rgbdcohorts/${tag}_seed${seed}"
    if [ -f "${out}.npz" ]; then say "  $tag seed $seed already on disk; skipping"; continue; fi
    "$PY" scripts/run_workflow_demo.py --headless \
        --workflow relocate --curriculum_stage 0 \
        --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
        --grasp_checkpoint "$G" --extract_checkpoint "$extract" --insert_checkpoint "$I" \
        --num_envs 8 --seed "$seed" --steps 1900 \
        --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
        --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
        --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
        --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
        "$@" \
        --report "${out}_report.json" --episode_metrics "${out}.npz" \
        > "${out}.log" 2>&1
    rc=$?
    say "  $tag seed $seed exit=$rc"
    ERR=$(./.venv/Scripts/python.exe -c "import json,sys; print(json.load(open(sys.argv[1])).get('error') or '')" "${out}_report.json" 2>/dev/null)
    if [ -n "$ERR" ] || [ ! -f "${out}.npz" ]; then
      say "  $tag seed $seed FAILED: ${ERR:-no episode metrics}"
      return 1
    fi
  done
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py \
      --episodes "artifacts/campaign/rgbdcohorts/${tag}_seed4070.npz" \
                 "artifacts/campaign/rgbdcohorts/${tag}_seed5070.npz" \
                 "artifacts/campaign/rgbdcohorts/${tag}_seed6070.npz" \
      --output "evidence/workflow_robot_carried_vision_${tag}_certification.json" \
      --title "$title" --scope "$SCOPE_1" "$SCOPE_2" "$note" "$SCOPE_3" \
      > "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log" 2>&1
  rc=$?
  say "$tag aggregate exit=$rc"
  tail -6 "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log"
}

say "STAGE 1/3  the submission gate: two seeds for each RGB-D arm"

finish_arm noised_extract \
  "RGB-D chain with extraction trained on the estimator's error" "$NOISED" \
  "One change from the published camera-driven cohort: the extraction policy resumed the certified v18pin checkpoint on a task whose module-derived observations carry the deployed estimator's certified residual, sample-and-hold and miss rate. Capture and the seating controller are unchanged."

finish_arm kinematic_velocity \
  "RGB-D chain with the module velocity taken from the robot rather than the cameras" "$E" \
  "One change from the published camera-driven cohort, and no retrain: every policy is the published checkpoint, and the module-velocity channel reports zero before capture and the wrist's own velocity after it. That is encoder and forward-kinematics information; no module state is read." \
  --module_velocity_source kinematics

finish_arm noised_extract_kinematic_leadin \
  "RGB-D chain, best available configuration: trained on the estimator's error, velocity from the robot, guard on the flare's catch" "$NOISED" \
  "Three changes from the published camera-driven cohort, each of which is measured alone in its own certification: the retrained extraction policy, the kinematic velocity channel, and the guarded advance admitting on the entry flare's catch rather than on the estimator's own noise bound." \
  --module_velocity_source kinematics --fiducial_guard_bounds lead_in

say "STAGE 2/3  the datum-pair perception certificate"
mkdir -p artifacts/campaign/datumpair datasets
"$PY" scripts/collect_grapple_vision.py \
    --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Collect-v0 \
    --output datasets/fiducial_rgbd_datum_pair_seed286.npz \
    --samples 1024 --num_envs 16 --seed 286 \
    --rgb_source raw --pose_distribution workflow_envelope \
    > artifacts/campaign/datumpair/collect.log 2>&1
rc=$?
say "collect exit=$rc"
if [ -f datasets/fiducial_rgbd_datum_pair_seed286.npz ]; then
  "$PY" scripts/certify_fiducial_perception.py \
      --dataset datasets/fiducial_rgbd_datum_pair_seed286.npz \
      --report evidence/fiducial_rgbd_datum_pair_seed286.json \
      > artifacts/campaign/datumpair/certify.log 2>&1
  rc=$?
  say "certify exit=$rc"
  tail -12 artifacts/campaign/datumpair/certify.log
fi

say "STAGE 3/3  the noised extraction skill certification"
mkdir -p artifacts/campaign/noisedcert
for arm in noised clean; do
  case "$arm" in
    noised) TASK="Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-Play-v0" ;;
    clean)  TASK="Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0" ;;
  esac
  rows=()
  for seed in 1070 2070 3070; do
    for stage in 0 1 2; do
      out="artifacts/campaign/noisedcert/${arm}_s${stage}_seed${seed}"
      "$PY" scripts/play.py --headless --task "$TASK" --checkpoint "$NOISED" \
          --num_envs 128 --episodes 512 --curriculum_stage "$stage" --seed "$seed" \
          --episode_metrics "${out}.npz" > "${out}.log" 2>&1
      rc=$?
      say "  $arm stage=$stage seed=$seed exit=$rc"
      [ -f "${out}.npz" ] && rows+=("${out}.npz")
    done
  done
  if [ "${#rows[@]}" -gt 0 ]; then
    ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/grapple_extract_v19noised_${arm}_certification.json" \
        --title "Extraction skill retrained on the estimator's error, scored on the ${arm} task" \
        --scope "$SCOPE_1" \
          "The checkpoint resumed the certified v18pin weights on a task whose module-derived observations carry the deployed estimator's certified residual, sample-and-hold and miss rate." \
        > "artifacts/campaign/noisedcert/aggregate_${arm}.log" 2>&1
    rc=$?
    say "$arm aggregate exit=$rc"
    tail -5 "artifacts/campaign/noisedcert/aggregate_${arm}.log"
  fi
done

say "evaluation supervisor done"
