#!/usr/bin/env bash
# The robot-carried relocation, and the measurement that decides how it has to
# be built.
#
# One stage per question, because each answers the next one's precondition:
#
#   scripts/run_robot_carried.sh passive    # can the parallel-jaw grip carry it
#   scripts/run_robot_carried.sh latched    # can a robot-side form lock carry it
#   scripts/run_robot_carried.sh sweep      # what rating does the lock need
#   scripts/run_robot_carried.sh certify    # multi-seed state batch
#   scripts/run_robot_carried.sh rgbd       # one full RGB-D end-to-end run
#
# ``passive`` is expected to fail and is not a failure of the session: it is the
# control that makes the latch a measured necessity rather than a convenience.
# Preserve its report.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/robotcarried"
mkdir -p "$OUT" evidence

GRASP_CKPT="${GRASP_CKPT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v6w65/nn/last_zero_g_blade_insertion_contact_ep_2400_rew__37.24023_.pth}"
EXTRACT_CKPT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v16w65/nn/last_zero_g_blade_insertion_contact_ep_9700_rew__176.34572_.pth}"
INSERT_CKPT="${INSERT_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v12w65/nn/last_zero_g_blade_insertion_contact_ep_7100_rew_-20.706831.pth}"

STATE_TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0"
VISION_TASK="Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
ENVS="${ENVS:-32}"
EPISODES="${EPISODES:-32}"

chain() {
  "$PYTHON" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 \
      --task "${TASK:-$STATE_TASK}" \
      --grasp_checkpoint "$GRASP_CKPT" --extract_checkpoint "$EXTRACT_CKPT" \
      --insert_checkpoint "$INSERT_CKPT" "$@"
}

stage="${1:-}"

case "$stage" in
  passive)
    # The control. No latch, no shuttle: the robot holds the module with the
    # pads alone and is asked to fly it to the next bay.
    echo "[$(date +%H:%M:%S)] PASSIVE robot-carried relocation, ${ENVS} envs"
    chain --num_envs "$ENVS" --episodes "$EPISODES" --seed "${SEED:-4070}" \
        --report "$OUT/passive_report.json" \
        --episode_metrics "$OUT/passive.npz" \
        --handoff_trace "$OUT/passive_trace.npz" \
        > "$OUT/passive.log" 2>&1
    echo "[$(date +%H:%M:%S)] passive exit=$?"
    ;;

  latched)
    echo "[$(date +%H:%M:%S)] LATCHED robot-carried relocation, ${ENVS} envs"
    chain --num_envs "$ENVS" --episodes "$EPISODES" --seed "${SEED:-4070}" \
        --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n "${LATCH_N:-600}" --latch_rated_torque_nm "${LATCH_NM:-30}" \
        --report "$OUT/latched_report.json" \
        --episode_metrics "$OUT/latched.npz" \
        --handoff_trace "$OUT/latched_trace.npz" \
        > "$OUT/latched.log" 2>&1
    echo "[$(date +%H:%M:%S)] latched exit=$?"
    ;;

  sweep)
    # What rating does the form lock actually need?
    #
    # The first latched run held the module -- grip error stayed at 12 to 37 mm
    # where the passive control lost it by 800 mm -- and still let the
    # tool-to-module transform move 0.4 m, which a fixed joint cannot do unless
    # it has broken. PhysX break thresholds are permanent, so a joint rated
    # below the transit's own reaction is a joint that is present for the first
    # second and absent afterwards. Sweep it, and report the rating the
    # workflow needs as a specification number rather than choosing one.
    for rating in "600 30" "3000 150" "20000 1000" "1000000 1000000"; do
      set -- $rating
      newtons="$1"; newtonmetres="$2"
      tag="n${newtons}_nm${newtonmetres}"
      echo "[$(date +%H:%M:%S)] SWEEP latch rating ${newtons} N / ${newtonmetres} N-m"
      chain --num_envs "${SWEEP_ENVS:-8}" --episodes "${SWEEP_EPISODES:-8}" --seed "${SEED:-4070}" \
          --latch_on_release --latch_joint_mode fixed \
          --latch_rated_force_n "$newtons" --latch_rated_torque_nm "$newtonmetres" \
          --report "$OUT/sweep_${tag}_report.json" \
          --handoff_trace "$OUT/sweep_${tag}_trace.npz" \
          > "$OUT/sweep_${tag}.log" 2>&1
      echo "[$(date +%H:%M:%S)]   exit=$?"
    done
    ;;

  smoke)
    # One environment, one pass, the demonstration path: it stops the moment the
    # workflow is judged instead of filling an episode budget, which is the
    # difference between a two-minute answer and a ten-minute one.
    echo "[$(date +%H:%M:%S)] SMOKE the latch hardware and its release"
    chain --num_envs 1 --steps "${STEPS:-3000}" --seed "${SEED:-4070}" \
        --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n "${LATCH_N:-1000000}" --latch_rated_torque_nm "${LATCH_NM:-1000000}" \
        --latch_position_stiffness_n_per_m "${MATING_K:-2500}" \
        --latch_rotation_stiffness_nm_per_rad "${MATING_KR:-10}" \
        --destination_channel_relief_m "${RELIEF:-0.0}" \
        --report "${REPORT:-$OUT/smoke_latched_report.json}" \
        --handoff_trace "${TRACE:-$OUT/smoke_latched_trace.npz}" \
        > "${LOG:-$OUT/smoke_latched.log}" 2>&1
    echo "[$(date +%H:%M:%S)] smoke exit=$?"
    ;;

  mating)
    # What compliance does the mating need, and does it remove the need to
    # widen the rack? Two questions, one grid, because they trade against each
    # other: a stiffer lock needs a wider channel and a softer one does not.
    for stiffness in ${MATING_K:-500 2500 10000}; do
      for relief in ${MATING_RELIEF:-0.0 0.00315}; do
        tag="k${stiffness}_r${relief}"
        echo "[$(date +%H:%M:%S)] MATING stiffness=${stiffness} N/m relief=${relief} m"
        chain --num_envs "${ENVS:-8}" --episodes "${EPISODES:-8}" --seed "${SEED:-4070}"             --latch_on_release --latch_joint_mode fixed             --latch_rated_force_n "${LATCH_N:-20000}" --latch_rated_torque_nm "${LATCH_NM:-1000}"             --latch_position_stiffness_n_per_m "$stiffness"             --destination_channel_relief_m "$relief"             --report "$OUT/mating_${tag}_report.json"             --episode_metrics "$OUT/mating_${tag}.npz"             > "$OUT/mating_${tag}.log" 2>&1
        echo "[$(date +%H:%M:%S)]   exit=$?"
      done
    done
    ;;

  certify)
    # The state batch. Three held-out seeds, pooled the way every other claim in
    # this repository is pooled, so the robot-carried chain gets a Wilson
    # interval rather than an anecdote.
    rows=()
    for seed in 4070 5070 6070; do
      out="$OUT/certify_seed${seed}"
      echo "[$(date +%H:%M:%S)] CERTIFY robot-carried relocation, seed ${seed}"
      chain --num_envs "${ENVS:-32}" --episodes "${EPISODES:-32}" --seed "$seed" \
          --latch_on_release --latch_joint_mode fixed \
          --latch_rated_force_n "${LATCH_N:-20000}" --latch_rated_torque_nm "${LATCH_NM:-1000}" \
          --report "${out}_report.json" --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      echo "[$(date +%H:%M:%S)]   exit=$?"
      rows+=("${out}.npz")
    done
    "$PYTHON" scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/workflow_robot_carried_relocate_certification.json" \
        --title "Robot-carried servicing workflow: relocation, bay 1 to bay 2" \
        --scope \
          "Simulation only. No result here was produced on real hardware." \
          "One continuous episode: learned capture, learned extraction, robot-carried transit on a visible robot-side form lock, guarded robot-driven insertion, release after settling." \
          "No world-mounted payload stage, no direct module pose write, and no teleport is active. The module is carried by the arm throughout." \
          "Capture and extraction are trained policies; the seat, the transit and the insertion are scripted and labelled as such." \
          "Success is the workflow's own condition re-checked after a 0.70 s settling window." \
        > "$OUT/aggregate_certify.log" 2>&1
    echo "[$(date +%H:%M:%S)] aggregate exit=$?"
    tail -6 "$OUT/aggregate_certify.log"
    ;;

  rgbd)
    # One full RGB-D end-to-end run, with the video the showcase needs.
    echo "[$(date +%H:%M:%S)] RGBD robot-carried relocation with video"
    chain --num_envs 1 --seed "${SEED:-4070}" --steps "${STEPS:-3600}" \
        --task "$VISION_TASK" --perception_backend fiducial_pnp \
        --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n "${LATCH_N:-20000}" --latch_rated_torque_nm "${LATCH_NM:-1000}" \
        --stable_lighting --inspection_view workcell \
        --video --video_dir "$OUT/video" --settle_steps 30 \
        --report "$OUT/rgbd_report.json" \
        --handoff_trace "$OUT/rgbd_trace.npz" \
        > "$OUT/rgbd.log" 2>&1
    echo "[$(date +%H:%M:%S)] rgbd exit=$?"
    ;;

  *)
    echo "usage: scripts/run_robot_carried.sh {passive|latched|sweep|smoke|mating|certify|rgbd}"
    exit 2
    ;;
esac
