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
#   scripts/run_robot_carried.sh rail       # the robot on a lateral rail
#   scripts/run_robot_carried.sh rgbd       # one full RGB-D end-to-end run
#   scripts/run_robot_carried.sh follower   # the same run on the old controller
#
# ``passive`` is expected to fail and is not a failure of the session: it is the
# control that makes the latch a measured necessity rather than a convenience.
# Preserve its report.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/robotcarried"
mkdir -p "$OUT" evidence

# **These three are the set that produced the published rate, and they are not
# a preference.** Read them out of the certification rather than trusting this
# comment: `evidence/robot_carried_full_chain_pin.json` records all three paths
# and their SHA-256, and the pooled report's `policy_set_sha256`
# (3D299D01AEDC8ED2770FFA29DF5F3659C3132423A28B53E4D516B4513072CD95) is the
# hash of that set.
#
# Corrected 2026-08-25. These defaults had been left on the superseded w65 set
# -- grasp v6w65, extract v16w65, insert v12w65 -- two promotions behind the
# checkpoints the 97.92% was measured on, so `run_robot_carried.sh certify` as
# documented did not reproduce the number it was documented as reproducing.
# `scripts/promote_checkpoints.py` exists to stop exactly this and did not cover
# this file; it does now. `src/zero_g_blade_swap/service/presets.py` is still on
# the w65 set and moving it would move a published service number, so it is a
# task rather than an edit: docs/NEXT_WORK.md T7.
#
# The extract filename is the single-underscore form on purpose. Epoch 12600
# exists under two rl-games naming conventions whose weights are byte-identical
# but whose file hashes are not, and the certification recorded this one.
GRASP_CKPT="${GRASP_CKPT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth}"
EXTRACT_CKPT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth}"
# Loaded, and never stepped: the seating phase is the scripted guarded advance.
# It stays in the set because the certification loaded it and the policy-set
# hash includes it, so dropping it would change the hash without changing a
# single action. `--insert_controller policy` is what makes it act.
INSERT_CKPT="${INSERT_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth}"

STATE_TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0"
VISION_TASK="Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
ENVS="${ENVS:-32}"
EPISODES="${EPISODES:-32}"
CERT_ENVS="${CERT_ENVS:-8}"

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
        --latch_position_stiffness_n_per_m "${MATING_K:-40000}" \
        --latch_rotation_stiffness_nm_per_rad "${MATING_KR:-20000}" \
        --destination_channel_relief_m "${RELIEF:-0.0046125}" \
        --mating_mode "${MATING_MODE:-compliant}" \
        --mating_force_cap_n "${MATING_CAP:-1000}" \
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
        --latch_position_stiffness_n_per_m "${MATING_K:-40000}" \
        --latch_rotation_stiffness_nm_per_rad "${MATING_KR:-20000}" \
        --destination_channel_relief_m "${RELIEF:-0.0046125}" \
        --mating_mode "${MATING_MODE:-compliant}" \
        --mating_force_cap_n "${MATING_CAP:-1000}" \
        ${CHAIN_EXTRA:-} \
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
      for relief in ${MATING_RELIEF:-0.004 0.008 0.012 0.016}; do
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
    # interval rather than an anecdote. These are bounded fixed cohorts, not
    # timeout-collected episodes: enabling the timeout reset path changes the
    # long-transit physics and produced non-finite rows on the same seed.
    # **With the rail.** This stage used to omit --robot_rail_on_relocation,
    # which certified a configuration the project's own measurements say does not
    # work: without a rail the arm has to translate the bay pitch at the retreat
    # depth, where its realised authority is 0.72, and the destination squaring
    # leg has never converged. A pooled number for that is a pooled number about
    # the wrong workcell.
    # CERT_TAG names the run so a re-certification cannot overwrite the report
    # it is supposed to be compared against. The default is the *before*, which
    # is preserved evidence, so a bare re-run of this stage would have replaced
    # the baseline with the result.
    CERT_TAG="${CERT_TAG:-relocate}"
    CERT_TITLE="${CERT_TITLE:-Robot-carried servicing workflow: relocation, bay 1 to bay 2}"
    # The comment above says a bare re-run would replace the baseline it is
    # supposed to be compared against. Saying so did not stop it being possible,
    # so this stops it: an aggregate that would overwrite an existing report
    # aborts before a single GPU-hour is spent, rather than after.
    #
    # Rule 6 -- failed and superseded results stay in evidence/, labelled -- is
    # only enforceable if the files survive, and the two preserved befores are
    # what make the 97.92% mean anything.
    CERT_OUT="evidence/workflow_robot_carried_${CERT_TAG}_certification.json"
    if [ -e "$CERT_OUT" ] && [ "${FORCE_OVERWRITE:-}" != "1" ]; then
      echo "REFUSING: $CERT_OUT already exists."
      echo "  It is preserved evidence, and this stage would replace it."
      echo "  Re-run with a new CERT_TAG, or FORCE_OVERWRITE=1 if replacing it is the intent."
      exit 3
    fi
    rows=()
    for seed in ${SEEDS:-4070 5070 6070}; do
      out="$OUT/certify_${CERT_TAG}_seed${seed}"
      echo "[$(date +%H:%M:%S)] CERTIFY robot-carried relocation, seed ${seed}"
      chain --num_envs "$CERT_ENVS" --seed "$seed" \
          --steps "${STEPS:-1900}" \
          --robot_rail_on_relocation \
          --latch_on_release --latch_joint_mode fixed \
          --latch_rated_force_n "${LATCH_N:-20000}" --latch_rated_torque_nm "${LATCH_NM:-1000}" \
        --latch_position_stiffness_n_per_m "${MATING_K:-40000}" \
        --latch_rotation_stiffness_nm_per_rad "${MATING_KR:-20000}" \
        --destination_channel_relief_m "${RELIEF:-0.0046125}" \
          --mating_mode "${MATING_MODE:-compliant}" \
          --mating_force_cap_n "${MATING_CAP:-1000}" \
          --release_sequence "${RELEASE_SEQUENCE:-simultaneous}" \
          ${CHAIN_EXTRA:-} \
          --report "${out}_report.json" --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      echo "[$(date +%H:%M:%S)]   exit=$?"
      rows+=("${out}.npz")
    done
    "$PYTHON" scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/workflow_robot_carried_${CERT_TAG}_certification.json" \
        --title "$CERT_TITLE" \
        --scope \
          "Simulation only. No result here was produced on real hardware." \
          "One continuous episode: learned capture, learned extraction, robot-carried transit on a visible robot-side form lock, guarded robot-driven insertion, release after settling." \
          "No world-mounted payload stage, no direct module pose write, and no teleport is active. The module is carried by the arm throughout." \
          "Capture and extraction are trained policies; the seat, the transit and the insertion are scripted and labelled as such." \
          "The transit legs are commanded from a solved inverse kinematics through actuator targets; the robot rides a lateral rail whose own load path is not modelled." \
          "Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s free-module recheck." \
        > "$OUT/aggregate_certify_${CERT_TAG}.log" 2>&1
    echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/workflow_robot_carried_${CERT_TAG}_certification.json"
    tail -6 "$OUT/aggregate_certify_${CERT_TAG}.log"
    ;;

  rail)
    # The robot on a lateral rail, which is what the crossing leg needs.
    #
    # The arm cannot hold the module square while translating the bay pitch at
    # the retreat depth: the differential IK's realised authority there is 0.72
    # and the squaring leg that follows has never converged in any run this
    # branch recorded. Moving the base back recovers the authority and loses the
    # capture policy, measured at both 100 mm and 50 mm. A rail loses neither,
    # because parked opposite a bay the arm's configuration is the one it has at
    # bay 1. See scripts/check_workcell_geometry.py and section 6a of
    # docs/service_interface_spec.md.
    #
    # The rail carries the ROBOT. It is not --base_rail_on_relocation, which
    # hands the module to a world-mounted shuttle and is kept only as a
    # labelled historical baseline.
    echo "[$(date +%H:%M:%S)] RAIL robot-carried relocation, ${ENVS:-1} envs"
    chain --num_envs "${ENVS:-1}" --steps "${STEPS:-6500}" --seed "${SEED:-4070}" \
        --robot_rail_on_relocation \
        --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n "${LATCH_N:-1000000}" --latch_rated_torque_nm "${LATCH_NM:-1000000}" \
        --latch_position_stiffness_n_per_m "${MATING_K:-40000}" \
        --latch_rotation_stiffness_nm_per_rad "${MATING_KR:-20000}" \
        --destination_channel_relief_m "${RELIEF:-0.0046125}" \
        --mating_mode "${MATING_MODE:-compliant}" \
        --mating_force_cap_n "${MATING_CAP:-1000}" \
        --report "${REPORT:-$OUT/rail_report.json}" \
        --handoff_trace "${TRACE:-$OUT/rail_trace.npz}" \
        > "${LOG:-$OUT/rail.log}" 2>&1
    echo "[$(date +%H:%M:%S)] rail exit=$?"
    ;;

  follower)
    # The control for the solved-IK legs: the same run, same seed, same rail, on
    # IsaacLab's relative-mode differential IK. Kept as a stage rather than as a
    # remembered environment variable, because "the new controller is better" is
    # a claim and a claim needs its counterfactual next to it in the evidence.
    echo "[$(date +%H:%M:%S)] FOLLOWER control, relative-mode differential IK, ${ENVS:-1} envs"
    TRANSIT_SOLVED_IK=0 chain --num_envs "${ENVS:-1}" --steps "${STEPS:-6500}" --seed "${SEED:-4070}" \
        --robot_rail_on_relocation \
        --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n "${LATCH_N:-1000000}" --latch_rated_torque_nm "${LATCH_NM:-1000000}" \
        --latch_position_stiffness_n_per_m "${MATING_K:-40000}" \
        --latch_rotation_stiffness_nm_per_rad "${MATING_KR:-20000}" \
        --destination_channel_relief_m "${RELIEF:-0.0046125}" \
        --mating_mode "${MATING_MODE:-compliant}" \
        --mating_force_cap_n "${MATING_CAP:-1000}" \
        --report "${REPORT:-$OUT/follower_report.json}" \
        --handoff_trace "${TRACE:-$OUT/follower_trace.npz}" \
        > "${LOG:-$OUT/follower.log}" 2>&1
    echo "[$(date +%H:%M:%S)] follower exit=$?"
    ;;

  rgbd)
    # One full RGB-D end-to-end run, with the video the showcase needs.
    #
    # Lighting randomization and video recording are exclusive: the recorder
    # needs a stable exposure and the evidence needs the randomization the
    # perception was certified under. Both are switches, so the showcase runs
    # this stage twice -- once with ``STABLE_LIGHTING= VIDEO=`` for the
    # evidence, and once with the defaults for the video.
    echo "[$(date +%H:%M:%S)] RGBD robot-carried relocation with video"
    chain --num_envs 1 --seed "${SEED:-4070}" --steps "${STEPS:-1900}" \
        --task "$VISION_TASK" --perception_backend fiducial_pnp \
        --robot_rail_on_relocation \
        --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n "${LATCH_N:-20000}" --latch_rated_torque_nm "${LATCH_NM:-1000}" \
        --latch_position_stiffness_n_per_m "${MATING_K:-40000}" \
        --latch_rotation_stiffness_nm_per_rad "${MATING_KR:-20000}" \
        --destination_channel_relief_m "${RELIEF:-0.0046125}" \
        --mating_mode "${MATING_MODE:-compliant}" \
        --mating_force_cap_n "${MATING_CAP:-1000}" \
        --release_sequence "${RELEASE_SEQUENCE:-simultaneous}" \
        ${CHAIN_EXTRA:-} \
        ${STABLE_LIGHTING---stable_lighting} --inspection_view workcell \
        ${VIDEO---video --video_dir "$OUT/video"} \
        --report "${REPORT:-$OUT/rgbd_report.json}" \
        --handoff_trace "${TRACE:-$OUT/rgbd_trace.npz}" \
        > "${LOG:-$OUT/rgbd.log}" 2>&1
    echo "[$(date +%H:%M:%S)] rgbd exit=$?"
    ;;

  *)
    echo "usage: scripts/run_robot_carried.sh {rail|follower|passive|latched|sweep|smoke|mating|certify|rgbd}"
    exit 2
    ;;
esac
