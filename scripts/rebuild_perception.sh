#!/usr/bin/env bash
# Phase 8: rebuild perception on the changed geometry.
#
# A geometry change invalidates the pose head. Not because the module moved --
# the bays and the module's reset distribution are untouched -- but because the
# ARM moved, and the arm is in the frame. The head regresses the module's pose
# from a 64x64 image in which a 200 mm base shift changes what occludes what.
# Nothing about that is visible in the head's own validation error until it is
# re-measured, so it is re-measured.
#
# One stage per invocation:
#
#   scripts/rebuild_perception.sh collect   # 60,000 two-bay frames
#   scripts/rebuild_perception.sh head      # train and gate the pose head
#   scripts/rebuild_perception.sh arms      # oracle / camera / blind, three seeds
#   scripts/rebuild_perception.sh variance  # the camera arm twice per seed
#
# The camera arm is NOT deterministic and the state pipeline is: two runs with
# identical seed, task and checkpoints diverge at the first episode, and six runs
# span 80.73-86.46% with a 2.13-point standard deviation, while two oracle runs
# are bit-identical across all 192 episodes and 21 columns. So `arms` is one
# draw, `variance` is what says how wide a draw it is, and no camera number is
# quoted here without it.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
DATASET="${DATASET:-datasets/grapple_vision_two_slot_w65.npz}"
HEAD="${HEAD:-checkpoints/module_pose_head_two_slot_w65.pth}"
OUT="${OUT:-artifacts/perception_w65}"
SEEDS="${SEEDS:-4070 5070 6070}"
mkdir -p "$OUT" evidence datasets checkpoints

# Two-bay install workflow, which is what main's two-bay vision arms measured.
# Like for like: the vision arms exist to price perception against an oracle, and
# that needs a manipulation task that completes for all three arms.
TASK="${TASK:-Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Install-v0}"
WORKFLOW="${WORKFLOW:-install}"
STAGE="${STAGE:-2}"

stage="${1:-}"

case "$stage" in
  collect)
    echo "[$(date +%H:%M:%S)] COLLECT 60,000 two-bay frames on the moved workcell"
    "$PYTHON" scripts/collect_grapple_vision.py --headless --enable_cameras \
        --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Collect-v0 \
        --output "$DATASET" --samples "${SAMPLES:-60000}" --num_envs "${ENVS:-64}" \
        > "$OUT/collect.log" 2>&1
    echo "[$(date +%H:%M:%S)] collect exit=$? -> $DATASET"
    grep -E "frames|occupancy|bay" "$OUT/collect.log" | tail -6
    ;;

  head)
    echo "[$(date +%H:%M:%S)] TRAIN the two-bay pose head"
    "$PYTHON" scripts/train_pose_head.py \
        --dataset "$DATASET" --output "$HEAD" \
        --report evidence/module_pose_head_two_slot_w65.json \
        > "$OUT/head.log" 2>&1
    echo "[$(date +%H:%M:%S)] head exit=$? -> $HEAD"
    tail -8 "$OUT/head.log"
    ;;

  arms)
    for arm in oracle camera blind; do
      ARMS="$arm" TASK="$TASK" WORKFLOW="$WORKFLOW" STAGE="$STAGE" HEAD="$HEAD" \
        TAG="_twoslot_w65" SEEDS="$SEEDS" OUT="$OUT" \
        bash scripts/certify_vision_workflow.sh
    done
    ;;

  variance)
    # Run the camera arm twice per seed and report the spread, because a single
    # camera certification is one draw from a distribution the state pipeline
    # does not have.
    for seed in $SEEDS; do
      for repeat in a b; do
        ARMS="camera" TASK="$TASK" WORKFLOW="$WORKFLOW" STAGE="$STAGE" HEAD="$HEAD" \
          TAG="_var${repeat}" SEEDS="$seed" OUT="$OUT" \
          bash scripts/certify_vision_workflow.sh
      done
    done
    ;;

  *)
    echo "usage: scripts/rebuild_perception.sh {collect|head|arms|variance}"
    exit 2
    ;;
esac
