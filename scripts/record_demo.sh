#!/usr/bin/env bash
# Record demonstration clips of a trained insertion policy.
#
# Every clip is driven by a trained checkpoint. A scripted controller is allowed
# in this repository only as a physics feasibility test, never as a
# demonstration, so a missing checkpoint stops the run rather than being faked.
#
# Usage: scripts/record_demo.sh <play-task> <checkpoint> [view ...]
#   scripts/record_demo.sh Isaac-ZeroG-Blade-Insertion-ForceFeedback-Play-v0 \
#       logs/rl_games/zero_g_blade_insertion_rigid_grasp/<run>/nn/<file>.pth grasp side

set -u

PYTHON="C:/isaac-sim/python.bat"
STEPS="${STEPS:-450}"
STAGE="${STAGE:-2}"
SEED="${SEED:-4070}"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <play-task> <checkpoint> [view ...]" >&2
  exit 64
fi

task="$1"
checkpoint="$2"
shift 2
views=("$@")
if [ ${#views[@]} -eq 0 ]; then
  views=(grasp side)
fi

if [ ! -f "$checkpoint" ]; then
  echo "[demo] no checkpoint at '$checkpoint'; refusing to record a scripted stand-in" >&2
  exit 66
fi

label=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
mkdir -p artifacts/demo

for view in "${views[@]}"; do
  echo "[demo] $task / $view <- $checkpoint"
  "$PYTHON" scripts/play.py --headless \
      --task "$task" \
      --checkpoint "$checkpoint" \
      --num_envs 1 \
      --steps "$STEPS" \
      --curriculum_stage "$STAGE" \
      --seed "$SEED" \
      --inspection_view "$view" \
      --video --video_length "$STEPS" \
      --video_dir "artifacts/demo/${label}_${view}" \
      > "artifacts/demo/${label}_${view}.log" 2>&1
  echo "[demo]   exit=$?"
done
echo "[demo] done. Clips under artifacts/demo/ (untracked; publish via a GitHub Release)."
