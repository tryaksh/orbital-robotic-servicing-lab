#!/usr/bin/env bash
# Record demonstration clips of the three head-on grapple-pin skills.
#
# Every clip is driven by a trained checkpoint. A scripted controller is allowed
# in this repository only as a physics feasibility test, never as a
# demonstration, so if a skill has no checkpoint it is skipped rather than faked.
#
# Usage: scripts/record_grapple_demo.sh [skill ...]   (default: all three)

set -u

PYTHON="C:/isaac-sim/python.bat"
STEPS="${STEPS:-450}"
mkdir -p artifacts/demo

skills=("$@")
if [ ${#skills[@]} -eq 0 ]; then
  skills=(Grasp Extract Insert)
fi

for skill in "${skills[@]}"; do
  lower=$(echo "$skill" | tr '[:upper:]' '[:lower:]')
  run="grapple_${lower}_l0_seed70"
  checkpoint=$(ls -t logs/rl_games/*/"$run"/nn/*.pth 2>/dev/null | head -1)
  if [ -z "$checkpoint" ]; then
    echo "[demo] no checkpoint for $skill; skipping"
    continue
  fi
  # Two angles per skill: the tool and the pin close up, and the whole workcell.
  for view in grasp side; do
    echo "[demo] $skill / $view <- $checkpoint"
    "$PYTHON" scripts/play.py --headless \
        --task "Isaac-ZeroG-Blade-GrapplePin-${skill}-Play-v0" \
        --checkpoint "$checkpoint" \
        --num_envs 1 \
        --steps "$STEPS" \
        --curriculum_stage 2 \
        --seed 4070 \
        --inspection_view "$view" \
        --video --video_length "$STEPS" \
        --video_dir "artifacts/demo/${lower}_${view}" \
        > "artifacts/demo/${lower}_${view}.log" 2>&1
    echo "[demo]   exit=$?"
  done
done
echo "[demo] done. Clips under artifacts/demo/ (untracked; publish via a GitHub Release)."
