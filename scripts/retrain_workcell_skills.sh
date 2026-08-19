#!/usr/bin/env bash
# Fine-tune the three promoted skills onto the moved workcell.
#
# `GRAPPLE_ROBOT_ROOT_POS` moved from (-0.45, 0, 0.15) to (-0.65, 0, 0.15) and
# every head-on spawn pose was re-solved against it
# (`evidence/grapple_pin_head_on_pose_relocated.json`). That is a geometry
# change, not an observation or action change: the observation is still 50
# numbers and the action still 6, so **resuming is legal here** and it is how
# most policies in this project were made. Rule 12 forbids resuming across an
# action or observation dimension change, and nothing here changes either.
#
# What the policies actually see change is the arm's joint configuration, which
# is six of those fifty numbers and is different at every pose. Whether that is
# survivable was measured before this script was written rather than assumed --
# `scripts/probe_workcell_policies.sh`, rule 5 -- and the answer is in
# docs/status.md.
#
# One skill at a time: 512 environments plus an optimizer will not share 12 GB
# with another. Check for orphaned kit processes before launching (rule 13).
#
# Usage: scripts/retrain_workcell_skills.sh [skill ...]   (default: all three)

set -u

PYTHON="C:/isaac-sim/python.bat"
NUM_ENVS="${NUM_ENVS:-512}"
EPOCHS="${EPOCHS:-1200}"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="${OUT:-artifacts/workcell_retrain}"
mkdir -p "$OUT"

# The promoted set in CLAUDE.md, which is what these fine-tune from.
GRASP_CKPT="${GRASP_CKPT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth}"
EXTRACT_CKPT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v13unsat/nn/last_zero_g_blade_insertion_contact_ep_5700_rew__148.17932_.pth}"
INSERT_CKPT="${INSERT_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v10twoslot/nn/last_zero_g_blade_insertion_contact_ep_4400_rew__29.616938_.pth}"

# One insert policy for everything, and that is a simplification worth stating.
# main carries two: single-bay v6, which the installation chain was certified
# with, and two-bay v10twoslot, which the relocation needs. v10twoslot already
# scores 98.87% in the certified bay against v6's 98.27%, so keeping both buys
# nothing and costs a whole training run plus a second certification to keep
# current. The installation chain is re-certified on the two-bay policy here.
skills=("$@")
if [ ${#skills[@]} -eq 0 ]; then
  skills=(Grasp Extract Insert)
fi

for skill in "${skills[@]}"; do
  case "$skill" in
    Grasp)   task="Isaac-ZeroG-Blade-GrapplePin-Grasp-v0";        ckpt="$GRASP_CKPT";   run="grapple_grasp_l0_seed70_v6w65" ;;
    Extract) task="Isaac-ZeroG-Blade-GrapplePin-Extract-v0";      ckpt="$EXTRACT_CKPT"; run="grapple_extract_l0_seed70_v15w65" ;;
    Insert)  task="Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0"; ckpt="$INSERT_CKPT"; run="grapple_insert_l0_seed70_v11w65" ;;
    *) echo "unknown skill $skill"; exit 1 ;;
  esac
  if [ ! -f "$ckpt" ]; then echo "MISSING checkpoint for $skill: $ckpt"; exit 1; fi

  # rl-games counts epochs absolutely, so a resume has to be told the total.
  resume_epoch=$(echo "$ckpt" | sed -n 's/.*_ep_\([0-9]\+\)_.*/\1/p')
  target=$((resume_epoch + EPOCHS))

  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] SMOKE $skill"
  "$PYTHON" scripts/train.py --headless --task "$task" --num_envs 32 --smoke \
      > "$OUT/smoke_${skill}.log" 2>&1
  if grep -qE "^Traceback" "$OUT/smoke_${skill}.log"; then
    echo "[$(date +%H:%M:%S)] SMOKE FAILED for $skill:"
    grep -E "Error|Exception" "$OUT/smoke_${skill}.log" | tail -3
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] smoke clean"

  echo "[$(date +%H:%M:%S)] TRAIN $skill  $resume_epoch + $EPOCHS -> $target  run=$run"
  "$PYTHON" scripts/train.py --headless \
      --task "$task" \
      --num_envs "$NUM_ENVS" \
      --seed 70 \
      --robustness_level 0 \
      --max_iterations "$target" \
      --checkpoint "$ckpt" \
      --run_name "$run" \
      > "$OUT/train_${skill}.log" 2>&1
  echo "[$(date +%H:%M:%S)] train exit=$?  (judge progress from summaries/ and nn/ mtime, never this log)"
  ls -t "logs/rl_games"/*/"$run"/nn/*.pth 2>/dev/null | head -1
done

echo "[$(date +%H:%M:%S)] WORKCELL RETRAIN DONE"
