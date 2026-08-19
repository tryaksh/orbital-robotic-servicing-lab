#!/usr/bin/env bash
# Phase 1: run the UNCHANGED promoted policies on the changed workcell.
#
# This is rule 5 -- "before believing a reconstructed distribution, run the
# unchanged successor policy on it" -- applied to a workcell change instead of a
# reset distribution. It costs five minutes per skill and it is what sizes the
# rest of the session: the relocation's measured blocker is the arm, and the
# interface is what makes the failure ugly. If capture, extraction and insertion
# survive the moved base at anything near their certified rates, the gripper work
# is small; if they collapse, it is large. Either way the number is measured
# before anything is built on top of it.
#
# NOT a certification. One seed, fewer episodes, no aggregation, no gate. The
# certified numbers come from scripts/certify_demo_policies.sh after Phase 4.
#
# Usage: scripts/probe_workcell_policies.sh [skill ...]   (default: all three)

set -u

PYTHON="C:/isaac-sim/python.bat"
EVAL_ENVS="${EVAL_ENVS:-128}"
EVAL_EPISODES="${EVAL_EPISODES:-256}"
SEED="${SEED:-1070}"
OUT="${OUT:-artifacts/workcell_probe}"
mkdir -p "$OUT"

CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
GRASP_CKPT="${GRASP_CKPT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v6w65/nn/last_zero_g_blade_insertion_contact_ep_2400_rew__37.24023_.pth}"
EXTRACT_CKPT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v15w65/nn/last_zero_g_blade_insertion_contact_ep_7200_rew__136.52777_.pth}"
INSERT_CKPT="${INSERT_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v11w65/nn/last_zero_g_blade_insertion_contact_ep_5600_rew__-23.204594_.pth}"

skills=("$@")
if [ ${#skills[@]} -eq 0 ]; then
  skills=(Grasp Extract Insert)
fi

for skill in "${skills[@]}"; do
  case "$skill" in
    Grasp)   checkpoint="$GRASP_CKPT";   task="Isaac-ZeroG-Blade-GrapplePin-Grasp-Play-v0";   stages=(0 2) ;;
    Extract) checkpoint="$EXTRACT_CKPT"; task="Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0"; stages=(0 2) ;;
    # The promoted insert is the two-bay policy, so probe it on the two-bay task
    # where stage 0 is the certified bay and stage 1 the second one.
    Insert)  checkpoint="$INSERT_CKPT";  task="Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-Play-v0"; stages=(0 1) ;;
    *) echo "unknown skill $skill"; exit 1 ;;
  esac
  if [ ! -f "$checkpoint" ]; then echo "MISSING checkpoint for $skill: $checkpoint"; exit 1; fi
  lower=$(echo "$skill" | tr '[:upper:]' '[:lower:]')
  for stage in "${stages[@]}"; do
    out="$OUT/${lower}_s${stage}"
    "$PYTHON" scripts/play.py --headless \
        --task "$task" \
        --checkpoint "$checkpoint" \
        --num_envs "$EVAL_ENVS" \
        --episodes "$EVAL_EPISODES" \
        --curriculum_stage "$stage" \
        --seed "$SEED" \
        --report "${out}_play.json" \
        --episode_metrics "${out}.npz" \
        > "${out}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $skill stage=$stage exit=$? -> ${out}_play.json"
    "$PYTHON" - "${out}_play.json" <<'PY'
import json, sys
report = json.loads(open(sys.argv[1], encoding="utf-8").read())
rate = report.get("success_rate")
episodes = report.get("episodes_completed")
print(f"    success_rate={rate} episodes={episodes}")
PY
  done
done
echo "[$(date +%H:%M:%S)] WORKCELL POLICY PROBE DONE"
