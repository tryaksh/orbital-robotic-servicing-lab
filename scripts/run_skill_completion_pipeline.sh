#!/usr/bin/env bash
# Finish the three skills and both chains. One launch, unattended, ~4 h.
#
# Two blockers, each with a measurement behind it:
#
#   0. EXTRACT COULD NOT TRACK THE MODULE. Over 9,002 held-out episodes the
#      module rotates at 0.296 rad/s at p95 and 0.767 at maximum, while the
#      wrist's rotational action scale allowed 0.240 rad/s. The policy was
#      asked to hold an attitude against something faster than its action
#      space could move, which is why 99% of failures end at the 0.350 rad
#      limit and why no reward weighting closed it. Lateral authority was
#      0.03 m/s against 0.24 axial, an 8:1 asymmetry inherited from the
#      insertion task where the rails do the steering. Now 0.12 m/s lateral
#      and 0.60 rad/s rotational.
#
#   1. EXTRACT IN THE CHAIN. Extract v7 certifies at 28.48% alone with a 15.2 s
#      median, and chained it overruns its own 25 s budget in 138 of 192
#      removals while the module's orientation error reaches 0.85 rad against
#      0.12 in isolation. Of the 49 that do fire the predicate only 7 survive
#      the settling re-check. That is a skill trained on the wrong initial
#      states: its reset noise was +-0.020 rad and the hand-off is wider. Reset
#      noise doubled to (0.020, 0.030, 0.040) and the skill fine-tuned against
#      it. Isolated success may fall -- the distribution is genuinely harder --
#      and the chain is what this change is judged on.
#
#   2. CAPTURE'S WORST STAGE. Capture certifies 95.55% pooled but 92.61% at its
#      worst stage, which is below the gate, and 29 of the install chain's 79
#      failures are capture overrunning its own 6 s budget. More epochs on the
#      objective it already has, before anything more exotic.
#
# Evaluation runs at 256 environments rather than 128. Episodes are pooled, so
# this changes wall-clock and nothing else. Training stays at 512, which is the
# only count every promoted policy was produced at and the only one measured
# safe for full PPO memory on this 12 GB part.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
LOGS=artifacts/skill_completion
mkdir -p "$LOGS" evidence

EXTRACT_SRC="$CKPT_ROOT/grapple_extract_l0_seed70_v7attitude/nn/last_zero_g_blade_insertion_contact_ep_2400_rew__54.265915_.pth"
GRASP_SRC="$CKPT_ROOT/grapple_grasp_l0_seed70_v3/nn/last_zero_g_blade_insertion_contact_ep_700_rew__36.020187_.pth"
INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"

EXTRACT_RUN=grapple_extract_l0_seed70_v8handoff
EXTRACT_EPOCHS=3800      # 2400 + 1400
GRASP_RUN=grapple_grasp_l0_seed70_v5
GRASP_EPOCHS=1500        # 700 + 800

export EVAL_ENVS=256

say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt_at() { ls "$CKPT_ROOT/$1/nn/" 2>/dev/null | grep -E "^last_.*_ep_$2_rew_.*\.pth$" | sort | tail -1; }

train() {
  local task="$1" run="$2" src="$3" epochs="$4"
  if [ ! -f "$src" ]; then say "MISSING $src"; exit 1; fi
  say "TRAIN $run  (resume $(basename "$src") -> epoch $epochs)"
  "$PYTHON" scripts/train.py --headless --task "$task" \
      --num_envs 512 --seed 70 --robustness_level 0 \
      --max_iterations "$epochs" --checkpoint "$src" --run_name "$run" \
      > "$LOGS/train_${run}.log" 2>&1
  say "  exit=$?  -> $(ckpt_at "$run" "$epochs")"
}

stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then
  stages=(extract cert_extract chain_extract grasp cert_grasp chain_final)
fi

for stage in "${stages[@]}"; do
  echo "=============================================================="
  say "STAGE $stage"
  echo "=============================================================="
  case "$stage" in
    extract)
      train Isaac-ZeroG-Blade-GrapplePin-Extract-v0 "$EXTRACT_RUN" "$EXTRACT_SRC" "$EXTRACT_EPOCHS"
      ;;
    cert_extract)
      EXTRACT_CKPT="$CKPT_ROOT/$EXTRACT_RUN/nn/$(ckpt_at "$EXTRACT_RUN" "$EXTRACT_EPOCHS")" \
      EXTRACT_VERSION=v8 INTERFACE="the plain grapple pin, 25 s episode, hand-off reset envelope" \
        bash scripts/certify_demo_policies.sh Extract
      ;;
    chain_extract)
      GRASP_CKPT="$GRASP_SRC" \
      EXTRACT_CKPT="$CKPT_ROOT/$EXTRACT_RUN/nn/$(ckpt_at "$EXTRACT_RUN" "$EXTRACT_EPOCHS")" \
      INSERT_CKPT="$INSERT_CKPT" TAG=_v8 \
        bash scripts/certify_workflow.sh remove
      ;;
    grasp)
      train Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 "$GRASP_RUN" "$GRASP_SRC" "$GRASP_EPOCHS"
      ;;
    cert_grasp)
      GRASP_CKPT="$CKPT_ROOT/$GRASP_RUN/nn/$(ckpt_at "$GRASP_RUN" "$GRASP_EPOCHS")" \
      GRASP_VERSION=v5 INTERFACE="the plain grapple pin" \
        bash scripts/certify_demo_policies.sh Grasp
      ;;
    chain_final)
      GRASP_CKPT="$CKPT_ROOT/$GRASP_RUN/nn/$(ckpt_at "$GRASP_RUN" "$GRASP_EPOCHS")" \
      EXTRACT_CKPT="$CKPT_ROOT/$EXTRACT_RUN/nn/$(ckpt_at "$EXTRACT_RUN" "$EXTRACT_EPOCHS")" \
      INSERT_CKPT="$INSERT_CKPT" TAG=_final \
        bash scripts/certify_workflow.sh remove install
      say "auditing that every quoted chain loads a certified policy"
      "$PYTHON" scripts/check_evidence_currency.py artifacts/workflow_cert/*_final_seed*_report.json || true
      ;;
    *) say "unknown stage $stage"; exit 1 ;;
  esac
done

say "SKILL COMPLETION PIPELINE DONE"
