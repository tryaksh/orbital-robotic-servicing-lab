#!/usr/bin/env bash
# Retrain and re-certify all three skills against the anti-yaw yoke, then close
# the chain. One launch, unattended, ~4.5 h of GPU.
#
# The question this pipeline answers is single and stated up front: **does the
# anti-yaw yoke take the extract skill off 0.00%?** Extraction is the cleanest
# instrument this project owns for yaw, because it certifies at exactly zero
# while holding grip *position* at 12.2 mm for a whole 15 s pull and failing
# only on grip *attitude*, 0.299 rad against a 0.20 rad limit. Anything above
# zero is the yoke working. Everything after step 4 is downstream of that.
#
# Fine-tuning rather than training from scratch is deliberate and legitimate
# here. The yoke changes contact geometry; it changes neither the observation
# nor the action dimension, so the rule against resuming across an interface
# change does not apply, and the promoted L0 -> L1 -> L2 insertion lineage is
# precedent for fine-tuning across a physics change. It is also what makes this
# fit in one night.
#
# The pre-flight is one run, at the top, already done before this launched:
#   grasp_diagnostics.py --anti_yaw_yoke on the capture scene reports a grip in
#   96/96 environments, drive torque saturated at 10 N-m, and 69.7 N held
#   against the 66.4 N gate. The yoke does not cost the hold.
#
# Usage: scripts/run_yoke_pipeline.sh [stage ...]
#   stages: train_grasp cert_grasp train_extract cert_extract
#           train_insert cert_insert chain full
# Default: all of them, in that order.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
TRAIN_ENVS="${TRAIN_ENVS:-512}"
SEED="${SEED:-70}"
LEVEL="${LEVEL:-0}"
LOGS="artifacts/yoke_pipeline"
mkdir -p "$LOGS" evidence

# Sources. Each is the checkpoint its own certification in evidence/ names, so
# the lineage of every yoked policy traces back to a measured plain-pin one.
GRASP_SRC="$CKPT_ROOT/grapple_grasp_l0_seed70_v3/nn/last_zero_g_blade_insertion_contact_ep_700_rew__36.020187_.pth"
EXTRACT_SRC="$CKPT_ROOT/grapple_extract_l0_seed70_v4/nn/last_zero_g_blade_insertion_contact_ep_1200_rew__162.91257_.pth"
INSERT_SRC="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"

# Destinations.
GRASP_RUN=grapple_grasp_l0_seed70_v4
EXTRACT_RUN=grapple_extract_l0_seed70_v5
INSERT_RUN=grapple_insert_l0_seed70_v7

# RL-Games restores the epoch counter with the weights, so --max_iterations is
# an absolute epoch, not a budget. Source epoch plus the fine-tune length.
GRASP_EPOCHS=1100     # 700 + 400
EXTRACT_EPOCHS=1800   # 1200 + 600
INSERT_EPOCHS=3700    # 3200 + 500

say() { echo "[$(date +%H:%M:%S)] $*"; }

# The final checkpoint of a run, chosen by epoch number rather than by mtime:
# the reward is in the filename and cannot be predicted, and "newest file" picks
# up the best-so-far checkpoint rl_games also writes alongside the periodic ones.
#
# The epoch is passed in and matched exactly where possible. A run directory
# that already carried a longer lineage is exactly how the wrong policy gets
# certified, which is the mistake that cost this project a whole session.
ckpt_at() {
  local run="$1" epoch="$2"
  local dir="$CKPT_ROOT/$run/nn"
  local exact
  exact=$(ls "$dir" 2>/dev/null | grep -E "^last_.*_ep_${epoch}_rew_.*\.pth$" | sort | tail -1)
  if [ -n "$exact" ]; then echo "$exact"; return 0; fi
  say "  WARNING: $run has no epoch-$epoch checkpoint; falling back to its highest" >&2
  ls "$dir" 2>/dev/null \
    | grep -E '^last_.*_ep_[0-9]+_rew_.*\.pth$' \
    | sed -E 's/.*_ep_([0-9]+)_rew_.*/\1 &/' \
    | sort -n -k1,1 | tail -1 | cut -d' ' -f2-
}

train() {
  local task="$1" run="$2" src="$3" epochs="$4"
  if [ ! -f "$src" ]; then say "MISSING source checkpoint: $src"; exit 1; fi
  say "TRAIN $run  (resume $(basename "$src") -> epoch $epochs)"
  "$PYTHON" scripts/train.py --headless \
      --task "$task" \
      --num_envs "$TRAIN_ENVS" \
      --seed "$SEED" \
      --robustness_level "$LEVEL" \
      --max_iterations "$epochs" \
      --checkpoint "$src" \
      --run_name "$run" \
      > "$LOGS/train_${run}.log" 2>&1
  local status=$?
  say "  exit=$status"
  if [ "$status" -ne 0 ]; then tail -30 "$LOGS/train_${run}.log"; exit 1; fi
  local produced
  produced=$(ckpt_at "$run" "$epochs")
  if [ -z "$produced" ]; then say "  no checkpoint written by $run"; exit 1; fi
  say "  -> $produced"
}

stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then
  stages=(train_grasp cert_grasp train_extract cert_extract train_insert cert_insert chain full)
fi

for stage in "${stages[@]}"; do
  echo "=============================================================="
  say "STAGE $stage"
  echo "=============================================================="
  case "$stage" in
    train_grasp)
      train Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 "$GRASP_RUN" "$GRASP_SRC" "$GRASP_EPOCHS"
      ;;
    train_extract)
      train Isaac-ZeroG-Blade-GrapplePin-Extract-v0 "$EXTRACT_RUN" "$EXTRACT_SRC" "$EXTRACT_EPOCHS"
      ;;
    train_insert)
      train Isaac-ZeroG-Blade-GrapplePin-Insert-v0 "$INSERT_RUN" "$INSERT_SRC" "$INSERT_EPOCHS"
      ;;
    cert_grasp)
      GRASP_CKPT="$CKPT_ROOT/$GRASP_RUN/nn/$(ckpt_at "$GRASP_RUN" "$GRASP_EPOCHS")" \
      GRASP_VERSION=v4 INTERFACE="the yoked grapple pin (anti-yaw walls on)" \
        bash scripts/certify_demo_policies.sh Grasp
      ;;
    cert_extract)
      # The answer. Extract certified at 0.00% on the plain pin for one reason
      # and one only, so any non-zero success rate here is the yoke working.
      EXTRACT_CKPT="$CKPT_ROOT/$EXTRACT_RUN/nn/$(ckpt_at "$EXTRACT_RUN" "$EXTRACT_EPOCHS")" \
      EXTRACT_VERSION=v5 INTERFACE="the yoked grapple pin (anti-yaw walls on)" \
        bash scripts/certify_demo_policies.sh Extract
      ;;
    cert_insert)
      INSERT_CKPT="$CKPT_ROOT/$INSERT_RUN/nn/$(ckpt_at "$INSERT_RUN" "$INSERT_EPOCHS")" \
      INSERT_VERSION=v7 INTERFACE="the yoked grapple pin (anti-yaw walls on)" \
        bash scripts/certify_demo_policies.sh Insert
      ;;
    chain)
      GRASP_CKPT="$CKPT_ROOT/$GRASP_RUN/nn/$(ckpt_at "$GRASP_RUN" "$GRASP_EPOCHS")" \
      EXTRACT_CKPT="$CKPT_ROOT/$EXTRACT_RUN/nn/$(ckpt_at "$EXTRACT_RUN" "$EXTRACT_EPOCHS")" \
      INSERT_CKPT="$CKPT_ROOT/$INSERT_RUN/nn/$(ckpt_at "$INSERT_RUN" "$INSERT_EPOCHS")" \
      TAG=_yoked \
        bash scripts/certify_workflow.sh remove install
      ;;
    full)
      # The round trip has never been run above one environment; the
      # per-environment transit waypoint buffers exist and are untested. It is
      # last on purpose, so a failure here costs nothing already measured.
      grasp="$CKPT_ROOT/$GRASP_RUN/nn/$(ckpt_at "$GRASP_RUN" "$GRASP_EPOCHS")"
      extract="$CKPT_ROOT/$EXTRACT_RUN/nn/$(ckpt_at "$EXTRACT_RUN" "$EXTRACT_EPOCHS")"
      insert="$CKPT_ROOT/$INSERT_RUN/nn/$(ckpt_at "$INSERT_RUN" "$INSERT_EPOCHS")"
      say "FULL round trip, 64 environments"
      "$PYTHON" scripts/run_workflow_demo.py --headless \
          --workflow full --curriculum_stage 0 \
          --grasp_checkpoint "$grasp" \
          --extract_checkpoint "$extract" \
          --insert_checkpoint "$insert" \
          --num_envs 64 --episodes 64 --seed 4070 \
          --report "$LOGS/full_yoked_report.json" \
          --episode_metrics "$LOGS/full_yoked.npz" \
          > "$LOGS/full_yoked.log" 2>&1
      say "  exit=$? (n=64)"
      tail -20 "$LOGS/full_yoked.log"
      say "FULL round trip, one environment, recorded"
      "$PYTHON" scripts/run_workflow_demo.py --headless \
          --workflow full --curriculum_stage 0 \
          --grasp_checkpoint "$grasp" \
          --extract_checkpoint "$extract" \
          --insert_checkpoint "$insert" \
          --num_envs 1 --seed 4070 --video \
          --video_dir artifacts/demo/workflow_full_yoked \
          --report "$LOGS/full_yoked_n1_report.json" \
          > "$LOGS/full_yoked_n1.log" 2>&1
      say "  exit=$? (n=1, video)"
      tail -20 "$LOGS/full_yoked_n1.log"
      ;;
    *) say "unknown stage $stage"; exit 1 ;;
  esac
done

say "YOKE PIPELINE DONE"
