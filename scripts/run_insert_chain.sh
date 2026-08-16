#!/usr/bin/env bash
# Train the insert skill inside the chain, then certify the chain it belongs to.
#
# The installation chain is the one gap left in the relocation demonstration:
# 84.38% certified, against a 95% gate, and the whole of the shortfall is the
# insert phase, which runs at about 80% on the states a real capture hands it
# while certifying at 95.57% on its own reset.
#
# Four reconstructions of that hand-off as a *reset distribution* were built and
# refuted (0.00%, 26.32%, 47.17% against ~80%), so this trains in place instead:
# `Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0` runs the frozen capture policy
# inside the environment and hands the arm over on the chain's own predicate.
#
# The gate that had to pass before any of this ran: insert v6, unchanged, on the
# new task must score what it scores in the real chain, and the state it takes
# over in must match the chain's hand-off column by column. Both were measured
# first; see docs/status.md.
#
# Usage: scripts/run_insert_chain.sh          (smoke, train, certify)
#        SKIP_TRAIN=1 scripts/run_insert_chain.sh   (certify an existing run)

set -u

PYTHON="C:/isaac-sim/python.bat"
NUM_ENVS="${NUM_ENVS:-512}"
EPOCHS="${EPOCHS:-1500}"
SEED="${SEED:-70}"
RUN="${RUN:-grapple_insert_l0_seed70_v8chain}"
TASK="Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0"
LOGROOT="logs/rl_games"
CKPT_ROOT="$LOGROOT/zero_g_blade_insertion_contact"
mkdir -p artifacts/chain evidence

# The policy this fine-tunes from, and the frozen capture the task runs.
RESUME="${RESUME:-$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth}"
export CAPTURE_CHECKPOINT="${CAPTURE_CHECKPOINT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth}"

for checkpoint in "$RESUME" "$CAPTURE_CHECKPOINT"; do
  if [ ! -f "$checkpoint" ]; then
    echo "MISSING checkpoint: $checkpoint"
    exit 1
  fi
done

# rl-games treats `max_epochs` as an ABSOLUTE epoch number, not a number of
# additional epochs, and `--max_iterations` writes it straight through. Resuming
# insert v6 (epoch 3200) with `--max_iterations 1200` therefore does not train
# for 1200 epochs: the runner sees 3200 >= 1200, stops before the first update,
# and writes `..._ep_3201_rew_-inf.pth`, which looks exactly like a checkpoint.
# Measured here the expensive way -- the script went straight on to certify that
# file. EPOCHS below means *additional* epochs, and the absolute target is
# derived from the resume checkpoint's own epoch so the two cannot disagree.
RESUME_EPOCH=$(echo "$RESUME" | sed -n 's/.*_ep_\([0-9]\+\)_.*/\1/p')
if [ -z "$RESUME_EPOCH" ]; then
  echo "Cannot read the resume checkpoint's epoch from its filename: $RESUME"
  exit 1
fi
TARGET_EPOCH=$((RESUME_EPOCH + EPOCHS))
echo "[$(date +%H:%M:%S)] resuming at epoch $RESUME_EPOCH, training $EPOCHS more, absolute target $TARGET_EPOCH"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] SMOKE $TASK"
  "$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs 32 \
      --robustness_level 0 --smoke > artifacts/chain/smoke_insertchain.log 2>&1
  if grep -qE "^Traceback" artifacts/chain/smoke_insertchain.log; then
    echo "[$(date +%H:%M:%S)] SMOKE FAILED. Last error:"
    grep -E "Error|Exception" artifacts/chain/smoke_insertchain.log | tail -3
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] smoke clean"

  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] TRAIN insert-in-chain  envs=$NUM_ENVS epochs=$EPOCHS run=$RUN"
  echo "  resume:  $RESUME"
  echo "  capture: $CAPTURE_CHECKPOINT"
  echo "=============================================================="
  "$PYTHON" scripts/train.py --headless \
      --task "$TASK" \
      --num_envs "$NUM_ENVS" \
      --seed "$SEED" \
      --robustness_level 0 \
      --max_iterations "$TARGET_EPOCH" \
      --checkpoint "$RESUME" \
      --run_name "$RUN" \
      > artifacts/chain/train_insertchain.log 2>&1
  echo "[$(date +%H:%M:%S)] train exit=$?"
fi

checkpoint=$(ls -t "$LOGROOT"/*/"$RUN"/nn/*.pth 2>/dev/null | head -1)
if [ -z "$checkpoint" ]; then
  echo "[$(date +%H:%M:%S)] NO CHECKPOINT for $RUN"
  exit 1
fi
echo "[$(date +%H:%M:%S)] checkpoint: $checkpoint"

# The gate is the chain, not the skill. A skill certification is not evidence
# about the chain, so this goes straight to certify_workflow.sh with the new
# insert checkpoint in place.
INSERT_CKPT="$checkpoint" TAG="_insertchain" scripts/certify_workflow.sh install

echo "[$(date +%H:%M:%S)] INSERT-IN-CHAIN DONE"
