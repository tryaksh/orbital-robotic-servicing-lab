#!/usr/bin/env bash
# Fine-tune insertion from an actual recorded transit-to-insert handoff.

set -euo pipefail

PYTHON="C:/isaac-sim/python.bat"
TASK="Isaac-ZeroG-Blade-GrapplePin-InsertHandoff-v0"
RUN="${RUN:-grapple_insert_l0_seed70_v28recordedhandoff}"
OUT="${OUT:-artifacts/insert_v28recordedhandoff}"
NUM_ENVS="${NUM_ENVS:-512}"
# RL-Games interprets this as an absolute epoch. v27 is at 2300, so 2500 is a
# time-boxed 200-epoch distribution intervention, not a blind restart.
EPOCHS="${EPOCHS:-2500}"
RESUME_CKPT="${RESUME_CKPT:-logs/rl_games/zero_g_blade_insertion_contact/grapple_insert_l0_seed70_v27actionscale/nn/last_zero_g_blade_insertion_contact_ep_2300_rew_95.05724.pth}"
EXPECTED_SHA256="010E9D14B9E6C22F99B699820C349DAE3B184436C542615088B43F3B03FD1408"

mkdir -p "$OUT"
if [ ! -f "$RESUME_CKPT" ]; then
  echo "Missing frozen v27 checkpoint: $RESUME_CKPT" >&2
  exit 1
fi
actual_sha256=$(sha256sum "$RESUME_CKPT" | awk '{print toupper($1)}')
if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
  echo "v27 checkpoint hash is $actual_sha256, expected $EXPECTED_SHA256" >&2
  exit 1
fi
if [ -n "$(git status --porcelain=v1 --untracked-files=no)" ]; then
  echo "Commit tracked task changes before training so the run is reproducible." >&2
  exit 1
fi

echo "[$(date +%H:%M:%S)] smoke handoff-conditioned insertion"
"$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs 32 --smoke \
  > "$OUT/smoke.log" 2>&1

echo "[$(date +%H:%M:%S)] train $RUN from v27 epoch 2300 to absolute epoch $EPOCHS"
"$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs "$NUM_ENVS" \
  --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" --run_name "$RUN" \
  --checkpoint "$RESUME_CKPT" > "$OUT/train.log" 2>&1

checkpoint=$(ls -t "logs/rl_games/zero_g_blade_insertion_contact/$RUN/nn"/*.pth | head -1)
echo "[$(date +%H:%M:%S)] checkpoint=$checkpoint"
sha256sum "$checkpoint"
