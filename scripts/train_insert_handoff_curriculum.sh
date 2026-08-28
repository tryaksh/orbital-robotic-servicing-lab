#!/usr/bin/env bash
# Resume v24 through a measured, success-gated reverse station curriculum.

set -euo pipefail

PYTHON="C:/isaac-sim/python.bat"
TASK="Isaac-ZeroG-Blade-GrapplePin-InsertHandoffCurriculum-v0"
RUN="${RUN:-grapple_insert_l0_seed70_v26reverse}"
OUT="${OUT:-artifacts/insert_v26reverse}"
NUM_ENVS="${NUM_ENVS:-512}"
# RL-Games reads an absolute epoch. This is a time-boxed 1,200-epoch resume.
EPOCHS="${EPOCHS:-3300}"
RESUME_CKPT="${RESUME_CKPT:-logs/rl_games/zero_g_blade_insertion_contact/grapple_insert_l0_seed70_v24rack/nn/last_zero_g_blade_insertion_contact_ep_2100_rew_43.909218.pth}"
EXPECTED_SHA256="47AA9EFB60F7794BE5CDD1EBD0AD5EC0E94CE00345BCF975D83AE9418D9A1B9F"

mkdir -p "$OUT"
if [ ! -f "$RESUME_CKPT" ]; then
  echo "Missing frozen v24 checkpoint: $RESUME_CKPT" >&2
  exit 1
fi
actual_sha256=$(sha256sum "$RESUME_CKPT" | awk '{print toupper($1)}')
if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
  echo "v24 checkpoint hash is $actual_sha256, expected $EXPECTED_SHA256" >&2
  exit 1
fi
if [ -n "$(git status --porcelain=v1 --untracked-files=no)" ]; then
  echo "Commit tracked task changes before training so the run is reproducible." >&2
  exit 1
fi

echo "[$(date +%H:%M:%S)] smoke reverse-station insertion curriculum"
"$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs 32 --smoke \
  > "$OUT/smoke.log" 2>&1

echo "[$(date +%H:%M:%S)] train $RUN from v24 epoch 2100 to absolute epoch $EPOCHS"
"$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs "$NUM_ENVS" \
  --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" --run_name "$RUN" \
  --checkpoint "$RESUME_CKPT" > "$OUT/train.log" 2>&1

checkpoint=$(find "logs/rl_games/zero_g_blade_insertion_contact/$RUN/nn" -type f \
  -name "last_*_ep_${EPOCHS}_rew_*.pth" | grep -E 'rew_-?[0-9]+(\.[0-9]+)?\.pth$' | sort | head -1)
if [ -z "$checkpoint" ]; then
  echo "No unambiguous final epoch-$EPOCHS checkpoint was written." >&2
  exit 1
fi
echo "[$(date +%H:%M:%S)] checkpoint=$checkpoint"
sha256sum "$checkpoint"
