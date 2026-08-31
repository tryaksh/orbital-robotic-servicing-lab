#!/usr/bin/env bash
# Train insertion on a recorded chain handoff using only local assembly state.

set -euo pipefail

PYTHON="C:/isaac-sim/python.bat"
TASK="Isaac-ZeroG-Blade-GrapplePin-InsertTaskSpaceHandoff-v0"
RUN="${RUN:-grapple_insert_l0_seed70_v30taskspacehandoff}"
OUT="${OUT:-artifacts/insert_v30taskspacehandoff}"
NUM_ENVS="${NUM_ENVS:-512}"
EPOCHS="${EPOCHS:-100}"

mkdir -p "$OUT"
if [ -n "$(git status --porcelain=v1 --untracked-files=no)" ]; then
  echo "Commit tracked task changes before training so the run is reproducible." >&2
  exit 1
fi

echo "[$(date +%H:%M:%S)] smoke posture-invariant handoff insertion"
"$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs 32 --smoke \
  > "$OUT/smoke.log" 2>&1

echo "[$(date +%H:%M:%S)] train $RUN for $EPOCHS epochs from scratch"
"$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs "$NUM_ENVS" \
  --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" --run_name "$RUN" \
  > "$OUT/train.log" 2>&1

checkpoint=$(ls -t "logs/rl_games/zero_g_blade_insertion_contact/$RUN/nn"/*.pth | head -1)
echo "[$(date +%H:%M:%S)] checkpoint=$checkpoint"
sha256sum "$checkpoint"
