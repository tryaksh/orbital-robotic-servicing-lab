#!/usr/bin/env bash
# Certify a grapple skill: three curriculum stages on each of three held-out
# seeds, pooled into one gated report under evidence/.
#
# SKILL, CKPT, TAG and TITLE are the knobs. EXTRA passes through to play.py,
# which is how a *control* arm is run: the same protocol on an archived
# checkpoint under the criterion it was certified against.
#
#   SKILL=Extract CKPT=... TAG=extract_v18pin scripts/certify_grapple_skills.sh

set -u
PYTHON="C:/isaac-sim/python.bat"
OUT="${OUT:-artifacts/certify_skills}"
mkdir -p "$OUT" evidence

SKILL="${SKILL:?set SKILL to Grasp, Extract or Insert}"
CKPT="${CKPT:?set CKPT to the checkpoint to certify}"
TAG="${TAG:?set TAG for the evidence file name}"
TITLE="${TITLE:-Head-on grapple-pin ${SKILL,,} skill, ${TAG}}"
EXTRA="${EXTRA:-}"
STAGES="${STAGES:-0 1 2}"
EVAL_ENVS="${EVAL_ENVS:-128}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
PLAY="Isaac-ZeroG-Blade-GrapplePin-${SKILL}-Play-v0"
[ "$SKILL" = "Insert" ] && PLAY="Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-Play-v0"
# PLAY_TASK overrides the default for a checkpoint whose observation width is
# not the default task's. The force-feedback seating policy is the first of
# those: certifying it on a task it did not train on hands it an observation of
# the wrong size, which fails loudly but only after the run has started.
PLAY="${PLAY_TASK:-$PLAY}"

rows=()
for seed in 1070 2070 3070; do
  for stage in $STAGES; do
    out="$OUT/${TAG}_s${stage}_seed${seed}"
    "$PYTHON" scripts/play.py --headless --task "$PLAY" --checkpoint "$CKPT" \
        --num_envs "$EVAL_ENVS" --episodes "$EVAL_EPISODES" \
        --curriculum_stage "$stage" --seed "$seed" --grip_axis_metrics $EXTRA \
        --episode_metrics "${out}.npz" --report "${out}.json" \
        > "${out}.log" 2>&1
    echo "[$(date +%H:%M:%S)]   $TAG stage=$stage seed=$seed exit=$? $(grep -oE '"success_rate": [0-9.]+' "${out}.json" | head -1)"
    rows+=("${out}.npz")
  done
done

"$PYTHON" scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
    --output "evidence/grapple_${TAG}_certification.json" \
    --title "$TITLE" \
    --scope \
      "Simulation only. No result here was produced on real hardware." \
      "The grasp is physical pad-against-pin contact, not a fixed joint." \
      "One PPO training seed. The evaluation seeds are held out, but training repeatability is untested." \
    > "$OUT/aggregate_${TAG}.log" 2>&1
echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/grapple_${TAG}_certification.json"
tail -4 "$OUT/aggregate_${TAG}.log"
