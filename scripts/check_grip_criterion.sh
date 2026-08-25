#!/usr/bin/env bash
# Separate the criterion change from the policy change, before training anything.
#
# The extraction grip test moved from an isotropic 30 mm ball about the pin's
# drawing pose to bounds resolved on the pin's own axes. That is a change to what
# counts as a held module, so the *same* checkpoint has to be run under both or
# the next number confuses a better policy with a better ruler.
#
# --legacy_grip_ball_m 0.030 restores the old test exactly.

set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/diag"
V17="$CKPT_ROOT/grapple_extract_l0_seed70_v17m130/nn/last_zero_g_blade_insertion_contact_ep_10600_rew_168.46431.pth"
TASK="Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0"

echo "[$(date +%H:%M:%S)] SMOKE the changed task"
"$PYTHON" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Extract-v0 \
    --num_envs 32 --smoke > "$OUT/smoke_extract_pin.log" 2>&1
if grep -qE "^Traceback" "$OUT/smoke_extract_pin.log"; then
  echo "SMOKE FAILED"; grep -E "Error|Exception" "$OUT/smoke_extract_pin.log" | tail -3; exit 1
fi
echo "[$(date +%H:%M:%S)] smoke clean"

for pair in "s0 0" "s2 2"; do
  set -- $pair
  for arm in pin ball; do
    tag="crit_${arm}_$1"
    extra=""
    [ "$arm" = "ball" ] && extra="--legacy_grip_ball_m 0.030"
    echo "[$(date +%H:%M:%S)] $tag"
    "$PYTHON" scripts/play.py --headless --task "$TASK" --checkpoint "$V17" \
        --num_envs 128 --episodes 512 --curriculum_stage "$2" --seed 1070 \
        --grip_axis_metrics $extra \
        --episode_metrics "$OUT/${tag}.npz" --report "$OUT/${tag}.json" \
        > "$OUT/${tag}.log" 2>&1
    echo "[$(date +%H:%M:%S)]   exit=$? $(grep -oE '"success_rate": [0-9.]+' "$OUT/${tag}.json" | head -1)"
  done
done
echo "[$(date +%H:%M:%S)] DONE"
