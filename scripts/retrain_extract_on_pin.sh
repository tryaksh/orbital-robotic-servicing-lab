#!/usr/bin/env bash
# Extract, retrained against the criterion and the rack it actually has.
#
# Three things changed under this skill and none of them is a hyperparameter:
#
#  * the grip test is resolved onto the pin's own axes instead of an isotropic
#    30 mm ball about a drawing pose the load path sits 12.0 mm away from;
#  * the retention reward charges the two residuals that ball hid, so lateral
#    drift has a gradient for the first time -- its position term used to be
#    saturated on every step of every episode;
#  * the channel's lateral clearance is derived from what the pads can follow
#    instead of inherited from a 160 mm module, 15.750 -> 12.689 mm.
#
# Two evaluation points first, on the unchanged checkpoint, so the criterion
# change and the policy change are never quoted as one number. Then the run.

set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/extract_pin"
mkdir -p "$OUT"
V17="$CKPT_ROOT/grapple_extract_l0_seed70_v17m130/nn/last_zero_g_blade_insertion_contact_ep_10600_rew_168.46431.pth"
PLAY="Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0"
TRAIN="Isaac-ZeroG-Blade-GrapplePin-Extract-v0"
RUN="grapple_extract_l0_seed70_v18pin"
EPOCHS="${EPOCHS:-2000}"

for stage in 0 2; do
  tag="control_v17_s${stage}"
  echo "[$(date +%H:%M:%S)] $tag  (old policy, new criterion and rack)"
  "$PYTHON" scripts/play.py --headless --task "$PLAY" --checkpoint "$V17" \
      --num_envs 128 --episodes 512 --curriculum_stage "$stage" --seed 1070 \
      --grip_axis_metrics \
      --episode_metrics "$OUT/${tag}.npz" --report "$OUT/${tag}.json" \
      > "$OUT/${tag}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)]   exit=$rc $(grep -oE '"success_rate": [0-9.]+' "$OUT/${tag}.json" | head -1)"
done

resume=$(echo "$V17" | sed -n 's/.*_ep_\([0-9]\+\)_.*/\1/p')
target=$((resume + EPOCHS))
echo "[$(date +%H:%M:%S)] TRAIN extract  $resume + $EPOCHS -> $target  run=$RUN"
"$PYTHON" scripts/train.py --headless --task "$TRAIN" \
    --num_envs "${NUM_ENVS:-512}" --seed 70 --robustness_level 0 \
    --max_iterations "$target" --checkpoint "$V17" --run_name "$RUN" \
    > "$OUT/train.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] train exit=$rc"
ls -t "$CKPT_ROOT/$RUN/nn"/*.pth 2>/dev/null | head -3
