#!/usr/bin/env bash
# The insert policy, on a reward balanced for the stroke it has to drive.
#
# Measured on v16pin over 2,403 episodes: the module ends a median of 176 mm
# short with 20.7 mm of lateral error, holding the grip perfectly and spending
# every step of its clock. Two things in the objective explain that and both are
# now derived from the stroke rather than inherited from a 167 mm one:
#
#  * the progress cost was six times more sensitive to a millimetre of lateral
#    jitter than to a control step of axial progress;
#  * the only continuous pressure toward alignment was worth about 4 an episode
#    against a success worth 30, and it switched off over most of the stroke.
#
# From scratch. The policy being replaced learned to hold the module and wander,
# and that is the behaviour the old objective paid for.

set -u
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN="${RUN:-grapple_insert_l0_seed70_v17balance}"

OUT=artifacts/insert_balance EPOCHS="${EPOCHS:-1400}" RUN="$RUN" \
  bash scripts/train_insert_stroke.sh
NEW=$(ls -t "$CKPT_ROOT/$RUN/nn"/last_*.pth 2>/dev/null | head -1)
BEST="$CKPT_ROOT/$RUN/nn/zero_g_blade_insertion_contact.pth"
[ -z "$NEW" ] && { echo "[$(date +%H:%M:%S)] NO CHECKPOINT"; exit 1; }
echo "[$(date +%H:%M:%S)] final=$NEW"

for tag in final best; do
  ck="$NEW"; [ "$tag" = "best" ] && ck="$BEST"
  [ -f "$ck" ] || continue
  echo "[$(date +%H:%M:%S)] probe $tag"
  "C:/isaac-sim/python.bat" scripts/play.py --headless \
      --task Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-Play-v0 --checkpoint "$ck" \
      --num_envs 64 --episodes 256 --curriculum_stage 0 --seed 1070 --grip_axis_metrics \
      --episode_metrics "artifacts/insert_balance/probe_${tag}.npz" \
      --report "artifacts/insert_balance/probe_${tag}.json" \
      > "artifacts/insert_balance/probe_${tag}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)]   $tag exit=$rc $(grep -oE '"success_rate": [0-9.]+' "artifacts/insert_balance/probe_${tag}.json" | head -1)"
done
echo "[$(date +%H:%M:%S)] INSERT REBALANCE DONE"
