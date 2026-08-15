#!/usr/bin/env bash
# Derive the capture latch's required rating, by measuring it.
#
# The interface question this answers is the one section 8 of
# docs/service_interface_spec.md has been asking since the yaw result: a
# parallel-jaw grip on a passive feature cannot hold the module's attitude, so
# what does an active latch have to be rated for before extraction works?
#
# The instrument is the *already trained* extract v4 policy, run unchanged at
# every rating. Nothing is retrained, so any difference between points is the
# interface and cannot be a training artefact. That is the control the yoke
# experiment did not have.
#
# --grip_axis_metrics is on throughout, because the decomposition is what
# corrected this project's diagnosis: the residual rotation is 0.325 rad about
# the module's transverse axis and only 0.096 about the closing axis the
# anti-yaw yoke was built to oppose.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
EXTRACT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v4/nn/last_zero_g_blade_insertion_contact_ep_1200_rew__162.91257_.pth}"
RATINGS="${RATINGS:-0 10 20 40 80 160}"
ENVS="${ENVS:-256}"
EPISODES="${EPISODES:-512}"
STAGE="${STAGE:-0}"
SEED="${SEED:-1070}"
OUT=artifacts/latch
mkdir -p "$OUT"

if [ ! -f "$EXTRACT" ]; then echo "MISSING $EXTRACT"; exit 1; fi

for rating in $RATINGS; do
  echo "[$(date +%H:%M:%S)] latch rating ${rating} N-m"
  # Rating 0 is the control: the latch term is present and contributes nothing,
  # so it reproduces the passive interface on exactly this code path.
  "$PYTHON" scripts/play.py --headless \
      --task Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0 \
      --checkpoint "$EXTRACT" \
      --num_envs "$ENVS" \
      --episodes "$EPISODES" \
      --curriculum_stage "$STAGE" \
      --seed "$SEED" \
      --latch_rated_torque_nm "$rating" \
      --grip_axis_metrics \
      --report "$OUT/rating_${rating}.json" \
      > "$OUT/rating_${rating}.log" 2>&1
  echo "[$(date +%H:%M:%S)]   exit=$?"
done

echo "[$(date +%H:%M:%S)] LATCH RATING SWEEP DONE"
