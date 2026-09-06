#!/usr/bin/env bash
# The prediction test, on geometry that has never been run.
#
# The jam mechanism experiment separates two explanations for the twelve
# failures the prescribed clearance causes: a yaw about the vertical axis,
# wedging between the side guides, or a pitch about a horizontal axis, wedging
# under the vertical lead-in. Whichever it is, it makes a directional prediction
# about module geometry, and the prediction is testable on configurations that
# have never been run here -- so it cannot have informed the explanation.
#
# `--module_cross_section_m` takes width and thickness separately. The two
# section points this project already has vary both at once -- 120 x 16 mm is
# narrower *and* thinner, 140 x 26 mm wider *and* thicker -- so neither
# separates the axes. These two vary one each against the nominal 130 x 20:
#
#     140 x 20 mm   wider, nominal thickness    -> less LATERAL clearance only
#     130 x 26 mm   nominal width, thicker      -> less VERTICAL clearance only
#
# If the jam is a yaw, 140 x 20 jams more at the prescribed clearance and
# 130 x 26 does not. If it is a pitch, the reverse. A result where both move
# together, or neither does, falsifies the framing rather than either branch,
# and is the outcome worth watching for.
#
# Run at the prescribed relief with the pawls fitted, because that is the
# configuration the twelve jams occur in. Everything else is the reference
# workcell, unchanged: same task, same three checkpoints, same seeds, same 64
# environments, same success criterion.
set -u
cd /d/6axis-space-robotics || exit 1

ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"

say () { echo "[$(date +%H:%M:%S)] $*"; }

for ckpt in "$GRASP_CKPT" "$EXTRACT_CKPT" "$INSERT_CKPT"; do
  [ -f "$ckpt" ] || { say "missing checkpoint $ckpt; refusing to start"; exit 1; }
done

# Refuse unless the arm this is predicting against exists, so the comparison
# cannot silently become a standalone number.
for seed in 4070 5070 6070; do
  [ -f "artifacts/jam_mechanism/prescribed/seed${seed}/nominal.npz" ] || {
    say "no prescribed arm for seed $seed; run supervise_jam_mechanism.sh first"; exit 1; }
done

say "prediction test: 140x20 and 130x26 at the prescribed relief, pawls, three seeds"

run_section () {
  local tag="$1" width="$2" thickness="$3"
  for seed in 4070 5070 6070; do
    out="artifacts/jam_prediction/${tag}/seed${seed}"
    mkdir -p "$out"
    say "$tag seed $seed -> $out"
    POINTS=nominal TRACE=1 RELIEF=0.0 \
      SWEEP_EXTRA="--rack_retention --module_cross_section_m $width $thickness" \
      OUT="$out" SEED="$seed" ENVS=64 EPISODES=64 \
      bash scripts/sweep_chain_robustness.sh
    rc=$?
    # Verify by the artifact, never by the status line.
    if [ -f "$out/nominal.npz" ]; then
      say "  $tag seed $seed exit=$rc, episodes written"
    else
      say "  $tag seed $seed exit=$rc but NO EPISODES WRITTEN -- not usable"
    fi
  done
}

run_section wider_140x20 0.140 0.020
run_section thicker_130x26 0.130 0.026

say "jam prediction done"
