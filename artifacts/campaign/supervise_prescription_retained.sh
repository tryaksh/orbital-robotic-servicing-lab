#!/usr/bin/env bash
# The published refutation, re-run with the fixture the workcell actually has.
#
# `evidence/rack_prescription_paired_n192.json` is the only published *refutation*
# in the boundary set: rebuilding the destination channel at the design library's
# prescribed clearance took the chain from 110/192 to 64/192, 28 gained against
# 74 lost, McNemar two-sided p = 5.9e-06. The conclusion drawn was that the
# library's two-sided window is half wrong and the seated-rest upper bound does
# not govern.
#
# Both arms of that comparison ran with `destination_rack_retention.enabled =
# false`. The failure the prescription was supposed to fix -- 77 of 192 episodes
# arriving, seating and missing the terminal gate -- is now known to be the
# module coasting after release with nothing holding it, and fitting the pawls
# removes all 77 (110/192 -> 187/192, p = 1.3e-23). So the refutation was
# measured against a failure mode that the prescription could not have fixed and
# that the fixture does fix.
#
# This fills the missing cell of a 2x2 that is otherwise complete:
#
#     relief 4.61 mm, no pawls   110/192   artifacts/traced_nominal
#     relief 0.00 mm, no pawls    64/192   artifacts/relief0_seed*
#     relief 4.61 mm, pawls      187/192   artifacts/retention_nominal
#     relief 0.00 mm, pawls          ?     this run
#
# Two outcomes, both worth the twenty minutes. If this lands near 187/192 the
# prescription is neither harmful nor helpful once the bay holds the module, and
# the published refutation was an artefact of the missing fixture. If it stays
# low, the upper bound genuinely does not govern and the refutation stands --
# now properly scoped, and stronger for having survived.
#
# Matched to `supervise_retention_nominal.sh` in every other respect: same
# seeds, same 64 environments, same default step budget, same checkpoints. The
# per-phase budgets are identical across all existing arms, so the historical
# STEPS=6000 on the relief0 cohort is not a confound -- the phase budgets bind
# first.
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

SEEDS="4070 5070 6070"
for seed in $SEEDS; do
  [ -f "artifacts/retention_nominal/seed${seed}/nominal.npz" ] || {
    say "no retained-baseline arm for seed $seed; run supervise_retention_nominal.sh first"; exit 1; }
done

say "prescription arm with pawls: relief 0.0, --rack_retention, three seeds"

for seed in $SEEDS; do
  out="artifacts/prescription_retained/seed${seed}"
  mkdir -p "$out"
  say "seed $seed -> $out"
  POINTS=nominal TRACE=1 SWEEP_EXTRA=--rack_retention RELIEF=0.0 \
    OUT="$out" SEED="$seed" ENVS=64 EPISODES=64 \
    bash scripts/sweep_chain_robustness.sh
  rc=$?
  case "$rc" in
    0) say "  seed $seed finished" ;;
    *) say "  seed $seed exited $rc"; tail -3 "$out/nominal.log" 2>/dev/null ;;
  esac
  # Verify by the artifact, never by the status line. A Windows Application
  # Control block took two runs down on 2026-09-05 while the driver exited 0.
  if [ -f "$out/nominal.npz" ]; then
    relief=$(grep -aoE "channel relief [0-9.]+ mm per side" "$out/nominal.log" | head -1)
    say "  episodes written; $relief"
  else
    say "  NO EPISODES WRITTEN for seed $seed -- this run is not usable"
  fi
done

say "prescription retained done"
