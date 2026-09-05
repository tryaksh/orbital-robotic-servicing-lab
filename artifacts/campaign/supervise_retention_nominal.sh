#!/usr/bin/env bash
# The boundary sweep was run with the destination rack's retention switched off.
#
# `artifacts/traced_nominal/*/nominal_report.json` records
# `destination_rack_retention.enabled = false`, `mechanism = "none"`. Every
# boundary verdict this project has published -- the module-section points, the
# clearance axis, "not qualified" -- rests on that sweep, and the success
# definition re-checks the module after a **0.70 s free-module window**. In zero
# gravity with no pawls, a module released with any residual velocity coasts,
# and nothing brings it back.
#
# That is what the settle traces show: all 186 episodes are inside the 2.5 mm
# criterion at their worst sample while the robot holds them, and 41% drift out
# afterwards, with drift tracking velocity x free-time at rho = +0.84. Meanwhile
# a cohort that *does* fit the pawls reports `max_rack_to_module_position_drift_m
# = 0.0` on every environment where they engaged.
#
# So this re-runs the same point, same three seeds, same checkpoints, with
# `--rack_retention` and nothing else changed. It pairs episode-for-episode
# against `artifacts/traced_nominal`, which is why it must use the same seeds
# and environment count: the two arms are one experiment, not two cohorts.
#
# `SWEEP_EXTRA` is the sweep script's own mechanism for exactly this -- one
# driver flag applied to every point so a sweep can be compared against itself.
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

# The baseline arm must already exist, or there is nothing to pair against and
# this becomes an unpaired cohort at a different time on a different machine
# state -- which is the reading this project has been burned by before.
SEEDS="4070 5070 6070"
for seed in $SEEDS; do
  [ -f "artifacts/traced_nominal/seed${seed}/nominal.npz" ] || {
    say "no baseline arm for seed $seed; run supervise_traced_nominal.sh first"; exit 1; }
done

say "retention arm: three seeds, 64 environments, --rack_retention, nothing else changed"

for seed in $SEEDS; do
  out="artifacts/retention_nominal/seed${seed}"
  mkdir -p "$out"
  say "seed $seed -> $out"
  POINTS=nominal TRACE=1 SWEEP_EXTRA=--rack_retention OUT="$out" SEED="$seed" ENVS=64 EPISODES=64 \
    bash scripts/sweep_chain_robustness.sh
  rc=$?
  case "$rc" in
    0) say "  seed $seed finished" ;;
    *) say "  seed $seed exited $rc"; tail -3 "$out/nominal.log" 2>/dev/null ;;
  esac
  # Verify by the artifact, never by the status line.
  if [ -f "$out/nominal.npz" ]; then
    say "  episodes written$([ -f "$out/nominal_trace.npz" ] && echo ', trace present')"
  else
    say "  NO EPISODES WRITTEN for seed $seed -- this run is not usable"
  fi
done

say "retention nominal done"
