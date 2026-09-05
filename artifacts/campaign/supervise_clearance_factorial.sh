#!/usr/bin/env bash
# The clearance axis, both arms, self-contained.
#
# This is the last boundary axis with a published verdict on it, and it is the
# one where pairing against history is not safe. Two things are wrong with the
# existing arms:
#
#   * `--rack_clearance_scope` was added on 2026-09-03 and neither
#     `artifacts/robustness64/rack_lat_*.npz` (08:04) nor
#     `artifacts/robustness64_channel/rack_lat_*.npz` (13:14) records which
#     scope it ran under. The directory names and `docs/NOW.md` say the first is
#     `guides` and the second `channel`, but the reports do not, and this
#     project has been burned by inferring a flag from a filename before;
#   * both exist at seed 4070 only, so any comparison against them is n = 64
#     and unpaired across seeds.
#
# So rather than re-run one arm and pair it against an arm whose configuration
# has to be inferred, this measures the whole 2x2 fresh:
#
#     rack_lat 6 mm and 16 mm  x  retention absent and fitted
#
# at three seeds, with `--rack_clearance_scope channel` stated explicitly on
# every run. `channel` is the corrected reading -- it translates the lips and
# entry flares with the guides, so the mouth still funnels the module to the
# wall it will run against. The default, `guides`, is what the published sweep
# used and is the defect `docs/NOW.md` records: 6 mm per side goes from 0/64 to
# 36/64 once the mouth moves with the walls.
#
# **Read the delivery column first.** The module-section axis turned out to be
# entirely a capture failure once the pawls were fitted -- precision given
# delivery 1.000 at every seed -- and a clearance point may be the same.
# `scripts/qualify_handoff.py` splits it.
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
SCOPE="--rack_clearance_scope channel"

say "clearance factorial: rack_lat 6 and 16 mm, retention absent and fitted, three seeds"
say "  scope stated explicitly on every run: channel (mouth moves with the walls)"

for seed in $SEEDS; do
  for arm in absent pawls; do
    extra="$SCOPE"
    [ "$arm" = pawls ] && extra="$SCOPE --rack_retention"
    out="artifacts/clearance_factorial/${arm}/seed${seed}"
    mkdir -p "$out"
    say "seed $seed, retention $arm -> $out"
    POINTS="rack_lat_6mm rack_lat_16mm" RACK_LAT="6 16" TRACE=1 \
      SWEEP_EXTRA="$extra" OUT="$out" SEED="$seed" ENVS=64 EPISODES=64 \
      bash scripts/sweep_chain_robustness.sh
    rc=$?
    case "$rc" in
      0) say "  seed $seed $arm finished" ;;
      *) say "  seed $seed $arm exited $rc" ;;
    esac
    # Verify by the artifact, never by the status line.
    for point in rack_lat_6mm rack_lat_16mm; do
      if [ -f "$out/${point}.npz" ]; then
        say "    $point: episodes written"
      else
        say "    $point: NO EPISODES WRITTEN -- not usable"
      fi
    done
  done
done

say "clearance factorial done"
