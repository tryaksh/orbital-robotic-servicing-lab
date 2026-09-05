#!/usr/bin/env bash
# The module-section axis, re-run with the fixture the workcell actually has.
#
# `evidence/boundary_failure_modes_n192_v1.json` and the "not qualified"
# decision rest on two points: `section_120x16` and `section_140x26`, at 192
# episodes each. Both ran with `destination_rack_retention.enabled = false`, and
# fitting the pawls is now known to be worth 77 episodes in 192 at nominal --
# 110/192 to 187/192, p = 1.3e-23 -- because the failure it removes is the
# module coasting after release with nothing holding it.
#
# So neither section verdict is yet a statement about the workcell that was
# designed. These two points are the ones worth re-running first: they are the
# axis the boundary decision names, and they are the only sweep points besides
# nominal with a three-seed retention-absent arm to pair against.
#
# `rack_lat_6mm` and `rack_lat_16mm` are deliberately left out. They exist at
# one seed only, so re-running them with pawls would produce an unpaired
# comparison at n = 64, which is not what the clearance axis needs.
#
# Matched to `supervise_retention_nominal.sh` in every other respect: same
# seeds, 64 environments, default step budget, same checkpoints, one flag.
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

# Refuse rather than produce an unpaired arm: every point here must have a
# retention-absent counterpart at all three seeds.
SEEDS="4070 5070 6070"
declare -A BASE=( [4070]=artifacts/robustness64_corrected \
                  [5070]=artifacts/robustness64_seed5070 \
                  [6070]=artifacts/robustness64_seed6070 )
for point in section_120x16 section_140x26; do
  for seed in $SEEDS; do
    [ -f "${BASE[$seed]}/${point}.npz" ] || {
      say "no retention-absent arm for $point seed $seed; refusing"; exit 1; }
  done
done

say "module-section axis with pawls: two points, three seeds"

for seed in $SEEDS; do
  out="artifacts/section_retained/seed${seed}"
  mkdir -p "$out"
  say "seed $seed -> $out"
  POINTS="section_120x16 section_140x26" TRACE=1 SWEEP_EXTRA=--rack_retention \
    OUT="$out" SEED="$seed" ENVS=64 EPISODES=64 \
    bash scripts/sweep_chain_robustness.sh
  rc=$?
  case "$rc" in
    0) say "  seed $seed finished" ;;
    *) say "  seed $seed exited $rc" ;;
  esac
  # Verify by the artifact, never by the status line.
  for point in section_120x16 section_140x26; do
    if [ -f "$out/${point}.npz" ]; then
      say "  $point: episodes written"
    else
      say "  $point: NO EPISODES WRITTEN -- not usable"
    fi
  done
done

say "section retained done"
