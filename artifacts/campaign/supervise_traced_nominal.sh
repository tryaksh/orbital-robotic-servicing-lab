#!/usr/bin/env bash
# The cohort that varies has no trace, and the cohorts with traces do not vary.
#
# That sentence is the whole blocker. `artifacts/robustness192_section/nominal.npz`
# is 192 episodes at 110 successes -- the one large cohort whose outcomes
# straddle the criterion -- and it was run before `--handoff_trace` existed as a
# default, so it records nothing about the state the seating step was handed.
# Every cohort that *does* carry a trace sits deep inside the acceptable region:
# pooled over all eighteen of them, only four have both outcomes present at all,
# and those four contain six failures between them. Six events cannot separate
# two models.
#
# So this re-runs the nominal point at the same three seeds with TRACE=1 and
# nothing else changed. It is the smallest run that unblocks the day-14 test.
#
# **It carries its own control.** The existing cohort scored 110/192. If this
# reproduces that, the trace is non-perturbing and the pre-handoff state it
# records may be joined to the outcomes already published. If it does not, that
# is the more important result and the trace instrumentation is itself the
# finding -- either way the comparison is made before anything is fitted.
#
# One script, consecutive lines, no queue shell parked in a sleep loop. Twelve
# of those killed the machine on 2026-09-04 and the rule that followed is in
# AGENTS.md.
set -u
cd /d/6axis-space-robotics || exit 1

PYTHON="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"

say () { echo "[$(date +%H:%M:%S)] $*"; }

for ckpt in "$GRASP_CKPT" "$EXTRACT_CKPT" "$INSERT_CKPT"; do
  [ -f "$ckpt" ] || { say "missing checkpoint $ckpt; refusing to start"; exit 1; }
done

# The three seeds `robustness192_section/nominal.npz` was pooled from. Asserted
# rather than assumed: a sweep whose points differ only by a computed suffix
# must refuse to start if any suffix is empty or repeats, because that failure
# is silent, total, and only visible after the GPU is spent.
SEEDS="4070 5070 6070"
seen=""
for seed in $SEEDS; do
  [ -n "$seed" ] || { say "empty seed in the list"; exit 1; }
  case " $seed " in *" $seen "*) say "repeated seed $seed"; exit 1 ;; esac
  seen="$seen $seed"
done

say "traced nominal: three seeds, 64 environments, TRACE=1, nothing else changed"

for seed in $SEEDS; do
  out="artifacts/traced_nominal/seed${seed}"
  mkdir -p "$out"
  say "seed $seed -> $out"
  POINTS=nominal TRACE=1 OUT="$out" SEED="$seed" ENVS=64 EPISODES=64 \
    bash scripts/sweep_chain_robustness.sh
  rc=$?
  case "$rc" in
    0) say "  seed $seed finished" ;;
    *) say "  seed $seed exited $rc"; tail -3 "$out/nominal.log" 2>/dev/null ;;
  esac
  # Verify by the artifact, never by the log line.
  if [ -f "$out/nominal_trace.npz" ]; then
    say "  trace present: $(du -h "$out/nominal_trace.npz" | cut -f1)"
  else
    say "  NO TRACE WRITTEN for seed $seed -- this run is not usable"
  fi
done

say "traced nominal done"
