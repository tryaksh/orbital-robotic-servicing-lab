#!/usr/bin/env bash
# One reference workcell, three arms, and a question that has two answers.
#
# Twelve episodes in 192 still fail when the destination channel is rebuilt at
# the design library's prescribed clearance, even with the retention pawls
# fitted. They reach the insert phase, the insertion predicate never fires, and
# the module stops at x = 0.220 m against a seated plane at 0.676 m with a peak
# orientation error of 83.6 to 90.8 mrad. Two explanations fit that equally well
# through the instrumentation that existed:
#
#   YAW   -- the module rotates about the vertical axis and wedges between the
#            two side guides, so lateral clearance governs and the library's
#            2c/theta bound should have caught it. It did not: at these angles
#            it permits 250.7 mm of engagement and the module wedges at 88.0.
#   PITCH -- the module rotates about a horizontal axis and wedges under the
#            bay's vertical lead-in, so the governing clearance is the vertical
#            one and the lateral bound was never the relevant gate.
#
# `orientation_error_rad` is an axis-angle norm and reads identically for both.
# The insert trace now carries the full quaternion, so one run separates them.
#
# **The reference workcell, fixed for every arm here.** Task
# Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0, workflow relocate, curriculum
# stage 0, 64 environments, 64 episodes, seeds 4070/5070/6070, the three frozen
# checkpoints named below, robot-carried transit on the rail, latch on release
# as a fixed joint at 20 kN / 1 kN-m, compliant mating capped at 1 kN, and the
# unchanged success criterion. Nothing in that list varies between arms.
#
# Three arms, one variable each:
#
#   reference      relief 4.61 mm, pawls   -- must reproduce 187/192, which is
#                                             how the new trace column is shown
#                                             not to have perturbed anything
#   prescribed     relief 0.00 mm, pawls   -- must reproduce 175/192, and gives
#                                             the twelve jams with orientation
#   prescribed_bare relief 0.00 mm, none   -- the retention-idealisation check
#
# The third arm exists because the pawls are an idealisation: a break-rated
# fixed joint between rack and module, with `hardware_geometry: null`. The jams
# happen before seating, so retention is never in their path and the conclusion
# should not move at all. If it does, the idealisation is load-bearing and that
# is more important than the jam.
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

say "jam mechanism: three arms on one reference workcell, three seeds"

run_arm () {
  local arm="$1" relief="$2" extra="$3"
  for seed in 4070 5070 6070; do
    out="artifacts/jam_mechanism/${arm}/seed${seed}"
    mkdir -p "$out"
    say "$arm seed $seed -> $out"
    POINTS=nominal TRACE=1 RELIEF="$relief" SWEEP_EXTRA="$extra" \
      OUT="$out" SEED="$seed" ENVS=64 EPISODES=64 \
      bash scripts/sweep_chain_robustness.sh
    rc=$?
    # Verify by the artifact, never by the status line.
    if [ -f "$out/nominal.npz" ]; then
      say "  $arm seed $seed exit=$rc, episodes written"
    else
      say "  $arm seed $seed exit=$rc but NO EPISODES WRITTEN -- not usable"
    fi
  done
}

run_arm reference       0.0046125 "--rack_retention"
run_arm prescribed      0.0       "--rack_retention"
run_arm prescribed_bare 0.0       ""

say "jam mechanism done"
