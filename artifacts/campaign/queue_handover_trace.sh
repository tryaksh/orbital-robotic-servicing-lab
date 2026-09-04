#!/usr/bin/env bash
# Re-measure the four decisive boundary arms with the hand-over state recorded.
#
# **Why this needs GPU rather than another pass over the archives.** The episode
# archive stores each row at the moment of *judgement* -- `_freeze` in
# run_workflow_demo.py says so in its own docstring -- so every recorded grip
# error, attitude and velocity describes the state the outcome was decided in.
# A statistic built on that is a concurrent association, and for velocity it is
# circular: SEATED_CONDITIONS contains linear_velocity and angular_velocity, so
# an episode fails partly because its velocity is high. That is what retracted
# criterion_retention_v1.json.
#
# The question worth answering is whether a deviation *at hand-over* governs the
# outcome, because that is what makes a closed-form bound predictive rather than
# descriptive. Answering it needs the value at the transition into the seating
# phase, which `--handoff_trace` already records and no boundary arm was run
# with. TRACE=1 turns it on; nothing else changes, so these arms are directly
# comparable with the ones already published.
#
# Four points, chosen because each one already has a known answer to check
# against:
#
#   nominal              the floor. Retention should find nothing to explain.
#   section_120x16       the grip criterion violated. Its quantity should govern.
#   rack_lat_6mm         the entry criterion violated, corrector present. The
#                        jam mode does not occur, so there should be nothing.
#   rack_lat_6mm, no flares   the same point with the corrector deleted. If
#                        hand-over attitude governs anywhere, it is here.
#
# If hand-over attitude ranks the jams in the fourth arm and not the third, the
# transfer rule becomes a prediction. If it does not, the rule stays an
# observation and the paper says so -- which is the current position anyway, so
# this can only improve it.
set -u
cd /d/6axis-space-robotics || exit 1
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"

# Behind everything already queued. The submission gate and the force-feedback
# verification come first; this is a methods result and can wait for them.
until grep -q "traced rung done" artifacts/campaign/trace_rung.log 2>/dev/null; do sleep 300; done
echo "[$(date +%H:%M:%S)] hand-over traces; the evaluation slot is free"

echo "[$(date +%H:%M:%S)] nominal and the grip-violating section, traced"
TRACE=1 POINTS="nominal section_120x16" EPISODES=64 ENVS=64 STEPS=6000 \
  OUT=artifacts/handover_section SEED=4070 \
  bash scripts/sweep_chain_robustness.sh
rc=$?
echo "[$(date +%H:%M:%S)] section arms exit=$rc"

echo "[$(date +%H:%M:%S)] 6 mm per side with the flares fitted, traced"
TRACE=1 POINTS="rack_lat_6mm" EPISODES=64 ENVS=64 STEPS=6000 \
  OUT=artifacts/handover_flare_fitted SEED=4070 \
  SWEEP_EXTRA="--rack_clearance_scope channel" \
  bash scripts/sweep_chain_robustness.sh
rc=$?
echo "[$(date +%H:%M:%S)] flares fitted exit=$rc"

echo "[$(date +%H:%M:%S)] the same point with the flares deleted, traced"
TRACE=1 POINTS="rack_lat_6mm" EPISODES=64 ENVS=64 STEPS=6000 \
  OUT=artifacts/handover_flare_removed SEED=4070 \
  SWEEP_EXTRA="--rack_clearance_scope channel --remove_entry_flares" \
  bash scripts/sweep_chain_robustness.sh
rc=$?
echo "[$(date +%H:%M:%S)] flares removed exit=$rc"

echo "[$(date +%H:%M:%S)] hand-over traces done"
