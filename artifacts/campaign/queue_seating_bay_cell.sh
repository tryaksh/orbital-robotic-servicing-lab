#!/usr/bin/env bash
# The one missing cell of the seating experiment, and the gate it is measured against.
#
# WHAT IS ALREADY MEASURED
#
# The force-feedback seating policy `v33force` epoch 3000
# (sha256 86599FC2...) has three arms on disk, all on that one checkpoint:
#
#   skill alone, InsertForce-Play-v0, seeds 1070/2070/3070, 128 envs
#       2,977 / 3,001 = 99.20%   passes the 95% skill gate
#   chain, TwoSlotWorkflowForce-v0, seeds 4070/5070/6070, channel throat
#   11.065 mm per side (relief 0.0), guarded advance
#       23 / 96 = 23.96%   (and 7/24 at 8 envs)
#   the same, learned seating policy
#       24 / 96 = 25.00%   (and 8/24 at 8 envs)
#   chain, the same task and seeds at the shipped relieved throat
#   (relief 0.0046125 m, 15.678 mm per side), learned seating policy
#       4 / 24 = 16.67%
#
# THE CELL THAT IS MISSING
#
# There is no guarded-advance arm at the relieved throat on the force task, so
# the bay effect and the controller effect cannot be separated: the published
# 20/24 and 22/24 guarded numbers are the *state* task, which differs from the
# force task by a contact sensor and seven observation values. Comparing across
# them mixes a task change into a bay change, which is the mistake this
# repository has paid for most.
#
# This run fills it. Four cells, two factors, one checkpoint set:
#
#                        guarded advance      learned policy
#   throat 11.065 mm     7/24, 23/96          8/24, 24/96
#   throat 15.678 mm     THIS RUN             4/24
#
# THE GATE, REGISTERED BEFORE THE RUN
#
# The chain gate is unchanged at 95% and nothing here moves it. What this run
# decides is narrower, and both readings are stated now so neither can be chosen
# afterwards:
#
#   If the guarded arm at the relieved throat lands near the published state-task
#   rate (roughly 17-22 of 24), then the force task is not what costs the
#   controller anything and the 54-point drop from 20/24 to 7/24 belongs to the
#   BAY. The bay built to the tool's own prescription is the harder bay, and the
#   3.897 mm of relief that the tool calls inadmissible is what the scripted
#   controller has been living on.
#
#   If instead it lands near 7/24, then the drop belongs to the TASK, the
#   11.065 mm cohort says nothing about the bay, and the force-task comparison
#   is the only one the policy arms may be read against.
#
# Either way the seating phase does not change hands: the decision rule in
# `scripts/report_seating_head_to_head.py` requires the policy to win pooled AND
# on every shared seed, and at 11.065 mm it loses seed 4070 (1/8 against 2/8).
#
# One change from the policy arm it pairs with: `--insert_controller` is left at
# its default. Same task, same three held-out seeds, same eight environments,
# same capture, extraction and insert checkpoints -- the insert weights are
# loaded and never stepped, exactly as they are in every guarded arm, so the
# policy-set hash matches and the pairing is real.
#
# About 25 minutes.
set -u
cd /d/6axis-space-robotics || exit 1

ROOT="logs/rl_games/zero_g_blade_insertion_contact"
INSERT="$ROOT/grapple_insert_l0_seed70_v33force/nn/last_zero_g_blade_insertion_contact_ep_3000_rew_98.33571.pth"

say () { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -f "$INSERT" ]; then say "missing insert checkpoint $INSERT"; exit 66; fi
say "insert checkpoint sha256 $(sha256sum "$INSERT" | cut -c1-32)"

TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflowForce-v0" \
INSERT_CKPT="$INSERT" \
CERT_TAG="insert_v33force_relieved_chain_guarded" \
CERT_TITLE="Robot-carried relocation, scripted guarded advance, force task, shipped relieved throat 15.678 mm per side" \
SEEDS="4070 5070 6070" \
  bash scripts/run_robot_carried.sh certify
rc=$?
say "certify exit=$rc"
say "-> evidence/workflow_robot_carried_insert_v33force_relieved_chain_guarded_certification.json"
