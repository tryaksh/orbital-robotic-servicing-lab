#!/usr/bin/env bash
# Pay the extract policy to arrive settled, which nothing ever did.
#
# Every extraction figure this project published was measured under a criterion
# the code no longer contains: the settled-enough velocity limits were derived
# and tightened on 2026-08-15, and 0 of extract v8's 6,156 counted successes
# satisfies the linear limit now in force, the best of them by a factor of 3.1.
# The only extraction measured against the current criterion is v10, at 0.00%.
#
# v10's failure profile names the cause. It trained to a HIGHER reward than v8 --
# 158.7 against 148.4 -- while losing the grip in 8,988 of 9,010 episodes. The
# success predicate asks for a module that is clear AND settled; every dense term
# in the objective was about travel, so a progress term weighted 12 bought speed
# and the policy pulled through the line and out of the workspace. Velocity
# entered the objective only through a sparse terminal predicate, which is no
# gradient at all.
#
# extraction_settling_penalty reads EXTRACTION_LINEAR_VELOCITY_LIMIT and
# EXTRACTION_ANGULAR_VELOCITY_LIMIT rather than restating them, is exactly zero
# below them, and ramps over the last 60 mm so the module decelerates *into* the
# line instead of being charged once it is already past.
#
# The precondition the force-shaping work failed on is checked, not assumed:
# blade_velocity is already in the extract observation, so the policy can
# perceive what it is being asked to regulate and no dimension changes.
#
# Resuming across a reward change, and across the criterion change v8 predates.
# Both are reward/physics changes, which this repository allows and requires to
# be stated.
set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_extract_l0_seed70_v11settle
EPOCHS=4700   # 3500 + 1200
SRC="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/last_zero_g_blade_insertion_contact_ep_3500_rew_148.35918.pth"
mkdir -p artifacts/extract_settle

say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt() { ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${EPOCHS}_rew_.*\.pth$" | sort | tail -1; }

say "training $RUN from extract v8"
"$PYTHON" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Extract-v0 \
    --num_envs 512 --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" \
    --checkpoint "$SRC" --run_name "$RUN" > artifacts/extract_settle/train.log 2>&1
say "  exit=$?  -> $(ckpt)"
E="$CKPT_ROOT/$RUN/nn/$(ckpt)"
[ -f "$E" ] || { say "no checkpoint"; exit 1; }
sleep 45

say "certifying extract v11settle"
EVAL_ENVS=256 EXTRACT_CKPT="$E" EXTRACT_VERSION=v11settle \
  INTERFACE="the plain grapple pin, judged on the derived settling limits" \
  bash scripts/certify_demo_policies.sh Extract

# certify_workflow.sh's defaults are capture v3 and insert v6, which is exactly
# what the 14.06% remove_clock run used, so extract is the only thing that
# differs between that number and this one. Verified against the recorded
# checkpoint hashes rather than assumed: remove_clock and remove_settle both
# carry capture AF579F5A and insert 7E9A0C33.
say "removal chain"
EXTRACT_CKPT="$E" TAG=_settle bash scripts/certify_workflow.sh remove
say "EXTRACT SETTLING DONE"
