#!/usr/bin/env bash
# Train insert across the states the capture actually hands it.
#
# Measured, and it is the whole of the chain gap. `run_workflow_demo.py
# --handoff_trace` records the state every phase hands over in. Over 576 chained
# installations the arm pose the capture leaves for insert sits 0.148 rad from
# the nominal head-on pose on its worst axis at the median -- almost all of it
# wrist_1 -- while insert's own reset drew 0.020 rad of noise around that
# nominal. Even the 5th-percentile hand-off is three times the widest single
# joint value that reset could produce.
#
#   insert v6 on its own reset          95.57%
#   insert v6 on the measured hand-offs 31.25%
#
# That is not something capture v6 introduced: the same trace on capture v5 gives
# 0.138 rad, so this gap has been there the whole time and is what separates
# 95.57% alone from 84.38% chained.
#
# The reset samples measured poses rather than widening the noise, because a
# hand-off is a point on a manifold and not a box: the deviation is large *and*
# the grip error is 12.5 mm, since the capture servoed there. Per-joint noise
# wide enough to reach 0.28 rad on wrist_1 was tried first and is degenerate --
# it produces a large joint error and a large grip error together, the fingers
# close on nothing, and 534 of 534 episodes lost the grip at reset.
#
# The clock is deliberately untouched. Widening insert's episode would widen the
# chain's insert budget with it, because PHASE_BUDGET_S reads that field, and
# that is the "make the phase fit" move this repository forbids.
set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_insert_l0_seed70_v7align
EPOCHS=4200   # 3200 + 1000
SRC="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"
G="$CKPT_ROOT/grapple_grasp_l0_seed70_v6align/nn/last_zero_g_blade_insertion_contact_ep_2300_rew__35.843285_.pth"
E="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/last_zero_g_blade_insertion_contact_ep_3500_rew_148.35918.pth"
mkdir -p artifacts/insert_align

say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt() { ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${EPOCHS}_rew_.*\.pth$" | sort | tail -1; }

# The matched control. Changing the reset changes the task, so v7align's number
# is not comparable with v6's 95.57%; it is comparable with this one, and without
# it any improvement could be the task getting easier rather than the policy
# getting better.
say "control: insert v6 on the hand-off reset"
EVAL_ENVS=256 INSERT_CKPT="$SRC" INSERT_VERSION=v6onhandoff \
  INTERFACE="the plain grapple pin, reset from measured capture hand-offs" \
  bash scripts/certify_demo_policies.sh Insert

# Resuming across a reset-distribution change, which is an event change and
# alters neither the observation nor the action dimension. Stated because this
# repository requires it to be stated.
say "training $RUN from insert v6"
"$PYTHON" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Insert-v0 \
    --num_envs 512 --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" \
    --checkpoint "$SRC" --run_name "$RUN" > artifacts/insert_align/train.log 2>&1
say "  exit=$?  -> $(ckpt)"
I="$CKPT_ROOT/$RUN/nn/$(ckpt)"
[ -f "$I" ] || { say "no checkpoint"; exit 1; }
sleep 45

say "certifying insert v7align"
EVAL_ENVS=256 INSERT_CKPT="$I" INSERT_VERSION=v7align \
  INTERFACE="the plain grapple pin, reset from measured capture hand-offs" \
  bash scripts/certify_demo_policies.sh Insert

say "state-based install chain"
GRASP_CKPT="$G" EXTRACT_CKPT="$E" INSERT_CKPT="$I" TAG=_insertalign \
  bash scripts/certify_workflow.sh install

say "vision chain, all three arms"
for arm in blind camera oracle; do
  ARMS=$arm TAG=_insertalign HEAD=checkpoints/module_pose_head.pth \
    GRASP_OVERRIDE="$G" INSERT_OVERRIDE="$I" bash scripts/certify_vision_workflow.sh
done
say "INSERT ALIGNMENT DONE"
