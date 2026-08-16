#!/usr/bin/env bash
# Retrain capture against what the chain actually asks of it, then re-close.
#
# Capture certifies at 96.10% alone and is 68% of the chain's failures: 77 of
# 113 are captures overrunning their 6 s budget. The cause is a criterion
# mismatch, the third of this kind found in this project. The skill's success
# tolerance was 20 mm of grip error while the workflow refuses to hand over
# until 10 mm -- twice the precision, inside the same clock.
#
# Both move together because they are one defect measured two ways: the
# tolerance is now read from the workflow's own hand-off constant, and the
# episode goes 6 s -> 10 s because reaching 10 mm takes longer than reaching 20.
# The skill's standalone number is expected to *fall*; the chain is what this is
# judged on.
set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_grasp_l0_seed70_v6align
EPOCHS=2300   # 1500 + 800
SRC="$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth"
E="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/last_zero_g_blade_insertion_contact_ep_3500_rew_148.35918.pth"
I="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"
mkdir -p artifacts/align

say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt() { ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${EPOCHS}_rew_.*\.pth$" | sort | tail -1; }

say "training $RUN"
"$PYTHON" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Grasp-v0 \
    --num_envs 512 --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" \
    --checkpoint "$SRC" --run_name "$RUN" > artifacts/align/train.log 2>&1
say "  exit=$?  -> $(ckpt)"
G="$CKPT_ROOT/$RUN/nn/$(ckpt)"
[ -f "$G" ] || { say "no checkpoint"; exit 1; }
sleep 45

say "certifying capture v6"
EVAL_ENVS=256 GRASP_CKPT="$G" GRASP_VERSION=v6 INTERFACE="the plain grapple pin, chain-aligned 10 mm capture tolerance" \
  bash scripts/certify_demo_policies.sh Grasp

say "state-based chain"
GRASP_CKPT="$G" EXTRACT_CKPT="$E" INSERT_CKPT="$I" TAG=_aligned \
  bash scripts/certify_workflow.sh install

say "vision chain, all three arms"
for arm in blind camera oracle; do
  ARMS=$arm TAG=_aligned HEAD=checkpoints/module_pose_head.pth \
    GRASP_OVERRIDE="$G" bash scripts/certify_vision_workflow.sh
done
say "CAPTURE ALIGNMENT DONE"
