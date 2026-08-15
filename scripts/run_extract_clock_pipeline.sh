#!/usr/bin/env bash
# Certify the extract skill on its lengthened clock, then re-close both chains.
#
# Waits for the fine-tune to write its final checkpoint, so this can be launched
# while training is still running and left alone.
#
# The experiment: extract certified at 0.00% on a 15 s episode whose median
# cycle time was 15.000 s, with the module reaching 458 mm of the required 495.
# That is insert v5's situation, and insert v5 -> v6 was fixed by lengthening
# the episode *and* fine-tuning against the new horizon, which took it from
# 6.96% to 95.57%. Replaying v4 unchanged at 25 s and 40 s was measured first
# and does not work -- it converts 449 timeouts into 512 lost grips -- so the
# fine-tune is the load-bearing half.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_extract_l0_seed70_v6clock
FINAL_EPOCH=1800
GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v3/nn/last_zero_g_blade_insertion_contact_ep_700_rew__36.020187_.pth"
INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"
mkdir -p artifacts evidence

say() { echo "[$(date +%H:%M:%S)] $*"; }

ckpt_at() {
  ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${FINAL_EPOCH}_rew_.*\.pth$" | sort | tail -1
}

say "waiting for $RUN epoch $FINAL_EPOCH"
for _ in $(seq 1 240); do
  [ -n "$(ckpt_at)" ] && break
  sleep 30
done
EXTRACT_CKPT="$CKPT_ROOT/$RUN/nn/$(ckpt_at)"
if [ ! -f "$EXTRACT_CKPT" ]; then say "TIMED OUT waiting for the checkpoint"; exit 1; fi
say "training done -> $(basename "$EXTRACT_CKPT")"
# rl_games holds the file open briefly after writing it.
sleep 60

say "certifying extract v6"
EXTRACT_CKPT="$EXTRACT_CKPT" EXTRACT_VERSION=v6 \
  INTERFACE="the plain grapple pin, 25 s episode" \
  bash scripts/certify_demo_policies.sh Extract

say "certifying both chains"
GRASP_CKPT="$GRASP_CKPT" EXTRACT_CKPT="$EXTRACT_CKPT" INSERT_CKPT="$INSERT_CKPT" \
  TAG=_clock bash scripts/certify_workflow.sh remove install

say "EXTRACT CLOCK PIPELINE DONE"
