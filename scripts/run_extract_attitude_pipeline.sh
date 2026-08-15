#!/usr/bin/env bash
# Certify extract v7 -- the attitude-weighted retention penalty -- and re-close
# both chains against it. Waits for the fine-tune, so it can be armed early.
#
# The experiment, one failure category: extract v6 certifies at 10.09% and 7,971
# of its 8,093 failures end at the 0.350 rad grip-attitude limit while grip
# position still holds at 12.5 mm. Under the shared retention defaults that
# attitude costs about 0.16 per step at the success limit against a progress
# term weighted 12, so the policy was paid to trade attitude for travel. The
# extract task now charges it about 3.6 per step instead. Nothing else moved,
# and the shared default is untouched because insert v6 was certified under it.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_extract_l0_seed70_v7attitude
FINAL_EPOCH=2400
GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v3/nn/last_zero_g_blade_insertion_contact_ep_700_rew__36.020187_.pth"
INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"
mkdir -p artifacts evidence

say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt_at() { ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${FINAL_EPOCH}_rew_.*\.pth$" | sort | tail -1; }

say "waiting for $RUN epoch $FINAL_EPOCH"
for _ in $(seq 1 240); do
  [ -n "$(ckpt_at)" ] && break
  sleep 30
done
EXTRACT_CKPT="$CKPT_ROOT/$RUN/nn/$(ckpt_at)"
if [ ! -f "$EXTRACT_CKPT" ]; then say "TIMED OUT waiting for the checkpoint"; exit 1; fi
say "training done -> $(basename "$EXTRACT_CKPT")"
sleep 60

say "certifying extract v7"
EXTRACT_CKPT="$EXTRACT_CKPT" EXTRACT_VERSION=v7 \
  INTERFACE="the plain grapple pin, 25 s episode, attitude-weighted retention" \
  bash scripts/certify_demo_policies.sh Extract

say "certifying both chains"
GRASP_CKPT="$GRASP_CKPT" EXTRACT_CKPT="$EXTRACT_CKPT" INSERT_CKPT="$INSERT_CKPT" \
  TAG=_v7 bash scripts/certify_workflow.sh remove install

say "checking every quoted chain loads a certified policy"
"$PYTHON" scripts/check_evidence_currency.py artifacts/workflow_cert/*_v7_seed*_report.json || true

say "EXTRACT ATTITUDE PIPELINE DONE"
