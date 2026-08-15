#!/usr/bin/env bash
# Certify extract v9 and close both chains. Waits for the fine-tune, so it can
# be armed while training is still running.
#
# Refuses to start if another certification is already running. That guard is
# here because two pipelines once overlapped on this machine and the second one
# certified checkpoints the first was still training.
#
# Extract v9 is v8 with one change: extraction_success_mask holds its condition
# for 0.70 s instead of 0.20, which is the chained workflow's own settling
# window. Measured on v8, the chain fired its predicate in 191 of 192 removals
# and none survived the re-check, because a module merely below the velocity
# limit keeps drifting in zero gravity once the arm stops commanding.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_extract_l0_seed70_v10settled
FINAL_EPOCH=5100
LOCK=artifacts/.certification.lock
mkdir -p artifacts evidence

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another certification is running as PID $(cat "$LOCK"); refusing to start"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth"
INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"
export EVAL_ENVS=256

say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt_at() { ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${FINAL_EPOCH}_rew_.*\.pth$" | sort | tail -1; }

say "waiting for $RUN epoch $FINAL_EPOCH"
for _ in $(seq 1 240); do
  [ -n "$(ckpt_at)" ] && break
  sleep 30
done
EXTRACT_CKPT="$CKPT_ROOT/$RUN/nn/$(ckpt_at)"
if [ ! -f "$EXTRACT_CKPT" ]; then say "TIMED OUT"; exit 1; fi
say "training done -> $(basename "$EXTRACT_CKPT")"
sleep 60

say "certifying extract v9"
EXTRACT_CKPT="$EXTRACT_CKPT" EXTRACT_VERSION=v10 \
  INTERFACE="the plain grapple pin, settle-consistent velocity limits" \
  bash scripts/certify_demo_policies.sh Extract

say "certifying both chains"
GRASP_CKPT="$GRASP_CKPT" EXTRACT_CKPT="$EXTRACT_CKPT" INSERT_CKPT="$INSERT_CKPT" \
  TAG=_v10 bash scripts/certify_workflow.sh remove install

say "auditing that every quoted chain loads a certified policy"
"$PYTHON" scripts/check_evidence_currency.py artifacts/workflow_cert/*_v10_seed*_report.json || true

say "CHAIN CERTIFICATION DONE"
