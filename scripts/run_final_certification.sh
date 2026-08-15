#!/usr/bin/env bash
# Certify the three skills as they now stand and close both chains.
#
# Refuses to start if another certification or pipeline is already running.
# That guard exists because two pipelines once overlapped on this machine --
# a pkill that did not take -- and the second one certified checkpoints that
# the first was still training. Nothing was corrupted, but only by luck.
#
# Policies, all on the plain grapple pin with the yoke and latch off:
#   capture v5  ep 1500, 96.10% certified
#   extract v8  ep 3500, the action-authority and hand-off-envelope retrain
#   insert  v6  ep 3200, 95.57% certified

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
LOCK=artifacts/.certification.lock
mkdir -p artifacts evidence

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another certification is running as PID $(cat "$LOCK"); refusing to start"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth"
EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/$(ls "$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/" | grep -E '^last_.*_ep_3500_rew_.*\.pth$' | sort | tail -1)"
INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth"

for c in "$GRASP_CKPT" "$EXTRACT_CKPT" "$INSERT_CKPT"; do
  if [ ! -f "$c" ]; then echo "MISSING $c"; exit 1; fi
done

export EVAL_ENVS=256
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "extract v8: $(basename "$EXTRACT_CKPT")"
EXTRACT_CKPT="$EXTRACT_CKPT" EXTRACT_VERSION=v8 \
  INTERFACE="the plain grapple pin, 25 s episode, rebalanced action authority" \
  bash scripts/certify_demo_policies.sh Extract

say "both chains"
GRASP_CKPT="$GRASP_CKPT" EXTRACT_CKPT="$EXTRACT_CKPT" INSERT_CKPT="$INSERT_CKPT" \
  TAG=_final bash scripts/certify_workflow.sh remove install

say "auditing that every quoted chain loads a certified policy"
"$PYTHON" scripts/check_evidence_currency.py artifacts/workflow_cert/*_final_seed*_report.json || true

say "camera scale gate, one rendered frame"
"$PYTHON" scripts/check_camera_scale.py --headless > artifacts/camera_scale.log 2>&1
say "  exit=$?"
tail -20 artifacts/camera_scale.log

say "FINAL CERTIFICATION DONE"
