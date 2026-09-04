#!/usr/bin/env bash
# Settle the environment-count risk with one run, last, when nothing else wants
# the GPU.
#
# The same nominal point scores 0/32 in `artifacts/envcount_32x6000` and 35/64 in
# `artifacts/robustness64_corrected`. It is not the step budget and not rack
# retention. Either the batch size is a hidden parameter of every rate in the
# paper, or those probes were configured differently in a way no report of that
# era captured -- and reports have carried a `geometry_arm` block since
# 2026-09-03, so a fresh run at 32 environments can be diffed against the 64-
# environment one directly.
#
# This is deliberately the last thing in the queue. The session brief named the
# anomaly as a thing not to chase, and it is not being chased: it is one run of
# one point, behind every result that matters, and it either removes a threat to
# validity or turns it into a number the paper has to report.
set -u
cd /d/6axis-space-robotics || exit 1
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
export GRASP_CKPT="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
export EXTRACT_CKPT="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
export INSERT_CKPT="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"

until grep -q "hand-over traces done" artifacts/campaign/handover_trace.log 2>/dev/null; do sleep 300; done
echo "[$(date +%H:%M:%S)] everything else is finished; settling the environment count"

# Identical to the 64-environment nominal in every flag the sweep sets. Only
# ENVS differs, which is the whole question.
POINTS="nominal" EPISODES=64 ENVS=32 STEPS=6000 \
  OUT=artifacts/envcount_settle_32 SEED=4070 \
  bash scripts/sweep_chain_robustness.sh
rc=$?
echo "[$(date +%H:%M:%S)] 32-environment nominal exit=$rc"

echo "[$(date +%H:%M:%S)] geometry_arm blocks, 32 against 64:"
./.venv/Scripts/python.exe - <<'PY'
import json, pathlib
for label, path in (
    ("32 env, fresh", "artifacts/envcount_settle_32/nominal_report.json"),
    ("64 env, published", "artifacts/robustness64_corrected/nominal_report.json"),
):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"  {label}: missing")
        continue
    report = json.loads(p.read_text(encoding="utf-8"))
    overall = report.get("overall") or {}
    print(f"  {label}: {overall.get('successes')}/{overall.get('episodes')}")
    print(f"    geometry_arm: {json.dumps(report.get('geometry_arm'), sort_keys=True)}")
PY

echo "[$(date +%H:%M:%S)] environment count settled"
