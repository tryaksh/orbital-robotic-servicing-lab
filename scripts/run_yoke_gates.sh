#!/usr/bin/env bash
# Measure the anti-yaw yoke before any policy is trained against it.
#
# Three runs, one question each, on the same capture scene and the same
# capture/hold closure split the 69 N pull gate was measured with:
#
#   1. yaw gate, plain pin  -- how far does the payload rotate under a moment
#      about the closing axis, which is the axis a pair of flat pads cannot
#      oppose because their contact normals lie along it;
#   2. yaw gate, yoked pin  -- the same grid, so the two are comparable;
#   3. axial pull gate, yoked pin -- does the interface still clear the 66.4 N
#      the promoted insertion policy's own contact reaction demands. Adding
#      geometry that fixes rotation is worthless if it costs the hold.
#
# This is deliberately before retraining anything. The order of work has put the
# physics gate before PPO since the first grapple-pin session, so that a policy
# is never trained against an interface that cannot carry the load.

set -u

PYTHON="C:/isaac-sim/python.bat"
TASK="Isaac-ZeroG-Blade-GrapplePin-Capture-v0"
# The measured capture window, biased low, plus the firming delta that produced
# 69 N. See evidence/grapple_pin_capture_plateau.json.
CLOSURES="0.44 0.48 0.52"
HOLD_DELTA=0.20
TORQUE_LEVELS="${TORQUE_LEVELS:-41}"
MAX_TORQUE="${MAX_TORQUE:-12.0}"
FORCE_LEVELS="${FORCE_LEVELS:-121}"
MAX_FORCE="${MAX_FORCE:-120.0}"
mkdir -p artifacts evidence

run() {
  local label="$1"; shift
  echo "[$(date +%H:%M:%S)] $label"
  "$PYTHON" scripts/grasp_diagnostics.py --headless \
      --task "$TASK" \
      --closures $CLOSURES \
      --hold_closure_delta "$HOLD_DELTA" \
      "$@" > "artifacts/${label}.log" 2>&1
  local status=$?
  echo "[$(date +%H:%M:%S)]   exit=$status"
  if [ "$status" -ne 0 ]; then
    tail -20 "artifacts/${label}.log"
  fi
}

# --plain_pin is explicit because the task now configures the yoke on. Without
# it the "plain" baseline would silently measure the yoked pin.
run yaw_gate_plain_pin \
    --load_axis yaw --plain_pin --force_levels "$TORQUE_LEVELS" --max_torque_nm "$MAX_TORQUE" \
    --report evidence/grapple_pin_yaw_gate_plain.json

run yaw_gate_yoked_pin \
    --load_axis yaw --anti_yaw_yoke --force_levels "$TORQUE_LEVELS" --max_torque_nm "$MAX_TORQUE" \
    --report evidence/grapple_pin_yaw_gate_yoked.json

run axial_gate_yoked_pin \
    --load_axis axial --anti_yaw_yoke --force_levels "$FORCE_LEVELS" --max_force_n "$MAX_FORCE" \
    --report evidence/grapple_pin_axial_pull_gate_yoked.json

echo "[$(date +%H:%M:%S)] YOKE GATES DONE"
