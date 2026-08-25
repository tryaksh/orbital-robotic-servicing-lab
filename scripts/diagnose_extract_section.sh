#!/usr/bin/env bash
# Why did 900 epochs of fine-tuning move extract 1.4 points?
#
# The one thing that changed under the extract task between the 94.89% it was
# certified at and the 74.27% it scores now is the module's cross-section:
# 450 x 160 x 35 mm became 450 x 130 x 20 mm. Length, mass and the grapple pin
# are all untouched, so the section is the whole difference.
#
# A cross-section is not cosmetic here. The module is inside the rack's own
# channel for the entire pull, and what the channel holds is set by the gap
# around the module. This is a 2x2: two policies, two sections, one seed, one
# stage, everything else fixed. If the section is the cause, both policies move
# together with it and neither moves much with the retraining.
#
# ``--grip_axis_metrics`` also splits the grip error into the gripper's own
# axes, which is what says *how* a grip is lost rather than that it was.

set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/diag"
mkdir -p "$OUT"

V17="$CKPT_ROOT/grapple_extract_l0_seed70_v17m130/nn/last_zero_g_blade_insertion_contact_ep_10600_rew_168.46431.pth"
V16="$CKPT_ROOT/grapple_extract_l0_seed70_v16w65/nn/last_zero_g_blade_insertion_contact_ep_9700_rew__176.34572_.pth"

TASK="Isaac-ZeroG-Blade-GrapplePin-Extract-Play-v0"
ENVS="${ENVS:-128}"
EPISODES="${EPISODES:-512}"
SEED="${SEED:-1070}"

run() {
  tag="$1"; ckpt="$2"; stage="$3"; shift 3
  echo "[$(date +%H:%M:%S)] $tag"
  "$PYTHON" scripts/play.py --headless --task "$TASK" --checkpoint "$ckpt" \
      --num_envs "$ENVS" --episodes "$EPISODES" --curriculum_stage "$stage" \
      --seed "$SEED" --grip_axis_metrics \
      --episode_metrics "$OUT/${tag}.npz" --report "$OUT/${tag}.json" \
      "$@" > "$OUT/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)]   exit=$? $(grep -oE '"success_rate": [0-9.]+' "$OUT/${tag}.json" | head -1)"
}

run x_v17_s0_new "$V17" 0
run x_v17_s0_old "$V17" 0 --blade_cross_section 0.160 0.035
run x_v16_s0_new "$V16" 0
run x_v16_s0_old "$V16" 0 --blade_cross_section 0.160 0.035
run x_v17_s2_new "$V17" 2
echo "[$(date +%H:%M:%S)] DONE"
