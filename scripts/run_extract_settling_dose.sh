#!/usr/bin/env bash
# Dose-response on the settling weight, to tell a trade from a threshold.
#
# At weight -2.0 the settling penalty took terminal linear velocity from
# 0.0685 to 0.0202 m/s and grip attitude from 0.108 to 0.354 rad, putting 70.6%
# of episodes at the 0.350 rad failure limit against 1.8% before. Two quantities
# moving in opposite directions with everything else held still is a trade, but
# a single point cannot distinguish a trade from a cliff the policy fell off.
#
# A quarter of the weight, everything else identical -- same source checkpoint,
# same 1,200 epochs, same seed, same environments. If the coupling is real and
# monotonic, both quantities should land between the two known points. If
# attitude is at 0.35 rad here too, it is a threshold and the mechanism is
# different from the one docs/status.md currently records.
set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_extract_l0_seed70_v12dose
EPOCHS=4700
SRC="$CKPT_ROOT/grapple_extract_l0_seed70_v8handoff/nn/last_zero_g_blade_insertion_contact_ep_3500_rew_148.35918.pth"
mkdir -p artifacts/extract_dose
say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt() { ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${EPOCHS}_rew_.*\.pth$" | sort | tail -1; }

say "training $RUN (settling weight -0.5)"
"$PYTHON" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Extract-v0 \
    --num_envs 512 --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" \
    --checkpoint "$SRC" --run_name "$RUN" > artifacts/extract_dose/train.log 2>&1
say "  exit=$?  -> $(ckpt)"
E="$CKPT_ROOT/$RUN/nn/$(ckpt)"
[ -f "$E" ] || { say "no checkpoint"; exit 1; }
sleep 45
say "certifying extract v12dose"
EVAL_ENVS=256 EXTRACT_CKPT="$E" EXTRACT_VERSION=v12dose \
  INTERFACE="the plain grapple pin, settling weight -0.5" \
  bash scripts/certify_demo_policies.sh Extract
say "EXTRACT DOSE DONE"
