#!/usr/bin/env bash
# Stop the attitude penalty switching itself off where it is needed.
#
# extract v11settle fixed arrival speed -- 4.8x slower, module 3.2x straighter,
# cycle 1.6x shorter -- and certifies at 0.00% because grip attitude sits at
# 0.3538 rad and 70.6% of episodes die on the 0.350 rad limit.
#
# The attitude is on the CLOSING axis (p95 0.3575 against 0.100 transverse and
# 0.078 approach), and the anti-yaw yoke was re-tested against that and does
# nothing: closing-axis p95 0.3575 -> 0.3590 with the walls on. So the
# compliance is the pads camming open under load, not wall clearance, and no
# passive geometry addresses it.
#
# What does address it is that grip_retention_penalty saturates. With extract's
# parameters the raw cost passes the 25.0 clamp at about 0.325 rad, so from
# there upward 0.35 and 0.50 cost exactly the same and the term has no gradient.
# The policy parked at 0.3538 -- just past the knee. Raising the clamp to 60
# keeps it growing across the whole range an episode can reach (raw cost at the
# 0.350 failure limit is 29.0).
#
# Resumed from v11settle so the settling behaviour it already learned is kept
# and only the attitude gradient is added. Reward change, no dimension change.
set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN=grapple_extract_l0_seed70_v13unsat
EPOCHS=5700   # 4700 + 1000
SRC="$CKPT_ROOT/grapple_extract_l0_seed70_v11settle/nn/last_zero_g_blade_insertion_contact_ep_4700_rew_148.66235.pth"
mkdir -p artifacts/extract_unsat
say() { echo "[$(date +%H:%M:%S)] $*"; }
ckpt() { ls "$CKPT_ROOT/$RUN/nn/" 2>/dev/null | grep -E "^last_.*_ep_${EPOCHS}_rew_.*\.pth$" | sort | tail -1; }

say "training $RUN from v11settle"
"$PYTHON" scripts/train.py --headless --task Isaac-ZeroG-Blade-GrapplePin-Extract-v0 \
    --num_envs 512 --seed 70 --robustness_level 0 --max_iterations "$EPOCHS" \
    --checkpoint "$SRC" --run_name "$RUN" > artifacts/extract_unsat/train.log 2>&1
say "  exit=$?  -> $(ckpt)"
E="$CKPT_ROOT/$RUN/nn/$(ckpt)"
[ -f "$E" ] || { say "no checkpoint"; exit 1; }
sleep 45
say "certifying extract v13unsat"
EVAL_ENVS=256 EXTRACT_CKPT="$E" EXTRACT_VERSION=v13unsat \
  INTERFACE="the plain grapple pin, settling reward plus an unsaturated attitude penalty" \
  bash scripts/certify_demo_policies.sh Extract
say "removal chain"
EXTRACT_CKPT="$E" TAG=_unsat bash scripts/certify_workflow.sh remove
say "EXTRACT UNSAT DONE"
