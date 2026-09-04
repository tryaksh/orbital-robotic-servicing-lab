#!/usr/bin/env bash
# Verify the first seating policy that can feel contact, on both halves.
#
# This decides a headline claim. If the chain arm beats the scripted guarded
# advance, the paper reports a learned seating phase; if it does not,
# `docs/seating_controller.md` is the written defence and the paper reports a
# scripted one. Either is publishable and the difference is which.
#
# `queue_after_slot_a.sh` was going to do this and died in the 06:25 fork
# exhaustion before the checkpoint existed. Both halves matter and a skill
# certification alone has never been allowed to move the seating phase:
#
#   half 1  the skill on three held-out seeds, on the force task it trained on
#   half 2  the same weights inside the full chain against the scripted advance,
#           which is the arm that decides
#
# Runs behind the gate re-run, beside the factorial. Two evaluation processes and
# one training is well inside what this machine handles; the overnight failure
# was twelve sleeping shells, not two busy ones.
set -u
cd /d/6axis-space-robotics || exit 1
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
RUN="grapple_insert_l0_seed70_v33force"

say () { echo "[$(date +%H:%M:%S)] $*"; }

waited=0
until ls "$ROOT/$RUN/nn/"*ep_3000_*.pth >/dev/null 2>&1; do
  sleep 120
  waited=$((waited + 120))
  if [ "$waited" -gt 21600 ]; then say "no ep_3000 checkpoint after six hours; giving up"; exit 1; fi
done
until grep -q "gate re-run done" artifacts/campaign/supervise_gate_rerun.log 2>/dev/null; do sleep 180; done

CKPT=$(ls -t "$ROOT/$RUN/nn/"*ep_3000_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
if [ -z "$CKPT" ]; then say "no usable ep_3000 checkpoint"; exit 1; fi
say "verifying $CKPT on both halves"

CKPT="$CKPT" TAG=insert_v33force \
PLAY_TASK="Isaac-ZeroG-Blade-GrapplePin-InsertForce-Play-v0" \
CHAIN_TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflowForce-v0" \
  bash scripts/verify_insert_skill.sh > artifacts/campaign/verify_insert_v33force.log 2>&1
rc=$?
say "verify exit=$rc"
tail -30 artifacts/campaign/verify_insert_v33force.log
say "force verification done"
