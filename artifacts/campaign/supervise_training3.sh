#!/usr/bin/env bash
# The last gap in the seed spreads: extraction, seed 72.
#
# Capture has three seeds. Insertion gets its second and third from
# `supervise_training2.sh`. Extraction has 70 and 71 and has never had a third,
# so every extraction number in the paper still rests on a spread of two.
#
# Measured over the **final stage** only, exactly as seed 71 was: the same resume
# from `v17m130` at epoch 10,600, the same 2,000 epochs, the same task. Earlier
# stages are shared and any report quoting this has to say so, because a spread
# over one stage is not a spread over the procedure.
#
# One wait, on one predecessor, in one script. This is the third supervisor
# alive; the 06:25 crash was twelve, each with a child.
set -u
cd /d/6axis-space-robotics || exit 1
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
V17="$ROOT/grapple_extract_l0_seed70_v17m130/nn/last_zero_g_blade_insertion_contact_ep_10600_rew_168.46431.pth"

say () { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -f "$V17" ]; then say "no v17m130 checkpoint to resume from; refusing to start"; exit 1; fi

until grep -q "training supervisor 2 done" artifacts/campaign/supervise_training2.log 2>/dev/null; do sleep 600; done
say "training slot free; extraction seed 72, the third of the spread"

"C:/isaac-sim/python.bat" scripts/train.py --headless \
    --task Isaac-ZeroG-Blade-GrapplePin-Extract-v0 \
    --num_envs 512 --seed 72 --robustness_level 0 \
    --max_iterations 12600 --checkpoint "$V17" \
    --run_name grapple_extract_l0_seed72_v18stage \
    > artifacts/campaign/train_extract_seed72_v18stage.log 2>&1
rc=$?
say "extract seed 72 exit=$rc"
say "training supervisor 3 done"
