#!/usr/bin/env bash
# Phases 5, 6 and 7 on the moved workcell, in the order their gates allow.
#
# One stage per invocation, because each has a gate and the next must not start
# before it passes. Each is a separate call:
#
#   scripts/certify_relocation_workcell.sh promote   # move every script's defaults
#   scripts/certify_relocation_workcell.sh skills    # Phase 5: extract
#   scripts/certify_relocation_workcell.sh capture   # Phase 5: capture, already done as v6w65
#   scripts/certify_relocation_workcell.sh insert2   # Phase 5: insert, both bays
#   scripts/certify_relocation_workcell.sh chains    # Phase 6: removal + installation
#   scripts/certify_relocation_workcell.sh trace     # Phase 7: read the relocation
#   scripts/certify_relocation_workcell.sh latchab   # Phase 2: latch off vs on release
#   scripts/certify_relocation_workcell.sh relocate  # Phase 7: certify the relocation
#
# The run names are the ones scripts/retrain_workcell_skills.sh produces. w65 is
# the workcell: GRAPPLE_ROBOT_ROOT_POS x = -0.65.

set -u

PYTHON="C:/isaac-sim/python.bat"
GRASP_RUN="${GRASP_RUN:-grapple_grasp_l0_seed70_v6w65}"
# v15w65 and v11w65 were the first fine-tune pass and did not re-converge: zero
# successes in training and 0.10% on extraction's intermediate certification.
# v16w65 and v12w65 continue them to 9,700 and 9,600 epochs. Capture needed no
# second pass -- it certified at 94.46% after 900.
EXTRACT_RUN="${EXTRACT_RUN:-grapple_extract_l0_seed70_v16w65}"
INSERT_RUN="${INSERT_RUN:-grapple_insert_l0_seed70_v12w65}"
OUT="${OUT:-artifacts/workcell_cert}"
mkdir -p "$OUT" evidence

stage="${1:-}"

case "$stage" in
  promote)
    # Move every script's default checkpoint set at once. Four scripts carry
    # these defaults and all four have drifted behind the promoted set before.
    "$PYTHON" scripts/promote_checkpoints.py \
        --grasp "$GRASP_RUN" --extract "$EXTRACT_RUN" --insert "$INSERT_RUN"
    ;;

  skills)
    # Extraction only. Capture needed no second training pass -- it recovered in
    # 900 epochs and is already certified at 94.46% as v6w65 -- and re-running an
    # unchanged checkpoint against an unchanged task costs 36 minutes to
    # reproduce a file that already exists. Pass `capture` to redo it.
    #
    # Insert is certified by `insert2` instead, because the promoted insert is
    # the two-bay policy and its gate is the worse bay, not a single-slot pool.
    EXTRACT_VERSION=v16w65 \
      INTERFACE="the plain grapple pin, on the moved workcell at x = -0.65" \
      bash scripts/certify_demo_policies.sh Extract
    ;;

  capture)
    GRASP_VERSION=v6w65 \
      INTERFACE="the plain grapple pin, on the moved workcell at x = -0.65" \
      bash scripts/certify_demo_policies.sh Grasp
    ;;

  insert2)
    # Tagged, because run_relocation.sh's untagged name is the file main's
    # 98.34% lives in and that is the "before" half of the comparison.
    RUN="$INSERT_RUN" TAG="_w65" bash scripts/run_relocation.sh certify2
    ;;

  chains)
    # Both existing chains, on the changed geometry. A relocation bought by
    # breaking removal is not a demonstration.
    TAG="_w65" bash scripts/certify_workflow.sh remove install
    ;;

  trace)
    RUN="$INSERT_RUN" TAG="_w65" EPISODES="${EPISODES:-64}" bash scripts/run_relocation.sh trace
    ;;

  latchab)
    # Phase 2, measured where it matters instead of against a synthetic torque.
    # Two traces, identical but for the latch, under the real transit load.
    RUN="$INSERT_RUN"
    INSERT_CKPT=$(ls "logs/rl_games"/*/"$RUN"/nn/*_ep_*.pth 2>/dev/null |
      sed -n 's/.*_ep_\([0-9]\+\)_.*/\1 &/p' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
    if [ -z "$INSERT_CKPT" ]; then echo "NO INSERT CHECKPOINT for $RUN"; exit 1; fi
    CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
    GRASP_CKPT=$(ls "logs/rl_games"/*/"$GRASP_RUN"/nn/*_ep_*.pth |
      sed -n 's/.*_ep_\([0-9]\+\)_.*/\1 &/p' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
    EXTRACT_CKPT=$(ls "logs/rl_games"/*/"$EXTRACT_RUN"/nn/*_ep_*.pth |
      sed -n 's/.*_ep_\([0-9]\+\)_.*/\1 &/p' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
    for arm in off on; do
      extra=""
      [ "$arm" = "on" ] && extra="--latch_on_release"
      out="$OUT/latch_${arm}"
      "$PYTHON" scripts/run_workflow_demo.py --headless \
          --workflow relocate --curriculum_stage 0 \
          --task Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0 \
          --grasp_checkpoint "$GRASP_CKPT" --extract_checkpoint "$EXTRACT_CKPT" \
          --insert_checkpoint "$INSERT_CKPT" \
          --num_envs 64 --episodes "${EPISODES:-192}" --seed 4070 \
          --report "${out}_report.json" --episode_metrics "${out}.npz" $extra \
          > "${out}.log" 2>&1
      echo "[$(date +%H:%M:%S)] latch=$arm exit=$?"
      grep -E "\[CHAIN\]" "${out}.log" | tail -3
    done
    ;;

  relocate)
    # NOT tagged. evidence/workflow_relocate_certification.json is the file this
    # whole branch exists to produce and main has no version of it to protect.
    RUN="$INSERT_RUN" bash scripts/run_relocation.sh relocate
    ;;

  *)
    echo "usage: scripts/certify_relocation_workcell.sh {promote|skills|capture|insert2|chains|trace|latchab|relocate}"
    exit 2
    ;;
esac
