#!/usr/bin/env bash
# The relocation roadmap, run end to end: slot 2, the skills it needs, the chain.
#
# One script per stage rather than one long one, because each stage has a gate
# and the next must not start before it passes. Each is a separate invocation:
#
#   scripts/run_relocation.sh calibrate   # item 2: is slot 2 reachable
#   scripts/run_relocation.sh smoke       # item 2: does the two-slot scene build
#   scripts/run_relocation.sh insert2     # item 3: insert, both slots
#   scripts/run_relocation.sh certify2    # item 3 gate: >= 95% on both slots
#   scripts/run_relocation.sh relocate    # item 5: certify the whole relocation
#
# Every checkpoint variable can be overridden; the defaults name the promoted set
# in CLAUDE.md and must be moved with it.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/relocation"
mkdir -p "$OUT" evidence

GRASP_CKPT="${GRASP_CKPT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v5/nn/last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth}"
EXTRACT_CKPT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v13unsat/nn/last_zero_g_blade_insertion_contact_ep_5700_rew__148.17932_.pth}"
INSERT_CKPT="${INSERT_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v6/nn/last_zero_g_blade_insertion_contact_ep_3200_rew__24.907995_.pth}"

stage="${1:-}"

case "$stage" in
  calibrate)
    # Rule 7: before believing a solver, converge it. A pose was once called
    # unreachable on a 400-step residual and converges to 0.0060 mm at 3,000.
    # The Capture task, not Extract: Extract holds the arm still for its first
    # second and every offset including zero reports unconverged.
    echo "[$(date +%H:%M:%S)] CALIBRATE the second slot at y = -0.22"
    "$PYTHON" scripts/calibrate_grasp_pose.py --headless \
        --task Isaac-ZeroG-Blade-GrapplePin-Capture-v0 \
        --steps 3000 --pin_blade --finger_joint 0.02 \
        --target_offset 0 -0.22 0 \
        --report "$OUT/slot_two_pose.json" \
        2>&1 | tee "$OUT/calibrate_slot_two.log" | grep -E "CALIB|residual|converged"
    ;;

  smoke)
    echo "[$(date +%H:%M:%S)] SMOKE the two-slot insert task"
    "$PYTHON" scripts/train.py --headless \
        --task Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0 \
        --num_envs 32 --robustness_level 0 --smoke \
        > "$OUT/smoke_two_slot.log" 2>&1
    status=$?
    echo "[$(date +%H:%M:%S)] smoke exit=$status"
    grep -E "^\[INFO\]|Error|Exception" "$OUT/smoke_two_slot.log" | tail -12
    ;;

  insert2)
    # Fine-tune the promoted insert across both bays at once. Both slots stay in
    # the mixture rather than training slot 2 alone, because the gate is
    # "insert >= 95% on both" and a policy that forgets the first bay to learn
    # the second has not done the job.
    EPOCHS="${EPOCHS:-1200}"
    RUN="${RUN:-grapple_insert_l0_seed70_v10twoslot}"
    RESUME_EPOCH=$(echo "$INSERT_CKPT" | sed -n 's/.*_ep_\([0-9]\+\)_.*/\1/p')
    TARGET=$((RESUME_EPOCH + EPOCHS))
    echo "[$(date +%H:%M:%S)] TRAIN insert on both slots: $RESUME_EPOCH + $EPOCHS -> $TARGET"
    "$PYTHON" scripts/train.py --headless \
        --task Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0 \
        --num_envs "${NUM_ENVS:-512}" --seed 70 --robustness_level 0 \
        --max_iterations "$TARGET" --checkpoint "$INSERT_CKPT" --run_name "$RUN" \
        > "$OUT/train_two_slot.log" 2>&1
    echo "[$(date +%H:%M:%S)] train exit=$?"
    ;;

  certify2)
    # Stage 0 is the certified bay, stage 1 the new one. Both are reported, and
    # the gate is the worst of them, not the pool: a policy that scores 99% on
    # one bay and 90% on the other has not passed.
    RUN="${RUN:-grapple_insert_l0_seed70_v10twoslot}"
    checkpoint="${CERTIFY_CKPT:-$(ls "logs/rl_games"/*/"$RUN"/nn/*_ep_*.pth 2>/dev/null |
      sed -n 's/.*_ep_\([0-9]\+\)_.*/\1 &/p' | sort -k1,1n | tail -1 | cut -d' ' -f2-)}"
    if [ -z "$checkpoint" ]; then echo "NO CHECKPOINT for $RUN"; exit 1; fi
    echo "[$(date +%H:%M:%S)] CERTIFY insert on both slots: $checkpoint"
    rows=()
    for seed in 1070 2070 3070; do
      for slot in 0 1; do
        out="$OUT/insert_twoslot_s${slot}_seed${seed}"
        "$PYTHON" scripts/play.py --headless \
            --task Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-Play-v0 \
            --checkpoint "$checkpoint" --num_envs "${EVAL_ENVS:-128}" \
            --episodes "${EVAL_EPISODES:-500}" --curriculum_stage "$slot" --seed "$seed" \
            --episode_metrics "${out}.npz" > "${out}.log" 2>&1
        echo "[$(date +%H:%M:%S)]   slot=$slot seed=$seed exit=$?"
        rows+=("${out}.npz")
      done
    done
    "$PYTHON" scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/grapple_insert_two_slot_certification.json" \
        --title "Head-on grapple-pin insert skill, both rack bays" \
        --scope \
          "Simulation only. No result here was produced on real hardware." \
          "Curriculum stage 0 is the certified bay at y = 0; stage 1 is the second bay at y = -0.22 m." \
          "The gate is the worst bay, not the pool: one policy has to seat a module in either slot." \
          "The module is held by physical pad-against-pin contact throughout, with no fixed joint." \
        > "$OUT/aggregate_two_slot.log" 2>&1
    echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/grapple_insert_two_slot_certification.json"
    tail -6 "$OUT/aggregate_two_slot.log"
    ;;

  relocate)
    echo "[$(date +%H:%M:%S)] CERTIFY the relocation chain"
    rows=()
    for seed in 4070 5070 6070; do
      out="$OUT/relocate_seed${seed}"
      "$PYTHON" scripts/run_workflow_demo.py --headless \
          --workflow relocate --curriculum_stage 0 \
          --task Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0 \
          --grasp_checkpoint "$GRASP_CKPT" --extract_checkpoint "$EXTRACT_CKPT" \
          --insert_checkpoint "$INSERT_CKPT" \
          --num_envs "${ENVS:-64}" --episodes "${EPISODES:-192}" --seed "$seed" \
          --report "${out}_report.json" --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      echo "[$(date +%H:%M:%S)]   relocate seed=$seed exit=$?"
      rows+=("${out}.npz")
    done
    "$PYTHON" scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/workflow_relocate_certification.json" \
        --title "Chained servicing workflow: relocation, bay 1 to bay 2" \
        --scope \
          "Simulation only. No result here was produced on real hardware." \
          "One continuous episode: capture, extract, lateral transit, insert into the second bay." \
          "The module is held by physical pad-against-pin contact throughout. No fixed joint, no software fixture." \
          "Capture, extraction and insertion are trained policies; the seating pause and the waypoint-followed transit are scripted and labelled." \
          "Success is the workflow's own condition re-checked after a 0.70 s settling window." \
        > "$OUT/aggregate_relocate.log" 2>&1
    echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/workflow_relocate_certification.json"
    tail -6 "$OUT/aggregate_relocate.log"
    ;;

  *)
    echo "usage: scripts/run_relocation.sh {calibrate|smoke|insert2|certify2|relocate}"
    exit 2
    ;;
esac
