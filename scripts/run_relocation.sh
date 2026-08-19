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
#   scripts/run_relocation.sh trace       # item 4 gate: the grip across the transit
#   scripts/run_relocation.sh relocate    # item 5: certify the whole relocation
#
# Every checkpoint variable can be overridden; the defaults name the promoted set
# in CLAUDE.md and must be moved with it.

set -u

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/relocation"
# Suffix for every report this script writes. A re-run on changed geometry must
# not overwrite the report that describes the old geometry: that report is the
# "before" half of every comparison, and once it is gone the comparison cannot be
# made again without re-running a workcell that no longer exists in the tree.
TAG="${TAG:-}"
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
        --output "evidence/grapple_insert_two_slot${TAG}_certification.json" \
        --title "Head-on grapple-pin insert skill, both rack bays" \
        --scope \
          "Simulation only. No result here was produced on real hardware." \
          "Curriculum stage 0 is the certified bay at y = 0; stage 1 is the second bay at y = -0.22 m." \
          "The gate is the worst bay, not the pool: one policy has to seat a module in either slot." \
          "The module is held by physical pad-against-pin contact throughout, with no fixed joint." \
        > "$OUT/aggregate_two_slot.log" 2>&1
    echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/grapple_insert_two_slot${TAG}_certification.json"
    tail -6 "$OUT/aggregate_two_slot.log"
    ;;

  trace)
    # Item 4's gate: is the module still held after being flown to the next bay?
    #
    # Read off the hand-offs rather than off the success rate, because a transit
    # that degrades the grip and an insertion that fails are different faults
    # with different fixes, and the chain's own number cannot tell them apart.
    # Both ends are reported: extract -> transit is the grip the transit was
    # handed, transit -> insert is what it gives back, and the gate is on the
    # second with the first as its control.
    RUN="${RUN:-grapple_insert_l0_seed70_v10twoslot}"
    INSERT_CKPT="${RELOCATE_INSERT_CKPT:-$(ls "logs/rl_games"/*/"$RUN"/nn/*_ep_*.pth 2>/dev/null |
      sed -n 's/.*_ep_\([0-9]\+\)_.*/\1 &/p' | sort -k1,1n | tail -1 | cut -d' ' -f2-)}"
    if [ -z "$INSERT_CKPT" ]; then echo "NO TWO-SLOT INSERT CHECKPOINT for $RUN"; exit 1; fi
    echo "[$(date +%H:%M:%S)] TRACE the relocation hand-offs"
    "$PYTHON" scripts/run_workflow_demo.py --headless \
        --workflow relocate --curriculum_stage 0 \
        --task Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0 \
        --grasp_checkpoint "$GRASP_CKPT" --extract_checkpoint "$EXTRACT_CKPT" \
        --insert_checkpoint "$INSERT_CKPT" \
        --num_envs "${ENVS:-64}" --episodes "${EPISODES:-192}" --seed "${SEED:-4070}" \
        --report "$OUT/relocate_trace_report.json" \
        --handoff_trace "$OUT/relocate_handoff.npz" \
        > "$OUT/relocate_trace.log" 2>&1
    echo "[$(date +%H:%M:%S)] trace exit=$?"
    for phase in transit insert; do
      "$PYTHON" scripts/analyse_handoff.py "$OUT/relocate_handoff.npz" --to_phase "$phase" \
          --json "evidence/relocate_handoff_to_${phase}${TAG}.json" > "$OUT/handoff_${phase}.log" 2>&1
      echo "[$(date +%H:%M:%S)] -> evidence/relocate_handoff_to_${phase}${TAG}.json"
      grep -A 10 '"grip_error_m"' "$OUT/handoff_${phase}.log" | head -11
    done
    ;;

  relocate)
    # The relocation seats the module in the SECOND bay, so it must be driven by
    # the two-bay insert policy item 3 produced -- not by the promoted
    # single-slot v6, which has never seen that bay and whose goal command was a
    # constant when it was trained. Defaulting INSERT_CKPT to v6 here would run
    # the chain with a policy certified for the wrong slot and report the result
    # as a relocation, so the two-slot run is the default and v6 has to be asked
    # for by name.
    RUN="${RUN:-grapple_insert_l0_seed70_v10twoslot}"
    INSERT_CKPT="${RELOCATE_INSERT_CKPT:-$(ls "logs/rl_games"/*/"$RUN"/nn/*_ep_*.pth 2>/dev/null |
      sed -n 's/.*_ep_\([0-9]\+\)_.*/\1 &/p' | sort -k1,1n | tail -1 | cut -d' ' -f2-)}"
    if [ -z "$INSERT_CKPT" ]; then
      echo "NO TWO-SLOT INSERT CHECKPOINT for $RUN -- run 'insert2' and pass its gate first"
      exit 1
    fi
    echo "[$(date +%H:%M:%S)] CERTIFY the relocation chain"
    echo "[$(date +%H:%M:%S)]   insert policy: $INSERT_CKPT"
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
        --output "evidence/workflow_relocate${TAG}_certification.json" \
        --title "Chained servicing workflow: relocation, bay 1 to bay 2" \
        --scope \
          "Simulation only. No result here was produced on real hardware." \
          "One continuous episode: capture, extract, lateral transit, insert into the second bay." \
          "The module is held by physical pad-against-pin contact throughout. No fixed joint, no software fixture." \
          "Capture, extraction and insertion are trained policies; the seating pause and the waypoint-followed transit are scripted and labelled." \
          "Success is the workflow's own condition re-checked after a 0.70 s settling window." \
        > "$OUT/aggregate_relocate.log" 2>&1
    echo "[$(date +%H:%M:%S)] aggregate exit=$? -> evidence/workflow_relocate${TAG}_certification.json"
    tail -6 "$OUT/aggregate_relocate.log"
    ;;

  *)
    echo "usage: scripts/run_relocation.sh {calibrate|smoke|insert2|certify2|trace|relocate}"
    exit 2
    ;;
esac
