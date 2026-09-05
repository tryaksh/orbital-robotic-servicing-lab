#!/usr/bin/env bash
# The two measurements the handoff method needs, and one it gets for free.
#
# **The thesis this batch is evidence for.** A learned skill's own success rate
# does not predict whether it works inside the full task -- our seating skill
# certifies at 99.2% alone and delivers 25% in the chain -- so the handoff should
# not be triggered by the upstream skill finishing. It should be triggered by the
# downstream skill's physical precondition being measurably satisfied. The
# guarded advance in this repository already does that, and nothing has ever
# measured what it buys, because until today there was no way to switch it off.
#
#   STAGE 1  the gate, ablated. Same camera-driven chain, same three held-out
#            seeds, same checkpoints; one arm advances only while the deployed
#            estimate says the module is inside the entry envelope, the other
#            advances regardless. The detection interlock is on in both, so this
#            isolates the geometric gate and not the sensing.
#
#            Both arms are traced, which also produces the camera-driven
#            delivered-attitude measurement NEXT_WORK T21 asks for. The existing
#            8.19 mrad figure is from the exact-state chain, and the number that
#            belongs in a design rule is the one the deployed sensing produces.
#
#   STAGE 2  gravity, swept. Every number in this repository was measured at
#            zero, and the claim that orbit is different has been argued in prose
#            and never measured. Earth, Mars, the Moon and orbit, on the module
#            alone, with the rack's retention off so the module is genuinely free
#            after release -- because the claim is about what an unheld part does.
#
# Runs beside one training and nothing else. `aggregate_evaluation.py` exits 2
# on a failed gate and 1 on a refused cohort; these arms are not expected to pass
# a 95% gate, so the two are distinguished rather than treated alike -- see
# AGENTS.md.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED="$ROOT/grapple_extract_l0_seed70_v19noised/nn/last_zero_g_blade_insertion_contact_ep_14600_rew_166.19054.pth"
NCAP=$(ls -t "$ROOT/grapple_grasp_l0_seed70_v9noised/nn/"*ep_5100_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
VISION="Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
STATE="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

say () { echo "[$(date +%H:%M:%S)] $*"; }

free_mb () {
  powershell -NoProfile -Command '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)' 2>/dev/null || echo 9999
}
wait_for_memory () {
  waited=0
  while [ "$(free_mb)" -lt 4000 ]; do
    say "  $(free_mb) MB free; holding"
    sleep 300
    waited=$((waited + 300))
    if [ "$waited" -gt 7200 ]; then say "  two hours under 4 GB free; skipping this stage"; return 1; fi
  done
  return 0
}

report_rc () {
  case "$1" in
    0) say "  $2: gate passed" ;;
    2) say "  $2: gate not met, which for this arm is the measurement" ;;
    *) say "  $2: COHORT REFUSED (rc=$1)" ;;
  esac
}

if [ -z "$NCAP" ]; then say "no ep_5100 noised capture; refusing to start"; exit 1; fi
say "capture $(basename "$NCAP")"

# ---------------------------------------------------------------------------
say "STAGE 1/2  the handoff gate, on and ablated, camera-driven, traced"
# ---------------------------------------------------------------------------
OUT=artifacts/campaign/gate_ablation
mkdir -p "$OUT"
if wait_for_memory; then
  for arm in lead_in none; do
    rows=()
    for seed in 4070 5070 6070; do
      out="$OUT/guard_${arm}_seed${seed}"
      "$PY" scripts/run_workflow_demo.py --headless \
          --workflow relocate --curriculum_stage 0 --task "$VISION" \
          --grasp_checkpoint "$NCAP" --extract_checkpoint "$NOISED" --insert_checkpoint "$I" \
          --num_envs 16 --seed "$seed" --steps 1900 \
          --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
          --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
          --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
          --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
          --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
          --module_velocity_source kinematics --fiducial_guard_bounds "$arm" \
          --handoff_trace "${out}_trace.npz" \
          --report "${out}_report.json" --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      rc=$?
      say "  guard=$arm seed $seed exit=$rc"
      [ -f "${out}.npz" ] && rows+=("${out}.npz")
    done
    if [ "${#rows[@]}" -eq 3 ]; then
      ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
          --output "evidence/workflow_robot_carried_vision_gate_${arm}_n48_certification.json" \
          --title "RGB-D chain, handoff gate = ${arm}, 16 environments, 48 episodes" \
          --scope "$SCOPE_1" "$SCOPE_2" \
            "One arm of the handoff-gate ablation. The guarded advance admits on the entry-flare catch (lead_in) or not at all (none); the detection interlock is active in both, so the only difference is whether being outside the next skill's envelope holds the stroke." \
            "$SCOPE_3" \
          > "$OUT/aggregate_${arm}.log" 2>&1
      rc=$?
      report_rc "$rc" "guard=$arm aggregate"
      tail -4 "$OUT/aggregate_${arm}.log"
    else
      say "  guard=$arm: only ${#rows[@]} of 3 seeds produced episodes; not aggregating"
    fi
  done
else
  say "  stage 1 skipped on memory"
fi

# ---------------------------------------------------------------------------
say "STAGE 2/2  gravity swept: orbit, the Moon, Mars, Earth"
# ---------------------------------------------------------------------------
# No rack retention here, deliberately. The question is what an unheld module
# does after the robot lets go, and retention answers it before gravity can.
OUT2=artifacts/campaign/gravity
mkdir -p "$OUT2"
if wait_for_memory; then
  for g in 0.0 -1.62 -3.71 -9.81; do
    tag=$(echo "$g" | tr -d '-.' )
    rows=()
    for seed in 4070 5070 6070; do
      out="$OUT2/g${tag}_seed${seed}"
      "$PY" scripts/run_workflow_demo.py --headless \
          --workflow relocate --curriculum_stage 0 --task "$STATE" \
          --grasp_checkpoint "$G" --extract_checkpoint "$NOISED" --insert_checkpoint "$I" \
          --num_envs 16 --seed "$seed" --steps 1900 \
          --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
          --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
          --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
          --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
          --release_sequence simultaneous \
          --module_gravity_z "$g" \
          --report "${out}_report.json" --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      rc=$?
      say "  gravity $g seed $seed exit=$rc"
      [ -f "${out}.npz" ] && rows+=("${out}.npz")
    done
    if [ "${#rows[@]}" -eq 3 ]; then
      ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
          --output "evidence/workflow_robot_carried_gravity_${tag}_n48_certification.json" \
          --title "Robot-carried relocation, module gravity ${g} m/s^2, released unheld, 48 episodes" \
          --scope "$SCOPE_1" \
            "Gravity acts on the module only. The manipulator and the rack are weightless, so the single quantity that changes across this sweep is whether a released part settles." \
            "The rack's retention is off, so the module is genuinely free after both robot-side supports release. That is the condition the orbital argument is about and it is not the shipped configuration." \
            "Exact state, not camera-derived." \
          > "$OUT2/aggregate_${tag}.log" 2>&1
      rc=$?
      report_rc "$rc" "gravity $g aggregate"
      tail -4 "$OUT2/aggregate_${tag}.log"
    else
      say "  gravity $g: only ${#rows[@]} of 3 seeds produced episodes; not aggregating"
    fi
  done
else
  say "  stage 2 skipped on memory"
fi

say "method supervisor done"
