#!/usr/bin/env bash
# Does the gate buy more when the manipulator is worse? The dose-response arm.
#
# **This is the experiment that turns "the gate helps" into a claim a designer
# can use.** Tonight's ablation measures the gate on and off at the shipped robot
# base. On its own that says the gate helped once. The useful statement is
# directional: *the value of gating on the next skill's precondition scales with
# how imprecise the manipulator is* -- so a team with a good arm can skip it and
# a team with a poor one cannot. A monotone relationship is far harder to
# dismiss than a single contrast, and it is the form an industrial reader needs,
# because their question is not "does it work" but "do I need it".
#
# The prediction, stated before the runs so it can be wrong: at a base pose with
# better realised authority the gated and ungated arms should converge, and at
# the shipped pose they should separate. `evidence/workcell_geometry_check.json`
# reports the shipped base at 0.9439 worst crossing authority and 0.0410 minimum
# singular value, against 0.9977 and 0.2097 at x=-0.85. The shipped cell is the
# *worse* of the two, which is the direction that makes the test meaningful.
#
# Only the better base is run here. The shipped-base pair is tonight's ablation
# and is not re-run, so the two share checkpoints, seeds, cohort size and commit.
#
# **A smoke test first.** The capture and extraction policies were trained at
# x=-0.65 and nothing has ever run this chain from -0.85; `--robot_base_x` also
# has a three-layer defect in its history. Four environments and 400 steps decide
# whether the full batch is worth three and a half hours.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED="$ROOT/grapple_extract_l0_seed70_v19noised/nn/last_zero_g_blade_insertion_contact_ep_14600_rew_166.19054.pth"
NCAP=$(ls -t "$ROOT/grapple_grasp_l0_seed70_v9noised/nn/"*ep_5100_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
VISION="Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
BASE_X=-0.85
OUT=artifacts/campaign/dose
mkdir -p "$OUT"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

say () { echo "[$(date +%H:%M:%S)] $*"; }

run_chain () {
  local out="$1" arm="$2" envs="$3" steps="$4" seed="$5"
  "$PY" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 --task "$VISION" \
      --grasp_checkpoint "$NCAP" --extract_checkpoint "$NOISED" --insert_checkpoint "$I" \
      --num_envs "$envs" --seed "$seed" --steps "$steps" \
      --robot_base_x "$BASE_X" \
      --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
      --module_velocity_source kinematics --fiducial_guard_bounds "$arm" \
      --report "${out}_report.json" --episode_metrics "${out}.npz" \
      > "${out}.log" 2>&1
}

until grep -q "method supervisor done" artifacts/campaign/supervise_method.log 2>/dev/null; do sleep 300; done
say "method batch finished; dose-response at robot base x=$BASE_X"

if [ -z "$NCAP" ]; then say "no ep_5100 noised capture; refusing to start"; exit 1; fi

# ---------------------------------------------------------------------------
say "SMOKE  four environments at x=$BASE_X, before committing three and a half hours"
# ---------------------------------------------------------------------------
run_chain "$OUT/smoke" lead_in 4 900 4070
rc=$?
say "  smoke exit=$rc"
if [ ! -f "$OUT/smoke.npz" ]; then
  say "  the smoke run wrote no episodes; the chain does not survive this base pose"
  grep -aE "Traceback|RuntimeError|Refusing" "$OUT/smoke.log" | head -5
  say "dose supervisor done"
  exit 0
fi
reached=$(./.venv/Scripts/python.exe -c "
import numpy as np
a=np.load('$OUT/smoke.npz',allow_pickle=True); f=[str(x) for x in a['fields']]; r=a['rows'].astype(float)
print(int((r[:, f.index('reached_phase')] >= 4).sum()))" 2>/dev/null || echo 0)
say "  $reached of 4 environments reached the insert phase"
if [ "$reached" -lt 1 ]; then
  say "  nothing reached insertion at this base pose; the gate cannot be measured here"
  say "dose supervisor done"
  exit 0
fi

# ---------------------------------------------------------------------------
say "DOSE  gate on and ablated, at the better base pose, three seeds each"
# ---------------------------------------------------------------------------
for arm in lead_in none; do
  rows=()
  for seed in 4070 5070 6070; do
    out="$OUT/base85_${arm}_seed${seed}"
    run_chain "$out" "$arm" 16 1900 "$seed"
    rc=$?
    say "  base=$BASE_X guard=$arm seed $seed exit=$rc"
    [ -f "${out}.npz" ] && rows+=("${out}.npz")
  done
  if [ "${#rows[@]}" -eq 3 ]; then
    ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/workflow_robot_carried_vision_dose_base85_gate_${arm}_n48_certification.json" \
        --title "RGB-D chain, robot base x=-0.85, handoff gate = ${arm}, 48 episodes" \
        --scope "$SCOPE_1" "$SCOPE_2" \
          "The dose-response arm of the handoff-gate ablation. The manipulator is parked at a pose with better realised crossing authority than the shipped one (0.9977 against 0.9439), so the gate has less to absorb. Everything else matches the shipped-base pair measured the same night." \
          "$SCOPE_3" \
        > "$OUT/aggregate_${arm}.log" 2>&1
    rc=$?
    case "$rc" in
      0) say "  base85 guard=$arm: gate passed" ;;
      2) say "  base85 guard=$arm: gate not met, which for this arm is the measurement" ;;
      *) say "  base85 guard=$arm: COHORT REFUSED (rc=$rc)" ;;
    esac
    tail -4 "$OUT/aggregate_${arm}.log"
  else
    say "  base85 guard=$arm: only ${#rows[@]} of 3 seeds produced episodes; not aggregating"
  fi
done

say "dose supervisor done"
