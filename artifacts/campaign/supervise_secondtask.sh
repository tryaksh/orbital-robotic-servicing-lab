#!/usr/bin/env bash
# The second task, which is the difference between a case study and a method.
#
# **Everything measured so far is one job on one workcell.** The handoff claim --
# gate on the next skill's physical precondition rather than on the previous
# skill reporting success -- is demonstrated only on `relocate`: capture, pull
# clear, fly three legs, seat. A reviewer's first question is whether the gate is
# a property of the principle or of that particular job, and there is no answer
# in the repository.
#
# There are two other workflows and neither has ever been certified, though both
# have been run: `install` (capture a module at the rack mouth and seat it) and
# `remove` (capture an installed module and pull it clear). `install` is the one
# to take first. It shares the seating handoff, which is where the gate acts, and
# it drops the extraction and the three transit legs -- so the module arrives at
# the bay having taken a completely different path, which is exactly the variable
# that matters. An earlier run scored 85.94% on the state task at 64
# environments, so the path works.
#
# **A smoke test decides whether the batch is worth running at all.** The guard
# reads `relocation_staging_rot`, which is named for the workflow this has never
# been run under, and the camera-driven single-bay task has not carried a guarded
# insertion before. Four environments answer that; three hours should not.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
TASK="Isaac-ZeroG-Blade-GrappleVision-Workflow-v0"
OUT=artifacts/campaign/secondtask
mkdir -p "$OUT"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."

say () { echo "[$(date +%H:%M:%S)] $*"; }

run_install () {
  local out="$1" arm="$2" envs="$3" steps="$4" seed="$5"
  "$PY" scripts/run_workflow_demo.py --headless \
      --workflow install --curriculum_stage 2 --task "$TASK" \
      --grasp_checkpoint "$G" --extract_checkpoint "$E" --insert_checkpoint "$I" \
      --num_envs "$envs" --seed "$seed" --steps "$steps" \
      --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
      --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous --perception_backend fiducial_pnp \
      --fiducial_guard_bounds "$arm" \
      --report "${out}_report.json" --episode_metrics "${out}.npz" \
      > "${out}.log" 2>&1
}

until grep -q "dose supervisor done" artifacts/campaign/supervise_dose.log 2>/dev/null; do sleep 300; done
say "dose batch finished; the second task"

# ---------------------------------------------------------------------------
say "SMOKE  install on the camera-driven single-bay task, four environments"
# ---------------------------------------------------------------------------
run_install "$OUT/smoke" lead_in 4 900 4070
rc=$?
say "  smoke exit=$rc"
if [ ! -f "$OUT/smoke.npz" ]; then
  say "  no episodes written; install does not run on this task with a guarded insertion"
  grep -aE "Traceback|RuntimeError|Refusing|AttributeError" "$OUT/smoke.log" | head -5
  say "second task supervisor done"
  exit 0
fi
reached=$(./.venv/Scripts/python.exe -c "
import numpy as np
a=np.load('$OUT/smoke.npz',allow_pickle=True); f=[str(x) for x in a['fields']]; r=a['rows'].astype(float)
print(int((r[:, f.index('reached_phase')] >= 4).sum()))" 2>/dev/null || echo 0)
say "  $reached of 4 environments reached the insert phase"
if [ "$reached" -lt 1 ]; then
  say "  nothing reached insertion; the gate cannot be measured on this workflow"
  say "second task supervisor done"
  exit 0
fi

# ---------------------------------------------------------------------------
say "SECOND TASK  the same gate, on install rather than relocate, three seeds"
# ---------------------------------------------------------------------------
for arm in lead_in none; do
  rows=()
  for seed in 4070 5070 6070; do
    out="$OUT/install_${arm}_seed${seed}"
    run_install "$out" "$arm" 16 1900 "$seed"
    rc=$?
    # Verify by the artifact, never by the status. On 2026-09-05 seeds 5070 and
    # 6070 exited 0 after 36 seconds having written no episodes -- the session
    # was closing and both Isaac processes died in scene setup -- and the line
    # below said "exit=0" for a run that produced nothing. The aggregation was
    # never at risk, because `rows` is built from the file and the arm refuses
    # to aggregate below three seeds. The log was the only thing that lied.
    if [ -f "${out}.npz" ]; then
      say "  install guard=$arm seed $seed exit=$rc, episodes written"
      rows+=("${out}.npz")
    else
      say "  install guard=$arm seed $seed exit=$rc but NO EPISODES WRITTEN -- not a run"
    fi
  done
  if [ "${#rows[@]}" -eq 3 ]; then
    ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/workflow_install_vision_gate_${arm}_n48_certification.json" \
        --title "RGB-D install workflow, handoff gate = ${arm}, 16 environments, 48 episodes" \
        --scope "$SCOPE_1" "$SCOPE_2" \
          "The second task for the handoff-gate claim. The module is captured at the rack mouth and seated; there is no extraction and no transit, so it arrives at the bay by a different path than in the relocate workflow the gate was measured on." \
          "Success is the workflow's own condition re-checked after a 0.70 s settling window. No rack retention, which the single-bay profile does not carry." \
        > "$OUT/aggregate_${arm}.log" 2>&1
    rc=$?
    case "$rc" in
      0) say "  install guard=$arm: gate passed" ;;
      2) say "  install guard=$arm: gate not met, which for this arm is the measurement" ;;
      *) say "  install guard=$arm: COHORT REFUSED (rc=$rc)" ;;
    esac
    tail -4 "$OUT/aggregate_${arm}.log"
  else
    say "  install guard=$arm: only ${#rows[@]} of 3 seeds produced episodes; not aggregating"
  fi
done

# The paired reading, which is how every A/B in this project should be read.
if [ -f "$OUT/install_none_seed6070.npz" ] && [ -f "$OUT/install_lead_in_seed6070.npz" ]; then
  ./.venv/Scripts/python.exe scripts/compare_paired_arms.py \
      --baseline "$OUT/install_none_seed4070.npz" "$OUT/install_none_seed5070.npz" "$OUT/install_none_seed6070.npz" \
      --treatment "$OUT/install_lead_in_seed4070.npz" "$OUT/install_lead_in_seed5070.npz" "$OUT/install_lead_in_seed6070.npz" \
      --label "the handoff gate on the install workflow, 48 paired episodes" \
      --report evidence/handoff_gate_install_paired_n48.json
  rc=$?
  say "  paired reading exit=$rc -> evidence/handoff_gate_install_paired_n48.json"
fi

say "second task supervisor done"
