#!/usr/bin/env bash
# The decisive comparisons again, at a cohort size that can resolve them.
#
# **Today's two headline comparisons are both inconclusive by construction, and
# that is a sample-size problem rather than a result.** The seating head to head
# is 7/24 against 8/24 -- Wilson [14.9, 49.2] against [18.0, 53.3], intervals
# that overlap across almost their whole range. Twenty-four episodes cannot
# separate two controllers unless one of them is catastrophic.
#
# And `base_000` re-ran at 7/24 and then at a materially different rate with the
# same three seeds, the same three checkpoints and no source change between the
# commits (`git diff 031c70d 6bc5cd2 -- src/ scripts/` is empty). So a single
# 24-episode cell carries run-to-run spread comparable to the differences the
# 2x2x2 is being read off.
#
# The fix is episodes, not processes. A fourth Isaac process is what the memory
# will not take: three of them hold 4 to 6 GB resident each and leave about
# 5.6 GB of 31.4 free, and `AGENTS.md` records that under about 3 GB the machine
# pages and jobs get OOM-killed -- which is how the 06:25 campaign died. VRAM has
# room, system RAM does not, so the environments go up and the process count
# stays where it is.
#
#   A  the seating head to head at 32 environments: 96 episodes an arm, not 24.
#      The force workflow task renders no cameras, so this is the cheap one.
#   B  the camera-driven centrepiece at 16 environments: 48 episodes, not 24.
#      RGB-D is rendered per environment, which is why the vision cohorts have
#      always been 8, so this steps up rather than leaps.
#
# Both stages re-measure something already published at the small size. Neither
# overwrites it: the small cohorts stay in `evidence/` as their own arms, which
# is what makes "the interval was too wide" a checkable statement rather than a
# recollection.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED="$ROOT/grapple_extract_l0_seed70_v19noised/nn/last_zero_g_blade_insertion_contact_ep_14600_rew_166.19054.pth"
FORCE="$ROOT/grapple_insert_l0_seed70_v33force/nn/last_zero_g_blade_insertion_contact_ep_3000_rew_98.33571.pth"
FORCE_TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflowForce-v0"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

say () { echo "[$(date +%H:%M:%S)] $*"; }

# Free system memory in MB. The guard below is the one rule that has actually
# cost this project a night: VRAM is not the limit, resident memory is.
free_mb () {
  powershell -NoProfile -Command '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)' 2>/dev/null || echo 9999
}

wait_for_memory () {
  waited=0
  while [ "$(free_mb)" -lt 4000 ]; do
    say "  only $(free_mb) MB free; holding rather than risking the OOM that killed 06:25"
    sleep 300
    waited=$((waited + 300))
    if [ "$waited" -gt 7200 ]; then say "  two hours under 4 GB free; giving up on this stage"; return 1; fi
  done
  return 0
}

until grep -q "evaluation supervisor 2 done" artifacts/campaign/supervise_evaluation2.log 2>/dev/null; do sleep 300; done
say "evaluation slot free; re-measuring both headline comparisons at size"

# ---------------------------------------------------------------------------
say "STAGE A/2  the seating head to head at 32 environments, 96 episodes an arm"
# ---------------------------------------------------------------------------
if wait_for_memory; then
  TASK="$FORCE_TASK" INSERT_CKPT="$FORCE" RELIEF=0.0 CERT_ENVS=32 \
  CERT_TAG="insert_v33force_c11065_chain_guarded_n96" \
  CERT_TITLE="Robot-carried relocation, scripted guarded advance, force task, 11.065 mm channel, 32 environments" \
    bash scripts/run_robot_carried.sh certify
  rc=$?
  say "  guarded arm exit=$rc"

  TASK="$FORCE_TASK" INSERT_CKPT="$FORCE" RELIEF=0.0 CERT_ENVS=32 \
  CERT_TAG="insert_v33force_c11065_chain_policy_n96" \
  CERT_TITLE="Robot-carried relocation, learned force-feedback seating, force task, 11.065 mm channel, 32 environments" \
  CHAIN_EXTRA="--insert_controller policy" \
    bash scripts/run_robot_carried.sh certify
  rc=$?
  say "  policy arm exit=$rc"

  ./.venv/Scripts/python.exe scripts/report_seating_head_to_head.py \
      --guarded "evidence/workflow_robot_carried_insert_v33force_c11065_chain_guarded_n96_certification.json" \
      --policy "evidence/workflow_robot_carried_insert_v33force_c11065_chain_policy_n96_certification.json" \
      --report "evidence/seating_controller_head_to_head_n96.json"
  rc=$?
  say "  head to head at 96 episodes an arm exit=$rc -> evidence/seating_controller_head_to_head_n96.json"
else
  say "  stage A skipped on memory"
fi

# ---------------------------------------------------------------------------
say "STAGE B/2  the camera-driven chain on both retrained skills, 16 environments"
# ---------------------------------------------------------------------------
NCAP=$(ls -t "$ROOT/grapple_grasp_l0_seed70_v9noised/nn/"*ep_5100_*.pth 2>/dev/null | grep -vE '_rew__' | head -1)
if [ -z "$NCAP" ]; then
  say "  no ep_5100 noised capture; skipping stage B"
  say "big cohort supervisor done"
  exit 0
fi
if wait_for_memory; then
  OUT=artifacts/campaign/noisedchain48
  mkdir -p "$OUT"
  rows=()
  for seed in 4070 5070 6070; do
    out="$OUT/noisedcap48_seed${seed}"
    "$PY" scripts/run_workflow_demo.py --headless \
        --workflow relocate --curriculum_stage 0 \
        --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
        --grasp_checkpoint "$NCAP" --extract_checkpoint "$NOISED" --insert_checkpoint "$I" \
        --num_envs 16 --seed "$seed" --steps 1900 \
        --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
        --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
        --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
        --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
        --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
        --module_velocity_source kinematics --fiducial_guard_bounds lead_in \
        --report "${out}_report.json" --episode_metrics "${out}.npz" \
        > "${out}.log" 2>&1
    rc=$?
    say "  noised-capture chain, 16 envs, seed $seed exit=$rc"
    [ -f "${out}.npz" ] && rows+=("${out}.npz")
  done
  if [ "${#rows[@]}" -eq 3 ]; then
    ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/workflow_robot_carried_vision_noised_capture_extract_kinematic_leadin_n48_certification.json" \
        --title "RGB-D chain, both skills trained on the estimator's error, 16 environments, 48 episodes" \
        --scope "$SCOPE_1" "$SCOPE_2" \
          "The published camera-driven arm with one term changed -- the capture policy is the noised fine-tune -- measured at 16 environments rather than 8 because the 24-episode cohorts this project publishes carry intervals too wide to separate the arms." \
          "$SCOPE_3" \
        > "$OUT/aggregate_noisedcap48.log" 2>&1
    rc=$?
    say "  aggregate exit=$rc"
    tail -4 "$OUT/aggregate_noisedcap48.log"
  else
    say "  only ${#rows[@]} of 3 seeds produced episodes; not aggregating"
  fi
else
  say "  stage B skipped on memory"
fi

say "big cohort supervisor done"
