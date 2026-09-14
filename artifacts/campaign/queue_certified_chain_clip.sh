#!/usr/bin/env bash
# One recording of the chain as it is currently certified, and its own report.
#
# **Every clip in the repository predates the current result.** The nearest thing
# to a recording of the present chain is the RGB-D strict run at seed 6070, which
# is a complete changeout and a different task: it is driven by camera-derived
# module state, and the certified 22/24 is the state task. So no recording shows
# the certified chain, which is what this fixes.
#
# The arm is the certified one exactly -- state task, robot rail, form lock,
# `--rack_retention`, simultaneous release, the 0.70 s rack-only recheck -- with
# two differences forced by the recorder and disclosed here:
#
#   ONE ENVIRONMENT.  `--video` refuses `--num_envs > 1`, and the certification
#   runs eight. So this draws a single environment's reset state rather than one
#   of the certified cohort's eight, and it is a demonstration, not a rate. The
#   pooled rate stays 22/24.
#
#   STABLE LIGHTING.  The recorder needs a fixed exposure and the evidence needs
#   the randomization it was certified under; they are exclusive. The state task
#   carries no perception, so this costs nothing measured here -- but it is still
#   a difference and it is named.
#
# CHECK THE REPORT, NOT THE FILENAME. `3_full_chain_seated.mp4` was named from
# what the run looked like and its own report said `lateral_alignment: false` at
# 4.62 mm against a 2.5 mm tolerance. The three fields that decide whether a clip
# may be published as a completed changeout:
#
#   seated_conditions_still_held_after_settling
#   all_conditions_including_released_gripper
#   destination_rack_retention.observed_per_environment[0].full_rack_only_recheck_observed
#
# If they are not all true, the clip is a record of a failure whatever it looks
# like, and `docs/DEMOS.md` says so instead.
#
# About 8 minutes a seed. 5070 and 6070 both scored 8/8 in the certified cohort,
# so either is a reasonable draw; neither is a promise, because one environment
# is not one of that cohort's eight.
set -u
cd /d/6axis-space-robotics || exit 1

PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="artifacts/robotcarried"

say () { echo "[$(date +%H:%M:%S)] $*"; }

for seed in ${SEEDS:-5070 6070}; do
  tag="certified_chain_clip_seed${seed}"
  say "RECORDING the certified state chain, seed ${seed}"
  "$PYTHON" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 \
      --task Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0 \
      --grasp_checkpoint "$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth" \
      --extract_checkpoint "$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth" \
      --insert_checkpoint "$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth" \
      --num_envs 1 --seed "$seed" --steps 1900 \
      --robot_rail_on_relocation \
      --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 \
      --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m 0.0046125 \
      --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous \
      --rack_retention \
      --stable_lighting --inspection_view workcell \
      --video --video_dir "$OUT/video_${tag}" \
      --report "$OUT/${tag}_report.json" \
      > "$OUT/${tag}.log" 2>&1
  rc=$?
  say "  exit=$rc -> $OUT/${tag}_report.json"
  if [ "$rc" -ne 0 ]; then
    say "  run failed; not checking its report"
    continue
  fi
  # The three fields, read out of the report rather than off the screen.
  ./.venv/Scripts/python.exe - "$OUT/${tag}_report.json" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
retention = (report.get("destination_rack_retention") or {}).get("observed_per_environment") or [{}]
fields = {
    "seated_conditions_still_held_after_settling": report.get(
        "seated_conditions_still_held_after_settling"
    ),
    "all_conditions_including_released_gripper": report.get(
        "all_conditions_including_released_gripper"
    ),
    "full_rack_only_recheck_observed": retention[0].get("full_rack_only_recheck_observed"),
}
for name, value in fields.items():
    print(f"    {name} = {value}")
print("    PUBLISHABLE" if all(value is True for value in fields.values()) else "    NOT A SEATED CHAIN")
EOF
done
say "done"
