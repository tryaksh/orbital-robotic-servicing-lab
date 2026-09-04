#!/usr/bin/env bash
# Certify the three checkpoints `run_workflow_demo.py` actually loads.
#
# evidence/ carried certifications for grasp v2, extract v2 and insert v3 while
# the demo ran grasp v3, extract v4 and insert v5, and every figure quoted about
# the demo therefore described a superseded policy. This script produces the
# matching reports, named after the version they describe so the two can never
# drift apart silently again.
#
# Evaluation only. No training happens here, and no source file is edited while
# it runs: every play.py launch re-imports the package, so a broken edit would
# fail every remaining run in the sweep rather than one.
#
# Usage: scripts/certify_demo_policies.sh [skill ...]   (default: all three)

set -u

PYTHON="C:/isaac-sim/python.bat"
EVAL_ENVS="${EVAL_ENVS:-128}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
OUT="artifacts/certify"
mkdir -p "$OUT" evidence

# Defaults name the **promoted** set in CLAUDE.md and must be moved with it.
# They previously named capture v3 and extract v4 after v5 and v13unsat were
# promoted, so an unattended re-run would have written a report labelled with the
# new version describing the old policy.
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
GRASP_CKPT="${GRASP_CKPT:-$CKPT_ROOT/grapple_grasp_l0_seed70_v6w65/nn/last_zero_g_blade_insertion_contact_ep_2400_rew__37.24023_.pth}"
EXTRACT_CKPT="${EXTRACT_CKPT:-$CKPT_ROOT/grapple_extract_l0_seed70_v16w65/nn/last_zero_g_blade_insertion_contact_ep_9700_rew__176.34572_.pth}"
INSERT_CKPT="${INSERT_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v12w65/nn/last_zero_g_blade_insertion_contact_ep_7100_rew_-20.706831.pth}"
# The report file is named for the policy version, so a retrain must pass its
# own. Defaults describe the promoted plain-pin lineage.
GRASP_VERSION="${GRASP_VERSION:-v5}"
EXTRACT_VERSION="${EXTRACT_VERSION:-v14reset}"
INSERT_VERSION="${INSERT_VERSION:-v6}"
# Which physical interface these policies were trained against, recorded in the
# report so certifications of two different interfaces can never be read as
# comparable. There has only ever been one shipped interface -- the tapered pin
# -- but two alternatives were built and measured against it, and both are
# refuted: the anti-yaw yoke, whose code was deleted on 2026-08-18, and the keyed
# redesign, which does not fit inside the gripper
# (evidence/grapple_pin_keyed_interference.json).
INTERFACE="${INTERFACE:-the plain grapple pin}"

skills=("$@")
if [ ${#skills[@]} -eq 0 ]; then
  skills=(Grasp Extract Insert)
fi

for skill in "${skills[@]}"; do
  case "$skill" in
    Grasp)   checkpoint="$GRASP_CKPT";   version="$GRASP_VERSION"; stages=(0 1 2) ;;
    Extract) checkpoint="$EXTRACT_CKPT"; version="$EXTRACT_VERSION"; stages=(0 1 2) ;;
    # The insert task collapsed to a single reset distance when it was retrained
    # for the chain, so its poses_by_stage carries one entry. Asking for stage 1
    # or 2 would clamp back onto the same pose and file three copies of one
    # measurement under three stage labels.
    Insert)  checkpoint="$INSERT_CKPT";  version="$INSERT_VERSION"; stages=(0) ;;
    *) echo "unknown skill $skill"; exit 1 ;;
  esac
  if [ ! -f "$checkpoint" ]; then
    echo "[$(date +%H:%M:%S)] MISSING checkpoint for $skill: $checkpoint"
    exit 1
  fi

  play="Isaac-ZeroG-Blade-GrapplePin-${skill}-Play-v0"
  lower=$(echo "$skill" | tr '[:upper:]' '[:lower:]')
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] CERTIFY $skill $version"
  echo "  $checkpoint"
  echo "=============================================================="

  rows=()
  for seed in 1070 2070 3070; do
    for stage in "${stages[@]}"; do
      out="$OUT/${lower}_${version}_s${stage}_seed${seed}"
      "$PYTHON" scripts/play.py --headless \
          --task "$play" \
          --checkpoint "$checkpoint" \
          --num_envs "$EVAL_ENVS" \
          --episodes "$EVAL_EPISODES" \
          --curriculum_stage "$stage" \
          --seed "$seed" \
          --report "${out}_play.json" \
          --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      rc=$?
      echo "[$(date +%H:%M:%S)]   eval $skill stage=$stage seed=$seed exit=$rc"
      rows+=("${out}.npz")
    done
  done

  "$PYTHON" scripts/aggregate_evaluation.py \
      --episodes "${rows[@]}" \
      --output "evidence/grapple_${lower}_${version}_certification.json" \
      --title "Head-on grapple-pin ${lower} skill, ${version} (the checkpoint the workflow demo loads)" \
      --scope \
        "Simulation only. No result here was produced on real hardware." \
        "Interface: ${INTERFACE}. A policy trained against one pin says nothing about the other." \
        "The grasp is physical pad-against-pin contact, not a fixed joint." \
        "One PPO training seed. The evaluation seeds are held out, but training repeatability is untested." \
        "This certifies the skill in isolation. It says nothing about the chained workflow, which is certified separately." \
      > "$OUT/aggregate_${lower}_${version}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] aggregate $skill exit=$rc -> evidence/grapple_${lower}_${version}_certification.json"
  tail -5 "$OUT/aggregate_${lower}_${version}.log"
done

echo "[$(date +%H:%M:%S)] CERTIFY DONE"
