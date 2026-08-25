#!/usr/bin/env bash
# Verify an insert checkpoint the way extraction is verified: alone, then in the chain.
#
# **Why both halves, and why in this order.** Extraction has a skill
# certification AND a chain that runs it, and the agreement between them is what
# makes the extraction number mean something. Insertion has only ever had the
# first. That gap is how this project spent months with a skill certifying at
# 0.00% while the chain seated at 97.92% -- two honest numbers describing
# different problems, with nothing standing between them.
#
# A skill that certifies alone and loses in the chain is the failure mode this
# repository has paid for most, so a checkpoint is not "working" until both have
# been run and published beside each other.
#
#   STAGE 1  three curriculum stages, three held-out seeds, pooled with a gate.
#            The same protocol every other skill is certified under.
#   STAGE 2  the full chain with --insert_controller policy, against the
#            scripted guarded advance on the identical workcell. Head to head.
#
# The chain arm is the one that decides whether the chain changes. The scripted
# guarded advance keeps the seating phase until a policy beats it on the same
# three held-out seeds, and BASELINE below names the arm it has to beat -- which
# has to be the guarded advance on the *same rack*, or the comparison mixes a
# geometry change into a controller change. The 97.92% in `docs/NOW.md` was
# measured at a 12.689 mm channel throat and is not that arm.
#
# Usage:
#   CKPT=logs/.../nn/last_..._ep_1400_....pth TAG=insert_v22attitude \
#       scripts/verify_insert_skill.sh
#
#   STAGES=0 scripts/verify_insert_skill.sh     # the fast look, stage 0 only
#   SKIP_CHAIN=1 scripts/verify_insert_skill.sh # skill only
#
# About an hour for the full thing: ~45 min for the nine skill runs and ~25 min
# for the three chain seeds.

set -u

CKPT="${CKPT:?set CKPT to the insert checkpoint to verify}"
TAG="${TAG:?set TAG for the evidence file names}"
STAGES="${STAGES:-0 1 2}"
SEEDS="${SEEDS:-4070 5070 6070}"
# The guarded-advance arm on the same rack, same seeds, same everything but the
# seating controller. Overridable so a later rack change can name its own.
BASELINE="${BASELINE:-evidence/workflow_robot_carried_m130pin_guarded_c11065_certification.json}"

if [ ! -f "$CKPT" ]; then echo "MISSING checkpoint: $CKPT"; exit 66; fi

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "VERIFYING $TAG"
say "  checkpoint $CKPT"
say "  sha256     $(sha256sum "$CKPT" | cut -c1-32)"

# ---------------------------------------------------------------------------
say "STAGE 1/2  skill certification, stages '$STAGES', three held-out seeds"
# ---------------------------------------------------------------------------
SKILL=Insert CKPT="$CKPT" TAG="$TAG" STAGES="$STAGES" \
  TITLE="Head-on grapple-pin insert skill, ${TAG}, orientation scaled to the channel" \
  scripts/certify_grapple_skills.sh
say "  -> evidence/grapple_${TAG}_certification.json"

if [ "${SKIP_CHAIN:-}" = "1" ]; then
  say "SKIP_CHAIN set; stopping before the chain arm"
  exit 0
fi

# ---------------------------------------------------------------------------
say "STAGE 2/2  the same checkpoint inside the full chain, against the guarded advance"
# ---------------------------------------------------------------------------
# Only the seating controller changes. Same task, same capture and extraction
# checkpoints, same three held-out seeds, same workcell -- so the difference
# between this and $BASELINE is the seating phase and cannot be anything else.
INSERT_CKPT="$CKPT" \
CERT_TAG="${TAG}_chain_policy" \
CERT_TITLE="Robot-carried relocation, seating driven by the learned insert policy (${TAG})" \
CHAIN_EXTRA="--insert_controller policy" \
SEEDS="$SEEDS" \
  scripts/run_robot_carried.sh certify
say "  -> evidence/workflow_robot_carried_${TAG}_chain_policy_certification.json"

# ---------------------------------------------------------------------------
# The decision, arithmetically rather than by eye.
# ---------------------------------------------------------------------------
"$PYTHON" scripts/report_seating_head_to_head.py \
    --guarded "$BASELINE" \
    --policy "evidence/workflow_robot_carried_${TAG}_chain_policy_certification.json" \
    --report "evidence/seating_controller_head_to_head.json"

echo
say "DONE. Publish both, whichever wins:"
say "  skill  evidence/grapple_${TAG}_certification.json"
say "  chain  evidence/workflow_robot_carried_${TAG}_chain_policy_certification.json"
say "  the arm it must beat: $BASELINE"
echo
say "The chain keeps the scripted advance unless the chain arm beats it on the"
say "same seeds. A skill number alone does not move the seating phase."
