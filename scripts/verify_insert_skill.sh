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
# **Stage 0 only, and that is the task rather than a shortcut.** The insert task
# runs SingleStageCurriculumCfg with max_stage 0, so --curriculum_stage 1 raises
# "forced insertion stage must be in [0, 0]" and writes no rows. The default was
# "0 1 2", which cost six fast failures per verification and filled the logs with
# a ValueError that looks like a defect and is not. The published insert
# certification is stage 0 on three held-out seeds; this matches it.
STAGES="${STAGES:-0}"
SEEDS="${SEEDS:-4070 5070 6070}"
# The guarded-advance arm on the same rack, same seeds, same everything but the
# seating controller. Overridable so a later rack change can name its own.
BASELINE="${BASELINE:-evidence/workflow_robot_carried_m130pin_guarded_c11065_certification.json}"
# **The chain arm runs in the bay the default BASELINE was measured in, which is
# the unrelieved one.** `run_robot_carried.sh` defaults to
# `--destination_channel_relief_m 0.0046125`, a 15.678 mm channel that is
# 3.897 mm past the design library's own upper bound; the baseline named above is
# the 11.065 mm design point. Taking that default silently made the head to head
# a controller change *and* a 4.6 mm geometry change, which the header of this
# script says it must never be. The skill half needs no such setting: `play.py`
# has no relief flag, so the skill task is always the design-point bay. Override
# RELIEF only together with BASELINE.
export RELIEF="${RELIEF:-0.0}"
# `report_seating_head_to_head.py` at the end of this script reads `$PYTHON`,
# which nothing set. Under `set -u` that aborted the run after both arms had been
# paid for, so the 2026-09-04 force verification produced two evidence files and
# no decision between them.
PYTHON="${PYTHON:-C:/isaac-sim/python.bat}"

if [ ! -f "$CKPT" ]; then echo "MISSING checkpoint: $CKPT"; exit 66; fi

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "VERIFYING $TAG"
say "  checkpoint $CKPT"
say "  sha256     $(sha256sum "$CKPT" | cut -c1-32)"

# ---------------------------------------------------------------------------
say "STAGE 1/2  skill certification, stages '$STAGES', three held-out seeds"
# ---------------------------------------------------------------------------
SKILL=Insert CKPT="$CKPT" TAG="$TAG" STAGES="$STAGES" PLAY_TASK="${PLAY_TASK:-}" \
  TITLE="Head-on grapple-pin insert skill, ${TAG}, orientation scaled to the channel" \
  scripts/certify_grapple_skills.sh
skill_rc=$?
say "  -> evidence/grapple_${TAG}_certification.json"
# A skill half that writes nothing is the failure this script exists to expose,
# and it used to scroll past as one line among forty. `InsertForce-Play-v0`
# raised at construction, all three runs exited in ten seconds, and stage 2 began
# as though stage 1 had been measured.
if [ ! -f "evidence/grapple_${TAG}_certification.json" ]; then
  say "  STAGE 1 PRODUCED NO CERTIFICATION (certify exit=$skill_rc) -- the skill half did not run"
  say "  the chain half below still runs, but this verification is half a verification"
fi

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
# CHAIN_TASK likewise: a force-feedback policy needs the workflow task that
# reports contact, or the chain hands it an observation of the wrong width.
TASK="${CHAIN_TASK:-}" INSERT_CKPT="$CKPT" \
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
