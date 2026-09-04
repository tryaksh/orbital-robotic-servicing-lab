#!/usr/bin/env bash
# There is no perception certification for the datum layout actually deployed.
#
# Every fiducial certificate in evidence/ -- v2 seed 283, v3 seed 284, v4 seed
# 285 -- was collected when the module carried a single centred datum. The
# deployed system carries a *pair*, ArUco 23 aft and ArUco 15 forward at
# module-frame x = -+0.115 m, and that change was made after the last one.
#
# The mismatch is not theoretical. Replaying the preserved v4 dataset through the
# current detector gives a position p95 of 116.08 mm against the published
# 1.91 mm, and 116 mm is the datum offset: the current detector resolves a plate
# at 115 mm off centre in frames whose ground truth assumes a centred one. So the
# old datasets cannot be re-certified, and the surrogate that noises the skills'
# observations is calibrated from a certificate for a datum layout the chain no
# longer has.
#
# This collects 1,024 held-out frames on the current scene and certifies them.
# Seed 286, held out from every previous collection.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
until grep -q "noised skill certification done" artifacts/campaign/noised_skill_cert.log 2>/dev/null; do sleep 120; done
mkdir -p artifacts/campaign/datumpair datasets

echo "[$(date +%H:%M:%S)] collecting 1024 workflow-envelope frames on the datum pair"
"$PY" scripts/collect_grapple_vision.py \
    --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Collect-v0 \
    --output datasets/fiducial_rgbd_datum_pair_seed286.npz \
    --samples 1024 --num_envs 16 --seed 286 \
    --rgb_source raw --pose_distribution workflow_envelope \
    > artifacts/campaign/datumpair/collect.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] collect exit=$rc"

if [ -f datasets/fiducial_rgbd_datum_pair_seed286.npz ]; then
  echo "[$(date +%H:%M:%S)] certifying"
  "$PY" scripts/certify_fiducial_perception.py \
      --dataset datasets/fiducial_rgbd_datum_pair_seed286.npz \
      --report evidence/fiducial_rgbd_datum_pair_seed286.json \
      > artifacts/campaign/datumpair/certify.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] certify exit=$rc"
  tail -20 artifacts/campaign/datumpair/certify.log
else
  echo "[$(date +%H:%M:%S)] no dataset written; nothing to certify"
  tail -20 artifacts/campaign/datumpair/collect.log
fi
echo "[$(date +%H:%M:%S)] datum pair perception done"
