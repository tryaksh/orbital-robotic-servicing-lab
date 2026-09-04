#!/usr/bin/env bash
# Collect the pose-head dataset across camera calibrations, not just one.
#
# Measured on a head trained through a perfectly mounted camera: a 2 mm mount
# error quadruples its position error and a 5 mm error puts a third of its
# estimates outside the capture tolerance. That is a real barrier to anyone
# building against this, because it demands a mount and a recalibration
# interval nobody wants to promise.
#
# The standard answer is to randomize what you cannot control. Each run below
# collects through a differently mis-mounted camera, drawn inside the tolerance
# an integrator could plausibly hold, and the runs are concatenated into one
# training set. The offsets are per-run rather than per-episode because that is
# where the perturbation demonstrably reaches the render.

set -u
PYTHON="C:/isaac-sim/python.bat"
PER_RUN="${PER_RUN:-8000}"
ENVS="${ENVS:-64}"
mkdir -p datasets/calib artifacts

# Nominal first, so the robust head is never worse than the brittle one at the
# calibration it will actually be commissioned at.
i=0
for spec in "0 0" "3 4" "6 8" "9 12" "4 10" "8 3" "2 14" "10 6"; do
  set -- $spec
  mm=$1; mrad=$2
  echo "[$(date +%H:%M:%S)] run $i: mount ${mm} mm, tilt ${mrad} mrad"
  "$PYTHON" scripts/collect_grapple_vision.py --headless \
      --samples "$PER_RUN" --num_envs "$ENVS" \
      --camera_offset_mm "$mm" --camera_tilt_mrad "$mrad" \
      --seed $((90 + i)) \
      --output "datasets/calib/run_${i}.npz" >> artifacts/collect_calib.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)]   exit=$rc"
  i=$((i + 1))
done
echo "[$(date +%H:%M:%S)] CALIBRATION COLLECTION DONE"
