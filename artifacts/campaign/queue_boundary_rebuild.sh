#!/usr/bin/env bash
# When the corrected clearance points land, rebuild the sweep report and the
# boundary decision on them, keeping the guides-only arm as the loser.
set -u
cd /d/6axis-space-robotics || exit 1
while [ ! -f artifacts/robustness64_channel/rack_lat_16mm.npz ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] corrected clearance points landed"
rm -rf artifacts/robustness64_corrected
mkdir -p artifacts/robustness64_corrected
# Everything the clearance flag does not touch comes from the published sweep;
# only the two clearance points are replaced, and the arm is named in the report.
cp artifacts/robustness64/nominal.npz artifacts/robustness64_corrected/
cp artifacts/robustness64/nominal_report.json artifacts/robustness64_corrected/
for p in section_120x16 section_140x26 base_x_-0.70 base_y_+10mm; do
  cp "artifacts/robustness64/${p}.npz" artifacts/robustness64_corrected/
  cp "artifacts/robustness64/${p}_report.json" artifacts/robustness64_corrected/ 2>/dev/null || true
done
for p in rack_lat_6mm rack_lat_16mm; do
  cp "artifacts/robustness64_channel/${p}.npz" artifacts/robustness64_corrected/
  cp "artifacts/robustness64_channel/${p}_report.json" artifacts/robustness64_corrected/ 2>/dev/null || true
done
./.venv/Scripts/python.exe scripts/report_chain_robustness.py \
    --sweep_dir artifacts/robustness64_corrected \
    --report evidence/chain_robustness_sweep_n64_channel_v1.json
rc=$?
echo "[$(date +%H:%M:%S)] sweep report exit=$rc"
./.venv/Scripts/python.exe scripts/report_boundary_failure_modes.py \
    --sweep_dir artifacts/robustness64_corrected \
    --compare_dir artifacts/robustness64 \
    --compare_label "guides-only clearance arm, preserved" \
    --report evidence/boundary_failure_modes_v1.json
rc=$?
echo "[$(date +%H:%M:%S)] failure-mode report exit=$rc"
