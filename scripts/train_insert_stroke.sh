#!/usr/bin/env bash
# Train the insert skill on the stroke it is actually asked to drive.
#
# The checkpoint the chain carries certifies at 0.00% and the reason is not the
# policy: it reset at one module pose 167 mm from the seated plane and the chain
# hands it the module at the mouth, 529 mm out. Every state the chain produces
# was outside the distribution, so the driver seats with a scripted advance and
# `--insert_controller policy` exists only to keep that a measurement.
#
# What changed under this task:
#
#  * the reset spans the stroke, arm and module and fingers written together
#    from a bank solved in closed form (scripts/solve_insert_reset_bank.py);
#  * it starts already holding, at the closure the pads come to rest at, because
#    at the shallow end the module is outside the rack and closing on a free
#    mass in zero gravity throws it;
#  * the axial action scale is sized for the stroke, 45 -> 120 mm/s, from the
#    measured travel-to-cycle ratio and the unchanged 30 s budget;
#  * both bays seat at the plane the release interlock permits, and bay 1's goal
#    is no longer 74 mm past it.
#
# **And since 2026-08-25 the task carries the chain's load path itself.** The
# form lock, softened into the remote-centre mating compliance at control step
# 5, used to be reachable only through `train.py --latch_mating_compliance` --
# which `scripts/verify_insert_skill.sh` does not pass to `play.py`, so a
# checkpoint trained on the lock would have been certified on pad contact alone.
# It is on `ZeroGBladeGrapplePinInsertTwoSlotEnvCfg` now, so training and
# certification cannot disagree about it. Passing the flag is harmless and
# redundant.
#
# **From scratch.** Rule 12 permits resuming across a geometry change, but the
# thing being changed here is the distribution the policy is trained on, and the
# checkpoint to resume from scores zero. There is nothing to preserve.

set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="${OUT:-artifacts/insert_stroke}"
mkdir -p "$OUT"
TASK="Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0"
RUN="${RUN:-grapple_insert_l0_seed70_v15stroke}"
EPOCHS="${EPOCHS:-2000}"

echo "[$(date +%H:%M:%S)] SMOKE the changed insert task"
"$PYTHON" scripts/train.py --headless --task "$TASK" --num_envs 32 --smoke \
    > "$OUT/smoke.log" 2>&1
if grep -qE "^Traceback" "$OUT/smoke.log"; then
  echo "[$(date +%H:%M:%S)] SMOKE FAILED:"; grep -E "Error|Exception" "$OUT/smoke.log" | tail -5; exit 1
fi
echo "[$(date +%H:%M:%S)] smoke clean"

# One short play on the untrained task, to catch a reset that is dead on step 1
# before hours go into it. This is the gate the four refuted resets were each
# run against, and it is cheaper than the alternative.
echo "[$(date +%H:%M:%S)] RESET GATE: is the module actually held at reset"
"$PYTHON" scripts/play.py --headless \
    --task "Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-Play-v0" \
    --checkpoint "${GATE_CKPT:-$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/$(ls -t "$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn" 2>/dev/null | grep '^last' | head -1)}" \
    --num_envs 64 --episodes 128 --curriculum_stage 0 --seed 1070 --grip_axis_metrics \
    --episode_metrics "$OUT/reset_gate.npz" --report "$OUT/reset_gate.json" \
    > "$OUT/reset_gate.log" 2>&1
echo "[$(date +%H:%M:%S)]   exit=$?"
OUT="$OUT" "$PYTHON" - <<'PY' 2>/dev/null || true
import json, numpy as np, os
rows = np.load(f"{os.environ['OUT']}/reset_gate.npz", allow_pickle=True)
fields = [str(name) for name in rows["fields"]]
index = {name: position for position, name in enumerate(fields)}
data = rows["rows"]
dead = float((data[:, index["control_steps"]] <= 2).mean())
print(f"    dead on the first two control steps: {dead * 100:.1f}%")
print(f"    median grip offset along the pin: {np.median(data[:, index['grip_offset_approach_axis_m']]) * 1000:.2f} mm")
PY

echo "=============================================================="
echo "[$(date +%H:%M:%S)] TRAIN insert from scratch, $EPOCHS epochs, run=$RUN"
echo "=============================================================="
# 512 environments, and the ceiling is PhysX rather than memory. The task runs
# `replicate_physics=False` because the form lock is a procedurally authored
# joint and replication copies only env 0's, so every environment authors its
# own joint and the scene build cost grows with the count. 1024 is kept as the
# first attempt for a task that ever runs replicated again; the fallback is what
# makes trying it free rather than a gamble.
for envs in ${NUM_ENVS:-512 256}; do
  echo "[$(date +%H:%M:%S)] attempting $envs environments"
  "$PYTHON" scripts/train.py --headless --task "$TASK" \
      --num_envs "$envs" --seed 70 --robustness_level 0 \
      --max_iterations "$EPOCHS" --run_name "$RUN" \
      ${RESUME_CKPT:+--checkpoint "$RESUME_CKPT"} \
      > "$OUT/train.log" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S)] train exit=$status at $envs environments"
  if [ $status -eq 0 ] && ls "$CKPT_ROOT/$RUN/nn"/last_*.pth >/dev/null 2>&1; then break; fi
  if grep -qiE "out of memory|CUDA error|cudaError|CUDA out" "$OUT/train.log"; then
    echo "[$(date +%H:%M:%S)] out of memory at $envs environments; falling back"
    continue
  fi
  break
done
ls -t "$CKPT_ROOT/$RUN/nn"/*.pth 2>/dev/null | head -3
