# Agent instructions

**Read [`docs/STATUS.md`](docs/STATUS.md) first.** It is the single source of
current state: what runs, what the numbers are, which checkpoints the chain uses,
what was found and settled, what is still open, and how to reproduce any of it.
Everything below is the working rules, not the state.

Branch: `industrial-relocation`.

## Required engineering approach

The industrial hybrid, which is what is built:

1. trained capture and extraction;
2. deterministic collision-checked robot motion while retaining the module, on a
   visible robot-side form lock;
3. guarded robot-driven alignment and insertion, advancing only while the
   deployed estimator says the module is inside its envelope;
4. release of the lock's rigidity where the rack takes over, and of the hand only
   after settled seating is verified.

Do not use a world constraint, teleport, direct module pose write, or hidden
carrier. The form lock's load path is a break-rated PhysX fixed joint between
`wrist_3_link` and the module while rigid, and a bounded spring-damper on the
same pair while compliant. Its hardware is authored on the wrist and its
clearances are derived by `scripts/check_service_latch_clearance.py`. That
simplification is disclosed in every report and in the specification.

## Rules

1. **Label on the controller that ran.** A phase may not be called learned unless
   a policy produced the actions. Never key that label on a configuration flag.
2. **Derive geometry, do not tune it.** Every geometric requirement is computed by
   a check that runs without a simulator, and that check validates itself against
   the simulator's own recorded configurations before it reports.
3. **Change one thing at a time and keep the losing arm.** A criterion change and
   a policy change must never be quoted as one number. `play.py` carries
   `--legacy_grip_ball_m` and `--legacy_unbounded_reset` for exactly this.
4. **Never widen a tolerance to pass a gate.** If a criterion is wrong, replace it
   with one derived from the parts, and re-run the old checkpoint under both.
5. **Check the geometry before spending the GPU.** A policy cannot make a 3 mm
   swing fit through a 0.5 mm gap. Extract gained 13 points from task corrections
   and 0 from 2,000 epochs; that ordering is the norm here, not the exception.
6. **Keep failed results.** Superseded and failed runs stay in `evidence/`,
   labelled. Claims that turn out to be wrong go in `evidence/RETRACTED.md`.
7. **Batch long GPU runs and time-box them** so certification always has clock
   left. A 1024-environment PPO run fits in 12 GB alongside a small evaluation
   process; two full training runs do not.

## Completion rule

The robot must visibly carry the module from source to destination, the report
must show bounded tool-to-module pose throughout transit, RGB-D perception must
remain active, the module must settle in the destination for 0.70 seconds, and
the compute service must save a clear video and hashed artifacts. If part of this
cannot be completed, report the measured blocker; never fall back to the hidden
payload stage and call it success.

This is met at the pooled rate. The same rule applied to the individual *skills*
is not — see `docs/STATUS.md` §2 and §5.

## Preserve

- `evidence/workflow_robot_carried_m130pin_guarded_certification.json` — the rate
- `evidence/workflow_robot_carried_m130_guarded_certification.json` and
  `evidence/workflow_robot_carried_relocate_certification.json` — the two befores
- `evidence/robot_carried_full_chain_pin.json` — one run of it, end to end
- `evidence/extract_attribution.json` — one change a row, on one checkpoint
- `evidence/grapple_extract_v17m130_on_pin_criterion_certification.json` and
  `evidence/grapple_grasp_v7m130_on_derived_rack_certification.json` — the
  controls that make the comparison legitimate rather than a re-baselining
- `evidence/workcell_geometry_check.json` — where the arm stands, what the
  channel admits, which module sections the rack accepts
- `evidence/chain_robustness_sweep.json` — what breaks the chain first
- `evidence/insert_reset_bank.json` — the stations an insertion may start from
- `evidence/robot_carried_rigid_mating_refuted.json` — why the lock has to soften
- `evidence/fiducial_rgbd_service_plate.json`,
  `evidence/full_chain_state_16_report.json`,
  `evidence/full_chain_rgbd_service_seed4070.json`,
  `evidence/service_latch_clearance.json`
- `docs/service_interface_spec.md`, `docs/STATUS.md`, `docs/archive/`
- the service API/dashboard and its security boundaries
- checkpoint and source SHA-256 provenance

## Main files

| Path | What it is |
| --- | --- |
| `scripts/run_workflow_demo.py` | The chain driver: phases, controllers, reports |
| `scripts/run_robot_carried.sh` | Every way of running the chain, one stage per question |
| `scripts/check_workcell_geometry.py` | Where the arm stands, what the channel admits, the lateral-clearance window, the module-section envelope. No simulator |
| `scripts/check_service_latch_clearance.py` | The form lock's clearances, derived from the measured gripper envelope |
| `scripts/solve_insert_reset_bank.py` | Paired arm and module poses along the seating stroke, solved in closed form and gated |
| `scripts/certify_grapple_skills.sh` | One skill, three stages, three held-out seeds, pooled with a gate |
| `scripts/sweep_chain_robustness.sh`, `scripts/report_chain_robustness.py` | One variable at a time around the certified point |
| `scripts/report_extract_attribution.py` | The ladder that separates criterion, rack, reset and policy |
| `src/zero_g_blade_swap/arm_kinematics.py` | One UR10e chain: NumPy for the checks, batched Torch for the transit legs and the reset pairing |
| `src/zero_g_blade_swap/grapple_geometry.py` | The pin, and what counts as holding it |
| `src/zero_g_blade_swap/service_latch.py` | The form lock's geometry and release interlock |
| `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py` | The three skill tasks |
| `src/zero_g_blade_swap/tasks/blade_swap/assets.py` | Rack, module, pin, rails, lead-ins |
| `src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py` | Every predicate and reward the skills use |
| `src/zero_g_blade_swap/tasks/blade_swap/insert_reset_bank.py` | Generated; do not hand-edit |
| `src/zero_g_blade_swap/service/presets.py` | What the live service actually runs |
