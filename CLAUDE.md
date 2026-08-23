# Claude Session Instructions

Read [`docs/robot_carried_handoff.md`](docs/robot_carried_handoff.md) before
making changes. It states what is proven, what is scripted, and what is open.
[`docs/claude_opus_5_handoff.md`](docs/claude_opus_5_handoff.md) is the task that
produced it and is kept for the reasoning it records.

## Current truth

- Branch: `industrial-relocation`.
- RGB-D perception, occupancy planning, the local compute service, telemetry,
  provenance, and artifact export work.
- **The robot carries the module.** A parallel-jaw grip on the passive pin loses
  it entirely across a bay-to-bay flight — 0 of 16 environments retained the
  planned tool-to-module transform, median drift 808 mm and 3.14 rad — and a
  visible robot-side form lock holds it to 2.6 mm across 32 environments, 11 of
  which held the transform inside tolerance for the whole flight.
- **The seating is not closed**, and the reason is *not* the carry-versus-mate
  conflict this file used to state. Four faults in the chain were found and
  fixed; three published conclusions were retracted. The corrected chain drives
  the module 362 mm into the destination channel and stops 163 mm short, with
  the guarded advance holding a commanded depth a full mating stroke ahead for
  875 of 900 steps and the guard never firing. The blocker is the delivered
  attitude — 47 to 67 mrad, split 13.8 mrad pitch and 15.1 mrad yaw — which is
  upstream of anything the mating interface can do. Section 4 of the handoff has
  the measurements and the redesign.
- **Before believing a mating measurement, check the controller.** The guarded
  advance's axial target used to be rebuilt each step from the module it was
  pushing, which reads as a bounded lead and behaves as a deadlock. Every
  stiffness, force cap and clearance in the old grid was measured through it.
- One full RGB-D chain runs end to end on seed 6070 with randomization on:
  `evidence/robot_carried_rgbd_seed6070.json`. It meets every seating condition
  except axial depth. Seed 4070 does not get through capture on the vision task.
- The clearance sweep is the result to quote:
  `evidence/robot_carried_seating_sweep.json`. Every extra millimetre of
  per-side channel relief buys about 1.2 mm of seating travel and costs about
  3.5 mrad of squareness, so the two requirements cross 15 mm into a 163 mm
  travel. There is no channel width for this workcell.
- The world-mounted payload stage behind `--base_rail_on_relocation` is retained
  **only** as a labelled historical baseline. It is not reachable from the live
  preset, and `tests/test_robot_carried_contract.py` keeps it out.
- The two-bay insert policy certifies at 10.5% on this workcell, 0.00% in the
  first bay. Insertion here is scripted and guarded and is labelled so.

## Required engineering approach

The industrial hybrid, which is what is built:

1. trained capture and extraction;
2. deterministic collision-checked robot motion while retaining the module, on
   a visible robot-side form lock;
3. guarded robot-driven alignment and insertion, advancing only while the
   deployed estimator says the module is inside its envelope;
4. release of the lock's rigidity where the rack takes over, and of the hand
   only after settled seating is verified.

Do not use a world constraint, teleport, direct module pose write, or hidden
carrier. The form lock's load path is a break-rated PhysX fixed joint between
`wrist_3_link` and the module while rigid, and a bounded spring-damper on the
same pair while compliant. Its hardware is authored on the wrist and its
clearances are derived by `scripts/check_service_latch_clearance.py`. That
simplification is disclosed in every report and in the specification.

Train or fine-tune only where measurements justify it. A policy cannot make a
3 mm swing fit through a 0.5 mm gap; check the geometry before spending the GPU.

## Preserve

- `evidence/fiducial_rgbd_service_plate.json`
- `evidence/full_chain_state_16_report.json`
- `evidence/full_chain_rgbd_service_seed4070.json`
- `evidence/service_latch_clearance.json`
- `docs/service_interface_spec.md`
- the service API/dashboard and its security boundaries
- checkpoint and source SHA-256 provenance
- failed results as labelled historical evidence

## Main files

- `scripts/run_workflow_demo.py`
- `scripts/run_robot_carried.sh`
- `scripts/check_service_latch_clearance.py`
- `src/zero_g_blade_swap/service_latch.py`
- `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py`
- `src/zero_g_blade_swap/tasks/blade_swap/assets.py`
- `src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py`
- `src/zero_g_blade_swap/service/presets.py`

## Completion rule

The robot must visibly carry the module from source to destination, the report
must show bounded tool-to-module pose throughout transit, RGB-D perception must
remain active, the module must settle in the destination for 0.70 seconds, and
the compute service must save a clear video and hashed artifacts. If part of
this cannot be completed, report the measured blocker; never fall back to the
hidden payload stage and call it success.
