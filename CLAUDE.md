# Claude Session Instructions

Read and execute [`docs/claude_opus_5_handoff.md`](docs/claude_opus_5_handoff.md)
before making changes. It is the authoritative task for this branch.

## Current truth

- Branch: `industrial-relocation`.
- RGB-D perception, occupancy planning, the local compute service, telemetry,
  provenance, and artifact export work.
- The latest complete chain is only a baseline. It uses
  `--base_rail_on_relocation`, which transfers the module to a hidden
  world-mounted D6 payload stage after extraction. It is not a robot-carried
  result and must not be the final showcase.
- The oversized upright fiducial and heavy presentation noise are also not
  acceptable for the final video.
- Preserve the baseline evidence, but make the live preset and final video use
  the six-axis robot to carry the module.

## Required engineering approach

Prefer the industrial hybrid:

1. trained capture and extraction;
2. deterministic collision-checked robot motion while retaining the module;
3. visual/guarded robot-driven alignment and insertion;
4. release only after settled seating is verified.

Do not use a world constraint, teleport, direct module pose write, or hidden
carrier. If the present parallel-jaw interface cannot retain the module, prove
that with contact wrench and tool-to-module measurements, then implement a
visible, physically justified form-locking service interface. A simulated joint
is acceptable only as the implementation of that visible robot-to-module latch.

Train or fine-tune only where measurements justify it. Do not replace predictable
free-space motion with end-to-end RL without evidence that it helps.

## Preserve

- `evidence/fiducial_rgbd_service_plate.json`
- `evidence/full_chain_state_16_report.json`
- `evidence/full_chain_rgbd_service_seed4070.json`
- `docs/service_interface_spec.md`
- the service API/dashboard and its security boundaries
- checkpoint and source SHA-256 provenance
- failed results as labelled historical evidence

## Main files

- `scripts/run_workflow_demo.py`
- `scripts/plan_relocation_joint_path.py`
- `configs/ur10e_relocation_rrt.yaml`
- `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py`
- `src/zero_g_blade_swap/tasks/blade_swap/mdp/perception.py`
- `src/zero_g_blade_swap/tasks/blade_swap/assets.py`
- `src/zero_g_blade_swap/service/presets.py`

## Completion rule

The robot must visibly carry the module from source to destination, the report
must show bounded tool-to-module pose throughout transit, RGB-D perception must
remain active, the module must settle in the destination for 0.70 seconds, and
the compute service must save a clear video and hashed artifacts. If this cannot
be completed, report the measured design blocker; never fall back to the hidden
payload stage and call it success.
