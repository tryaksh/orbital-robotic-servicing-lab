# Prompt for Claude Opus 5

Take ownership of `D:\6axis-space-robotics` and finish the showcase in one focused session.

## Non-negotiable goal

The six-axis robot must visibly and physically carry the compute module from bay 0 to bay 1. The current successful demo is not acceptable as the final presentation because `--base_rail_on_relocation` transfers the module to an invisible world-aligned D6 "payload shuttle", opens the gripper, and moves the module independently of the robot. The large upright ArUco board and heavy radiation-noise rendering also make the video look artificial.

Correct this without losing the working perception, compute service, evidence discipline, or previous measured baseline. You may redesign the grasp/latch/interface, trajectory, controller, policies, task phases, and camera presentation. Train or fine-tune on the GPU if it is the fastest credible route. Make decisions independently and ask only if truly blocked.

## Preserve before changing

- Current branch: `industrial-relocation`.
- The current baseline is committed; keep the shuttle implementation available only as a clearly labelled historical/experimental baseline, not the default showcase.
- RGB-D perception passed on 1,024 rendered frames: `evidence/fiducial_rgbd_service_plate.json`.
- The shuttle baseline passed 16/16 state runs and one RGB-D chain: `evidence/full_chain_state_16_report.json` and `evidence/full_chain_rgbd_service_seed4070.json`. Do not relabel these as robot-carried results.
- The local API/dashboard/job lifecycle and provenance system are useful and should be reused, not rewritten.

## Where the unwanted behavior lives

- `scripts/run_workflow_demo.py`: `--base_rail_on_relocation`, `_engage_payload_stage`, `payload_stage_*`, shuttle transit/insertion branches.
- `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py`: `configure_base_rail()` authors the world-aligned D6 payload stage and modifies the destination mouth.
- `src/zero_g_blade_swap/tasks/blade_swap/mdp/perception.py`: contains shuttle-specific estimator state/propagation.
- `src/zero_g_blade_swap/service/presets.py`: the live preset currently enables the shuttle.
- `src/zero_g_blade_swap/tasks/blade_swap/assets.py`: authors the oversized fiducial service plate.

## Preferred engineering direction

Start from the last real robot-held state immediately after learned extraction. Keep a real mechanical load path from gripper to module throughout transit and alignment. A simulated latch/fixed or compliant constraint is acceptable only if it represents an explicit form-locking gripper mechanism, attaches between the gripper/tool and module after a measured capture condition, carries forces through the robot, and releases only after seating. Never write the module pose directly and never constrain it to the world.

Use a collision-checked robot trajectory or deterministic joint-space waypoints to retreat, cross bays, align, and insert. Reuse learned capture/extraction. Use the learned insert policy only if its reset distribution can be reached reliably; otherwise implement an honest guarded robot-held insertion controller or retrain/fine-tune it. The final video must show the gripper and module moving together, with bounded tool-to-module relative pose until the module is seated.

Keep real perception, but replace the billboard-like tag with a flush, realistically sized service fiducial or a small multi-tag layout that remains detectable. Produce a clean showcase render with sensible lighting and little/no presentation noise; retain noisy randomized frames for perception qualification. Add a visible pose/phase overlay if practical.

## Acceptance gates

Do not call the project complete until all are true:

1. The robot remains visibly attached to and carries the module through extraction, transit, alignment, and insertion.
2. No world-aligned payload stage, direct root-pose write, teleport, or hidden kinematic carrier is active in the showcased run.
3. Tool-to-module position/orientation stays bounded and is recorded in the report through transit.
4. The robot releases only after the module passes depth, lateral, attitude, velocity, and 0.70-second settling checks.
5. RGB-D perception and occupancy planning remain active and fail closed; simulator truth is diagnostic/supervisory only, not a policy observation.
6. Run at least one full RGB-D end-to-end success and a meaningful multi-episode state batch. Preserve failures honestly.
7. The live compute-service preset launches this robot-carried workflow and saves events, report, trace, hashes, and a clear video.
8. Tests, Ruff, evidence currency, and source-hash readiness pass. Commit the final result.

Spend the session on the physical workflow and video first. Do not expand portfolio prose, cloud infrastructure, or unrelated experiments. At handoff, state exactly what is learned, scripted, perceived, privileged, physically constrained, statistically supported, and still unvalidated on hardware.
