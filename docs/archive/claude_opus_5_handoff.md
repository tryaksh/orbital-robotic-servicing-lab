# Prompt for Claude Opus 5

Take ownership of `D:\6axis-space-robotics` and finish the showcase in one focused session.

## Non-negotiable goal

The six-axis robot must visibly and physically carry the compute module from bay 0 to bay 1. The current successful demo is not acceptable as the final presentation because `--base_rail_on_relocation` transfers the module to an invisible world-aligned D6 "payload shuttle", opens the gripper, and moves the module independently of the robot. The large upright ArUco board and heavy radiation-noise rendering also make the video look artificial.

Correct this without losing the working perception, compute service, evidence discipline, or previous measured baseline. You may redesign the grasp/latch/interface, trajectory, controller, policies, task phases, and camera presentation. Train or fine-tune on the GPU if it is the fastest credible route. Make decisions independently and ask only if truly blocked.

The owner explicitly rejects a hidden carrier as cheating. Do not optimize for another technical pass that avoids the central manipulation problem. A modern six-axis robot should be able to perform this swap using a hybrid strategy: trained control where contact uncertainty matters, deterministic collision-checked control for predictable free-space motion, and guarded control for insertion. If the present module/gripper interface cannot physically support that, prove the interface limitation with wrench/contact/relative-pose measurements and redesign the visible service interface. Report a design problem plainly rather than hiding it behind a constraint.

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

Start from the last real robot-held state immediately after learned extraction. First attempt the simplest industrial hybrid solution:

1. learned capture and extraction;
2. deterministic IK/joint-space retreat and collision-checked bay-to-bay motion while the robot keeps gripping the module;
3. visual/guarded alignment and robot-driven insertion;
4. release only after settled seating is verified.

Do not default to end-to-end RL merely because RL exists in the repository. Train or fine-tune capture, retention, or insertion only where measured failures show deterministic control is insufficient.

Keep a real load path from robot gripper to module throughout extraction, transit, alignment, and insertion. First test ordinary contact retention and record the tool-to-module transform, contact forces, grip torque, inertial motion, and failure instant. Distinguish arm payload/reach limits from gripper/interface wrench limits. The earlier tapered-pin measurement had little force margin, so a loss may be a service-interface problem rather than a policy problem.

If passive friction contact cannot constrain the module robustly in all six degrees of freedom, design a visible, plausible form-locking service interface: for example a keyed grapple feature, collar/detent, closing hook, or explicit tool-changer latch. Model its geometry, collision, engagement condition, rated load, and release. A fixed/compliant joint is allowed only as the physics implementation of that visible mechanical latch, between the robot tool and module, after verified engagement. It must not be an invisible convenience, must not attach to the world, and must be disclosed in the report and video. Never write the module pose directly.

Use a collision-checked robot trajectory or deterministic joint-space waypoints to retreat, cross bays, align, and insert. Reuse learned capture/extraction. Use the learned insert policy only if its reset distribution can be reached reliably; otherwise implement an honest guarded robot-held insertion controller or retrain/fine-tune it. The final video must show the robot, gripper, and module moving together, with bounded tool-to-module relative pose until the module is seated.

Keep real perception, but replace the billboard-like tag with a flush, realistically sized service fiducial or a small multi-tag layout that remains detectable. Produce a clean showcase render with sensible lighting and little/no presentation noise; retain noisy randomized frames for perception qualification. Add a visible pose/phase overlay if practical.

## Acceptance gates

Do not call the project complete until all are true:

1. The robot remains visibly attached to and carries the module through extraction, transit, alignment, and insertion.
2. No world-aligned payload stage, direct root-pose write, teleport, or hidden kinematic carrier is active in the showcased run.
3. Robot motion commands, not module motion commands, produce the transfer. Tool-to-module position/orientation stays bounded and is recorded through transit.
4. The robot releases only after the module passes depth, lateral, attitude, velocity, and 0.70-second settling checks.
5. RGB-D perception and occupancy planning remain active and fail closed; simulator truth is diagnostic/supervisory only, not a policy observation.
6. Run at least one full RGB-D end-to-end success and a meaningful multi-episode state batch. Preserve failures honestly.
7. The live compute-service preset launches this robot-carried workflow and saves events, report, trace, hashes, and a clear video.
8. Tests, Ruff, evidence currency, and source-hash readiness pass. Commit the final result.

If these gates cannot be met in one session, stop with the best measured diagnosis and a concrete visible interface redesign. Do not fall back to the shuttle and do not publish a floating-module video as success.

Spend the session on the physical workflow and video first. Do not expand portfolio prose, cloud infrastructure, or unrelated experiments. At handoff, state exactly what is learned, scripted, perceived, privileged, physically constrained, statistically supported, and still unvalidated on hardware.
