# Independent Project Assessment

**Decision date:** 2026-08-19
**Decision:** Pursue through a product and scope pivot.

This document records the repository state found at the start of the independent
audit. The compute service, leak-free perception path, and overview-camera work
described in the implementation decision were built afterward; the current run
instructions and validation state live in the README and compute-service demo
runbook.

## Executive verdict

This repository is worth pursuing. It contains genuine robotics engineering: contact-rich Isaac Lab environments, learned control policies, a real image-to-pose model, a reset-safe evaluation pipeline, and a currently working module-removal chain. It is not, today, an end-to-end autonomous orbital compute-module replacement system.

The strongest product is a **simulation-backed robotic serviceability qualification system**:

> Before hardware is built or expensive policy training begins, determine whether a module, service interface, rack, sensor layout, and robot workcell can be serviced—and produce measured reasons when they cannot.

Orbital compute servicing remains the demanding reference scenario. The delivered claim should be design qualification and simulation evidence, not flight readiness.

## What was verified directly

| Capability | Observed state | Honest conclusion |
| --- | --- | --- |
| Python contract/unit suite | 115/115 pass | Useful protection around geometry, metrics, checkpoints, and configuration contracts. It is not a substitute for Isaac integration runs. |
| Isaac Sim / Isaac Lab | Runs on the local RTX 5070 Ti | The simulation stack is usable and GPU work can continue. |
| Current removal chain | 573/576, 99.48% in tracked evidence; an independent single-chain run also completed and remained settled | This is the strongest execution proof in the repository. |
| Current installation chain | 1/576, 0.17%; an independent run captured in 2.13 s, then exhausted the 30 s insertion budget 152 mm short | Installation is broken on the current workcell. |
| Current two-bay insertion | 322/3066, 10.50% | It cannot support a relocation claim. |
| Current relocation | 0/64 | The arm reaches the planned path, but the contact grasp loses the module. |
| Perception model | A real CNN consumes rendered RGB and predicts module pose | It is genuine perception, not an oracle disguised as a network. |
| Stored two-bay pose-head evaluation | 2.81 mm mean and 6.47 mm p95 position error on a same-collection held-out tail | Promising simulation regression, but the split is optimistic and the model predates the current workcell. |
| Stored camera/oracle/blind workflow comparison | 84.90% / 88.72% / 34.03% | The image contains useful signal. These runs predate the workcell change and all fail the repository's 95% gate. |
| HTTP/RPC compute service | None | There is no service API, job lifecycle, worker isolation, artifact endpoint, or deployable demo entry point. |
| Public reproducibility | Incomplete | Checkpoints, datasets, videos, and logs are ignored; a clean clone cannot reproduce the learned demo. |

## The most important audit finding

The existing “vision workflow” is not end-to-end vision.

- Only the capture policy's `grip_error` is replaced with a camera estimate.
- Extraction still receives simulator-derived module travel, velocity, and grip geometry.
- Insertion still receives simulator-derived module-to-goal and grip geometry.
- The two-bay occupancy output is evaluated offline but never selects a source or destination bay.
- The occupancy dataset contains bay-one and bay-two examples but no genuine neither-bay transit examples, despite comments describing that class.
- The existing narrow camera does not keep the slot mouth in frame, so it cannot support visual insertion throughout the operational envelope.

Until these leaks are removed, the correct phrase is **vision-assisted capture followed by privileged-state manipulation**.

## Why the pivot is industrially relevant

The repository's negative results point at the real bottleneck: service interfaces and workcells are often not designed for robots.

- The workcell relocation removed one attitude/reach barrier and exposed a separate retention and insertion failure. Position reach alone was not enough.
- Lead-in geometry materially changes insertion performance.
- A parallel-jaw friction grip can qualify at capture and still fail when the unconstrained module is transported.
- A nominal interface load and the simulated insertion reaction leave inadequate margin unless the interface is form-locking or actively latched.

This problem exists beyond the reference space scenario. SoftBank publicly describes a robot-friendly, cableless rack with blind-mate power, networking, and liquid cooling, plus a floating guide structure that tolerates robot alignment error. ESA is likewise deploying and standardizing capture/navigation interfaces for future servicing. Those are direct market signals for **designing hardware around robotic serviceability**, not merely training a smarter arm around hostile hardware.

Primary references:

- [SoftBank robot-friendly server rack](https://www.softbank.jp/en/sbnews/entry/20251215_01)
- [NASA 2025 ISAM State of Play](https://ntrs.nasa.gov/citations/20250008988)
- [ESA MICE capture interface and navigation aids](https://www.esa.int/ESA_Multimedia/Images/2025/05/MICE_and_navigation_aids_on_LUR-1)
- [ESA Design-for-Removal interface requirements](https://technology.esa.int/upload/media/D4R---IRD-for-LEO-and-GEO-missions-v3-0.pdf)

## Product thesis

**User:** robotics, hardware, or reliability engineer responsible for an unattended modular system.
**Bottleneck:** a module, interface, or workcell reaches integration before anyone proves that a robot can see, capture, load, move, align, and seat it within safety margins.
**Product:** submit a service design and scenario; receive preflight feasibility checks, an observable simulation job, perception/control telemetry, failure attribution, and a reproducible qualification report.
**Initial reference workflow:** microgravity compute-module service using a six-axis arm and a deliberately serviceable grapple/rail interface.

## Delivery gates

The showcase is complete only when one command or API job can:

1. validate the requested workcell and interface configuration;
2. acquire overview and close-range camera observations;
3. estimate bay occupancy, module pose, visibility, and confidence;
4. choose and execute a servicing plan;
5. use no simulator-derived module state in deployment observations;
6. expose phase, pose error, force/torque proxy, safety state, and outcome as telemetry;
7. retain both successful and failed outcomes as replayable artifacts;
8. produce a provenance-rich JSON qualification report;
9. run in replay mode without Isaac Sim or private model files;
10. label simulation, abstraction, and unvalidated hardware boundaries plainly.

The first releasable proof may use the reliable visual removal workflow while installation is repaired. A “full swap” claim requires continuous capture, extraction, transfer, insertion, release, and postcondition verification on the same module.

## Explicitly out of scope for the current claim

- flight qualification or space-hardware validation;
- a free-flying servicing spacecraft;
- electrical, optical, coolant, or power connector mating;
- fault isolation and live workload migration;
- production safety certification;
- a YC-ready business with validated customers.

These are roadmap items, not implications of a simulation success rate.

## Implementation decision

Proceed with four coordinated changes:

1. add a serialized compute-service job API with live events and artifact retrieval;
2. replace the partial camera path with one shared, temporally filtered module-state estimate used by every manipulation phase;
3. redesign camera coverage and collect data across complete servicing trajectories;
4. ship an honest qualification demo that leads with the passing removal proof and refuses to promote installation or relocation until their gates pass.

This preserves the repository's strongest work while making the finished result clearer, harder to fake, and materially more relevant to industrial robotics.
