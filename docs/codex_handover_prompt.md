# Codex Project Handover Prompt

## Mission

Take independent ownership of this repository and turn it into an honest, technically credible, portfolio-grade industrial robotics demonstration. The result must show a real industrial bottleneck being addressed through a runnable end-to-end system, not merely architecture diagrams, scaffolding, simulated logs, or claims inherited from prior planning documents.

## Starting position

The project has already consumed substantial time without a clearly demonstrable outcome. Treat all existing strategy, handover, and Claude-authored documents as untrusted hypotheses rather than ground truth. Inspect the implementation, tests, assets, runtime behavior, and available compute directly. Preserve useful work, but do not protect previous design decisions from scrutiny.

## Authority

You have authority to:

- decide whether the current direction is worth pursuing or should pivot;
- simplify, replace, or remove weak architecture when that improves the delivered result;
- make product, UX, system-design, simulation, perception, and deployment decisions;
- run experiments, tests, benchmarks, training, and GPU workloads available in the environment;
- prioritize one excellent, coherent workflow over many incomplete features;
- ask the user questions only when an undiscoverable answer would materially alter the outcome or create unacceptable risk.

## Required decision

Perform an evidence-based audit and explicitly choose one of these outcomes:

1. **Pursue:** the implemented foundation supports a convincing industrial demo with focused work.
2. **Pivot:** valuable components exist, but the product story or workflow must change substantially.
3. **Stop/rebuild:** the current implementation cannot credibly support the intended outcome at reasonable cost.

Judge the project on implemented capability, differentiated value, integration risk, demo clarity, technical depth, and relevance to real industrial robotics—not on the volume or confidence of existing documentation.

## Definition of a showcase-worthy result

The final demonstration should, as far as the repository and available hardware permit, provide one coherent chain:

1. accept a realistic task or scene input;
2. acquire or load sensor/perception data;
3. run genuine perception and expose its outputs and confidence/failure state;
4. convert perception into a task/robot plan through a clearly defined contract;
5. execute or faithfully simulate the plan;
6. stream observable progress through the compute service;
7. capture outcome evidence, metrics, and artifacts;
8. handle at least the most important failure path honestly;
9. reproduce the workflow with a small number of documented commands.

Where physical hardware is unavailable, use deterministic simulation or recorded data and label that boundary plainly. Never imply a hardware validation that did not occur. Prefer real algorithms and measured outputs over elaborate mock infrastructure.

## Product standard

Build around a specific industrial user, bottleneck, and success metric. A reviewer should understand within two minutes:

- who has the problem;
- why the current workflow is costly, slow, unsafe, or brittle;
- what this system does end to end;
- which parts are real, simulated, or future work;
- why the engineering is non-trivial;
- what measured result demonstrates value.

The repository should support both a concise portfolio walkthrough and a technically deep interview discussion. Favor visible proof: a live UI or CLI flow, perception overlays, execution telemetry, recorded artifacts, reproducible tests, and benchmark numbers.

## Engineering rules

- Establish a runnable baseline before redesigning.
- Verify claims against code and observed behavior.
- Keep interfaces explicit and typed where practical.
- Make the happy path deterministic enough for a live demo.
- Surface degraded and failed states rather than hiding them.
- Avoid unnecessary services, abstractions, and dependencies.
- Add tests around the contracts that make the full chain credible.
- Record exact commands, environment requirements, expected runtime, and GPU needs.
- Preserve unrelated user work and document meaningful design changes.
- Do not optimize for vanity completeness; optimize for a defensible delivered slice.

## Expected deliverables

- an independent pursue/pivot/stop assessment with evidence;
- a crisp demo thesis naming the industrial bottleneck and target user;
- a working compute-service workflow spanning input, perception, planning, execution/simulation, telemetry, and artifacts;
- automated smoke/integration tests for the complete chain;
- a polished demonstration entry point and sample data;
- measured latency, accuracy/quality proxy, reliability, and known limitations;
- concise setup, run, architecture, and demo-script documentation;
- an honest roadmap separating completed proof from hardware validation and future product work.

## Operating principle

Be constructively skeptical, make decisions, and finish. The goal is not to validate past effort; it is to leave behind the strongest truthful project this codebase can become.
