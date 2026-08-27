# Now

Verified repository state. Evidence status is mechanical in
[`evidence/MANIFEST.json`](../evidence/MANIFEST.json); this file is the concise
interpretation of that index. Detailed work is in [`NEXT_WORK.md`](NEXT_WORK.md)
and the frozen study design is in [`PAPER_PLAN.md`](PAPER_PLAN.md).

Last verified: 2026-08-26. Active branch:
`paper/serviceability-qualification`, based on `main` at `bccce6d`.

Everything is simulated. Nothing has run on hardware.

## Trust snapshot

| Item | Verified state |
| --- | --- |
| Evidence | 27 canonical, 8 retracted, 137 historical; quote only canonical |
| Source provenance | 10 reports carry runtime source bindings; 1 is recovered and 9 are lost |
| Recovered chain run | `robot_carried_full_chain_c11065.json` |
| Boundary decision | `not_qualified`; only entry attitude is supported in simulation |
| CI architecture | core package and tests do not require optional FastAPI imports |
| Checkpoints | reports contain hashes, but weights under `logs/` and `checkpoints/` are absent from a clone |
| Hardware claim | none |

The nine lost reports were produced from uncommitted source that cannot be
recovered from git. Their episode data remain evidence, but their exact code is
not reproducible. T0 is therefore narrowed, not erased: re-run any lost report
that a final claim needs. New reports must start from a clean commit.

## What runs

One continuous zero-gravity episode uses a UR10e to capture, extract, carry,
insert and release a compute module. There is no world constraint, teleport,
direct module pose write or hidden carrier.

| Phase | Executed controller |
| --- | --- |
| Grasp | PPO capture policy |
| Extract | PPO extraction policy |
| Transit | collision-checked differential IK with a robot-side form lock |
| Insert | guarded advance on the deployed estimate |
| Release | deterministic, after all seating conditions hold for 0.70 s |

The chain labels a phase learned only when policy actions actually step it. The
learned v24 insertion alternative is evaluated separately and does not control
the certified chain.

## Measured state

### Chain and skills

The guarded state-task chain scores **97.92%**: 94/96 episodes over three
held-out seeds, Wilson 95% **[92.7%, 99.4%]**. The source-bound single run is
recoverable; older pooled source bindings are not.

| Certificate | Pooled result | Interpretation |
| --- | ---: | --- |
| Grasp v7m130 | **85.69%** | misses the 95% skill gate |
| Extract v18pin | **87.75%** | misses the 95% skill gate |
| Insert v20chain | **0.00%**, 0/1,536 | preserved negative result |
| Insert v24rack, isolated | **36.77%**, 1,103/3,000 | not predictive of the chain handoff |
| Insert v24rack, in chain | **0.00%**, 0/96 | guarded remains selected |

The older insertion diagnosis remains useful: three objective arms terminated
at 84.26, 84.61 and 84.58 mrad against a **52.4 mrad** tolerance. Changing the
reward did not move the interface-limited attitude. Those runs have the source
provenance caveat above.

Perception is certified separately on 1,024 rendered frames. The 97.92% chain
rate uses simulator state through the deployed estimator path; RGB-D and the
chain have not been combined at qualification scale.

### Conditioned insertion

`run_workflow_demo.py` can now start the real chain driver at any of the nine
closed-form insertion reset stations and records the insertion handoff state.
`report_conditioned_insertion.py` requires paired guarded/v24 arms with the same
seed and initial-state SHA-256 and retains both arms.

One station-8, seed-1070 Isaac smoke pair completed from clean commit `269613b`.
Both controllers failed one episode; v24 ended closer axially. It is a diagnostic
artifact under `artifacts/`, not canonical evidence. The full 3,456 reset-station
episodes and 192 chain-handoff episodes remain to run.

### Serviceability boundary

[`serviceability_boundary_validation_v1.json`](../evidence/serviceability_boundary_validation_v1.json)
is the current fail-closed comparison:

| Dimension | State | Why |
| --- | --- | --- |
| Rack clearance | mismatch | 6 mm and 16 mm analytical exclusions do not show Wilson-separated loss |
| Module section | mismatch | 120×16 supports the prediction; 140×26 contradicts it at current sample size |
| Robot base offset | partial | x = −0.70 m is supported; the +10 mm y loss lacks an analytical bound |
| Entry attitude | supported in simulation | clearance sweep and throat intervention track `2c/L` within the frozen gate |
| Capture geometry | analytical only | visual bounding-volume checks pass; canonical contact/load evidence is absent |
| Load path | idealized and unpaired | rigid failure and compliant success do not share states/seeds or reaction telemetry |
| Base compliance | excluded | the authored spring is outside the load path because the robot root is fixed |

The validator changed no tolerance and discarded no arm. The present
configuration is **not qualified**.

## Claim limits

- The latch meshes are visual; module retention is a break-rated fixed joint
  while rigid and a bounded spring-damper while compliant.
- Historical simulator force probes include a 66.4 N derived axial requirement,
  but contact geometry and simulator forces are idealized and are not hardware
  load qualification.
- The robot root is welded to the world, so servicing-spacecraft reaction and
  compliant-base tolerance are not modeled.
- The robustness sweep has 16 episodes per point and ranks sensitivities; it is
  not a tolerance band.
- Every policy comes from one training seed.
- No current video shows the fully certified configuration.

## Reproduce and continue

```powershell
# CPU trust gate
.\.venv\Scripts\python.exe scripts/check_criterion_currency.py
.\.venv\Scripts\python.exe scripts/check_source_provenance.py --depth 200
.\.venv\Scripts\python.exe scripts/build_evidence_manifest.py --check
.\.venv\Scripts\python.exe scripts/build_script_index.py --check
.\.venv\Scripts\python.exe -m pytest -m "not isaac and not camera and not benchmark"

# Current negative boundary result; non-zero means not qualified
.\.venv\Scripts\python.exe scripts/validate_serviceability_boundary.py `
  --output <new-versioned-evidence-path>

# Exact GPU conditioned matrix, from a clean commit
& .\scripts\run_conditioned_insertion.ps1 -IncludeChainHandoffs `
  -OutputRoot artifacts\conditioned-insertion\v1 `
  -EvidencePath evidence\insertion_conditioned_controller_v1.json
```

Do not overwrite v1 evidence. Use a new version only after adding comparison
arms.

## Branches

| Branch | Status |
| --- | --- |
| `paper/serviceability-qualification` | Active qualification work; CI decoupling, provenance, conditioned insertion and boundary validation |
| `main` | Baseline at `bccce6d`; unchanged during this work |
| `industrial-relocation` | Preserved earlier work, 15 commits behind `main`; not identical to it |
| `keyed-interface` | Preserved losing keyed-interface exploration; do not delete |
| `origin/agent/zero-g-blade-swap` | Preserved historical eight-phase line; superseded |

The next gates are the complete conditioned matrix, qualification-count boundary
arms, paired contact/load-path experiments, a base-compliance path that can
actually deflect, clean-source reruns for quoted results, and the RGB-D chain.
