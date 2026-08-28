# Now

Verified repository state. Evidence status is mechanical in
[`evidence/MANIFEST.json`](../evidence/MANIFEST.json); this file is the concise
interpretation of that index. Detailed work is in [`NEXT_WORK.md`](NEXT_WORK.md)
and the frozen study design is in [`PAPER_PLAN.md`](PAPER_PLAN.md).

Last verified: 2026-08-27. Active branch:
`paper/serviceability-qualification`, based on `main` at `bccce6d`.

Everything is simulated. Nothing has run on hardware.

## Trust snapshot

| Item | Verified state |
| --- | --- |
| Evidence | 30 canonical, 9 retracted, 140 historical; quote only canonical |
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
| Grasp v7m130, current rack | **86.90%**, 7,829/9,009 | unchanged checkpoint; misses the 95% gate |
| Extract v18pin, current rack | **87.64%**, 7,891/9,004 | unchanged checkpoint; misses the 95% gate |
| Insert v20chain | **0.00%**, 0/1,536 | preserved negative result |
| Insert v24rack, isolated | **36.77%**, 1,103/3,000 | not predictive of the chain handoff |
| Insert v24rack, in chain | **0.00%**, 0/96 | guarded remains selected |
| Insert v25 handoff-only probe | **0.00%**, 0/64 | targeted reset improved errors but produced no success |

The older insertion diagnosis remains useful: three objective arms terminated
at 84.26, 84.61 and 84.58 mrad against a **52.4 mrad** tolerance. Changing the
reward did not move the interface-limited attitude. Those runs have the source
provenance caveat above.

The current-rack skill reruns use the same checkpoint hashes as the earlier
85.69% grasp and 87.75% extraction certificates. Their Wilson intervals overlap
the earlier results. Narrowing the rack therefore did not repair either skill;
the predicted extraction benefit from the smaller channel corner is refuted.

Perception is certified separately on 1,024 rendered frames. The 97.92% chain
rate uses simulator state through the deployed estimator path; RGB-D and the
chain have not been combined at qualification scale.

### Conditioned insertion

[`insertion_conditioned_controller_v3.json`](../evidence/insertion_conditioned_controller_v3.json)
pairs guarded and v24 insertion on identical seeds, initial-state hashes,
budgets and fixed-to-compliant load paths. All 60 arms ran from clean commit
`daa53d6`; aggregation is bound to clean commit `e6d841d`.

| Start condition | Guarded | v24 |
| --- | ---: | ---: |
| Reset stations 0–3 | 0/768 | 0/768 |
| Reset station 4 | 0/192 | 75/192 |
| Reset station 5 | 0/192 | 165/192 |
| Reset station 6 | 0/192 | 174/192 |
| Reset station 7 | 0/192 | 192/192 |
| Reset station 8 | 0/192 | 180/192 |
| Real chain handoff | **94/96** | **0/96** |

Across reset stations alone, v24 is 786/1,728 (**45.49%**) and guarded is
0/1,728. The isolated skill has learned the late stroke, not the state its caller
provides. The predeclared every-condition rule therefore keeps guarded insertion.
The earlier combined v1 report is retracted because its reset arms disabled the
task's load path; its valid handoff-only rows remain preserved separately.

The first targeted correction is also preserved, not promoted:
[grapple_insert_v25handoff_probe.json](../evidence/grapple_insert_v25handoff_probe.json).
Resuming v24 for 400 epochs only at station 0 reduced median axial error on the
identical seed from 247.6 to 230.5 mm, lateral error from 10.6 to 8.9 mm and
orientation error from 110.8 to 104.0 mrad, but remained 0/64. It was also 0/64
on its own noisy training task. The next run therefore starts from v24's
successful stations 6--8 and moves the frontier toward station 0 only after
held-out success at the active frontier; it is not a blind epoch extension.

### Serviceability boundary

[`serviceability_boundary_validation_v2.json`](../evidence/serviceability_boundary_validation_v2.json)
is the current fail-closed comparison:

| Dimension | State | Why |
| --- | --- | --- |
| Rack clearance | mismatch | 6 mm and 16 mm analytical exclusions do not show Wilson-separated loss |
| Module section | mismatch | 120×16 supports the prediction; 140×26 contradicts it at current sample size |
| Robot base offset | mismatch | +10 mm rail-stop error is kinematically feasible but causes a Wilson-separated simulation loss |
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
  -OutputRoot artifacts\conditioned-insertion\reproduction `
  -EvidencePath evidence\insertion_conditioned_controller_v4.json
```

Never overwrite evidence; use a new versioned filename for another run.

## Branches

| Branch | Status |
| --- | --- |
| `paper/serviceability-qualification` | Active qualification work; CI decoupling, provenance, conditioned insertion and boundary validation |
| `main` | Baseline at `bccce6d`; unchanged during this work |
| `industrial-relocation` | Preserved earlier work, 15 commits behind `main`; not identical to it |
| `keyed-interface` | Preserved losing keyed-interface exploration; do not delete |
| `origin/agent/zero-g-blade-swap` | Preserved historical eight-phase line; superseded |

The next gate is the success-gated reverse-curriculum insertion policy, tested
across the same station and real-handoff matrix. After insertion: the RGB-D
chain, qualification-count boundary arms, paired contact/load-path experiments,
a base-compliance path that can actually deflect, and clean-source reruns for
older quoted results.
