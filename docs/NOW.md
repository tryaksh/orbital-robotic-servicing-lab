# Now

Verified repository state. Evidence status is mechanical in
[`evidence/MANIFEST.json`](../evidence/MANIFEST.json); bounded tasks are in
[`NEXT_WORK.md`](NEXT_WORK.md). Last verified: 2026-08-31 on
`paper/serviceability-qualification`, based on `main` at `bccce6d`.

Everything is simulated. Nothing has run on hardware.

## Trust snapshot

| Item | Verified state |
| --- | --- |
| Evidence | 32 canonical, 11 retracted, 140 historical; quote only canonical |
| Source provenance | 12 reports carry runtime source bindings; one is mechanically recovered, two new reports record clean source revisions, and nine older reports remain lost because they used uncommitted code |
| Current completion result | 17/24, **70.83%**, after releasing both robot-side supports and rechecking rack-only seating for 0.70 s |
| Boundary decision | **not qualified**; only entry attitude is supported in simulation |
| Live RGB-D service | fail-closed; current flush-tag evidence fails and the prior floating-tag evidence is retracted |
| CI architecture | core modules and CPU tests do not require optional FastAPI imports |
| Checkpoints | reports contain hashes, but weights under `logs/` and `checkpoints/` are absent from a clone |
| Hardware claim | none |

T0 remains open for the nine source-bound reports whose exact uncommitted code
cannot be recovered. New strict-chain and RGB-D evidence starts from clean
commits; the bounded audit finds no lost binding in either new RGB-D report.

## What runs

One continuous zero-gravity episode uses a UR10e to capture, extract, carry,
insert and release a compute module. There is no world constraint, teleport,
direct module pose write or hidden carrier.

| Phase | Controller that executes |
| --- | --- |
| Capture | PPO capture policy |
| Extract | PPO extraction policy |
| Transit | collision-checked solved IK with a robot-side form lock |
| Insert | guarded axial advance while the deployed estimate is inside the derived entry envelope |
| Release | simultaneous hand and compliant-latch release, followed by a 0.70 s rack-only recheck |

The robot rail indexes the world-fixed robot base and does not model its own load
path. A phase is labelled learned only when policy actions actually step it.

## Measured state

### Chain and skills

The current strict chain scores **70.83%**: 17/24 fixed-cohort episodes over
three held-out seeds, Wilson 95% **[50.8%, 85.1%]**. It fails the 95% gate. The
three seed results are 4/8, 8/8 and 5/8. There were no non-finite episodes.

The prior **97.92%** result (94/96) is retained as a legacy supported-settle
baseline. It did not independently release both robot-side supports and then
recheck the module under the rack alone, so it is not the current completion
rate.

| Certificate | Result | Decision |
| --- | ---: | --- |
| Capture v7m130, derived rack | **86.90%**, 7,829/9,009 | misses 95%; the earlier current-rack arm was **85.69%** |
| Extract v18pin, derived rack | **87.64%**, 7,891/9,004 | misses 95%; the earlier pin certificate was **87.75%** |
| Learned insert v20chain | **0.00%**, 0/1,536 | preserved negative baseline |
| Learned insert v24, isolated | **36.77%**, 1,103/3,000 | does not transfer to the chain |
| Learned insert v24, real chain handoff | **0.00%**, 0/96 | not selected |
| Guarded insert, real chain handoff | 94/96 under the legacy supported-settle criterion | selected, but strict completion is 17/24 |

Insertion was not extended blindly. The audit corrected action scaling, matched
the skill and chain handoff geometry, added handoff-conditioned resets, projected
the controller onto module-relative assembly state, and tested staged load-path
release. Learned v24 still fails every recorded predecessor handoff. Its isolated
certificate therefore describes late-stroke states, not the state its caller
delivers. More epochs are not justified until that interface distribution is
made identical and the losing arm is replayed.

The older insertion diagnosis remains preserved: three objective arms ended at
84.26, 84.61 and 84.58 mrad against a **52.4 mrad** success tolerance. Changing
the reward did not move the interface-limited attitude.

### Destination load transfer

The strict simultaneous-release baseline is 17/24. A paired hand-first ablation
on identical seeds and states is **12/24 (50.00%)**, Wilson 95%
**[31.4%, 68.6%]**. It loses, so sequencing is not promoted. The failures after
robot support is removed identify missing rack-side retention/contact physics,
not an insertion tolerance to relax.

### RGB-D perception

The former passing certificate used a tilted tag floating 90 mm above the
current module and is retracted. The current tag is flush with the module top
face. On 256 newly rendered frames it detects 137/256 overall (**53.52%**) and
74/171 in the critical rack region (**43.27%**) against a 99% critical gate.
When visible, position p95 is 3.92 mm, orientation p95 is 47.6 mrad and occupancy
is exact; visibility, not detected-frame precision, is the measured blocker.

A logic defect was also fixed: missed detections could propagate the module as
if attached to the moving tool before capture. The estimator now holds the last
observation until physical capture is verified, uses tool forward kinematics
only while captured, and fails closed after handoff. The current strict RGB-D
run detected 716/865 frames (**82.77%**) but ended during extraction and claims
no relocation. The live service remains unavailable rather than consuming the
retracted certificate.

### Serviceability boundary

[`serviceability_boundary_validation_v2.json`](../evidence/serviceability_boundary_validation_v2.json)
is the current fail-closed comparison:

| Dimension | State |
| --- | --- |
| Rack clearance | mismatch; analytical exclusions do not show separated simulation loss |
| Module section | mismatch; one exclusion agrees and one contradicts at current sample size |
| Robot base offset | mismatch; +10 mm is kinematically feasible but loses in simulation |
| Entry attitude | supported in simulation against the derived `2c/L` boundary |
| Capture geometry | analytical only; no current contact/load certificate |
| Load path | idealized; strict release exposes missing rack-side retention |
| Base compliance | excluded; the fixed robot root prevents the authored spring from deflecting |

Every losing arm is retained. No tolerance was widened. The envelope is **not
qualified**.

## Claim limits

- The robot-side latch geometry is visual. Its rigid fixed joint and compliant
  spring-damper are idealized simulation load paths.
- Simulator force probes are diagnostics, not hardware load ratings.
- The robot root is fixed to the world; spacecraft reaction and compliant-base
  tolerance are not modeled.
- The robustness sweep ranks sensitivities but is not a qualified tolerance band.
- Every learned policy comes from one training seed.
- No current video demonstrates the strict passing configuration.

## Reproduce and continue

```powershell
# CPU trust gate
.\\.venv\\Scripts\\python.exe scripts/check_criterion_currency.py
.\\.venv\\Scripts\\python.exe scripts/check_source_provenance.py --depth 200
.\\.venv\\Scripts\\python.exe scripts/build_evidence_manifest.py --check
.\\.venv\\Scripts\\python.exe scripts/build_script_index.py --check
.\\.venv\\Scripts\\python.exe -m pytest -m "not isaac and not camera and not benchmark"

# Current boundary decision; non-zero means not qualified
.\\.venv\\Scripts\\python.exe scripts/validate_serviceability_boundary.py `
  --output <new-versioned-evidence-path>

# GPU: strict fixed-cohort chain and one RGB-D chain
bash scripts/run_robot_carried.sh certify
bash scripts/run_robot_carried.sh rgbd
```

Never overwrite evidence; use a new versioned filename.

## Branches

| Branch | Status |
| --- | --- |
| `paper/serviceability-qualification` | active: strict release, insertion handoff audit, current RGB-D gate and boundary validation |
| `main` | baseline at `bccce6d`; unchanged |
| `industrial-relocation` | preserved earlier work; not identical to `main` |
| `keyed-interface` | preserved losing keyed-interface exploration; do not delete |
| `origin/agent/zero-g-blade-swap` | preserved historical line; superseded |

The next gate is a visible rack-side retention/contact model tested against the
17/24 baseline on identical states, followed by a camera/tag geometry change
that passes the flush-tag visibility gate. Only then should strict RGB-D batches
be repeated. Learned insertion remains a separate interface-transfer problem;
do not spend more GPU until its reset and real-handoff distributions are the
same by construction.
