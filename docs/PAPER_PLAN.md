# Serviceability qualification study plan

Frozen 2026-08-26. This file defines the research question and evidence gates.
It is not a manuscript. Do not draft one until every required gate below passes.

## Research question

Can a modular orbital replacement task be qualified by deriving a
serviceability envelope from the module, rack, capture interface, robot and
load path; testing the predicted boundary in simulation and on a minimal
physical interface rig; and evaluating learned skills on the states that the
real chain hands them rather than only on isolated reset distributions?

The unit of analysis is a complete service configuration, not a controller.
The result must say which configurations are inside the supported envelope,
which are outside, which contradict the model, and which remain untested.

## Novelty boundary

The proposed contribution is:

1. a constraint-intersection method that derives a serviceability envelope
   before training or simulation;
2. a fail-closed comparison of analytical predictions with controlled
   simulation arms, retaining every losing and contradictory arm; and
3. a chain-handoff audit that compares an isolated skill certificate with the
   same controller evaluated at each reset station and at recorded predecessor
   terminal states.

This is not a new reinforcement-learning algorithm, a new standard connector,
a general space-robotics benchmark, a flight qualification, or evidence that
simulation contact forces are hardware loads. The current latch is visual
geometry whose load path is an idealized wrist-to-module joint. The robot root
is fixed to the world. Those facts bound every claim unless the ablations below
replace them with measured evidence.

## Related work and precise separation

- The [International External Robotic Interface Interoperability Standards
  (IERIIS)](https://internationaldeepspacestandards.com/wp-content/uploads/2024/02/robotic_baseline_final_3-2019.pdf)
  define common fixture and ORU interface classes for interoperable robotic
  handling. This study asks a different question: whether a particular
  module-rack-interface-robot combination can complete a service chain within
  derived geometric and load constraints.
- SIROM integrates mechanical, power, data and thermal connectivity and was
  developed with robotic manipulation and modular servicing in scope
  ([Jankovic et al., IEEE Aerospace 2018](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/9378)).
  HOTDOCK likewise combines mechanical, power, data and thermal coupling in an
  androgynous form-fit interface
  ([Letier et al., IAC 2020](https://elib.dlr.de/139963/)). The iBOSS iSSI work
  reports design and qualification of a multifunctional modular-satellite
  interface
  ([Kortmann et al., IAC 2018](https://www.sla.rwth-aachen.de/cms/institut-fuer-strukturmechanik-und-leichtbau/Forschung/Publikationen/~faog/Details/?file=745832&lidx=1)).
  These are primary interface precedents, not baselines that this simplified
  latch claims to outperform.
- [Space Robotics Bench](https://arxiv.org/abs/2509.23328) provides broad,
  massively parallel task generation and learning baselines for space
  robotics. [Orsula et al.](https://arxiv.org/abs/2405.01134) study procedural
  generation and domain randomization for space peg-in-hole RL. The present
  question is narrower: deriving a serviceable design region, testing its
  boundary, and exposing chain-conditioned failures under fixed controllers.
- [Lee et al., Adversarial Skill Chaining](https://proceedings.mlr.press/v164/lee22a.html)
  identify terminal-state/next-skill initial-state mismatch and learn to
  regularize it. This study does not propose another chaining algorithm. It
  contributes a qualification test: preserve predecessor handoff states, replay
  identical states and seeds under each controller, and report where an
  isolated certificate ceases to predict chain behavior.

## Frozen variables

| Class | Variables | Treatment |
| --- | --- | --- |
| Module | length, width, height, mass | geometry is derived; mass is a simulation ablation |
| Rack | lateral and vertical clearance, destination relief, lead-in geometry | never tune to pass; bracket analytical bounds |
| Robot | base x/y offset, reach residual, Jacobian authority | compare the CPU kinematic model with the executed chain |
| Entry | yaw/pitch attitude, axial engagement, lateral error | evaluate through `2c/L` and `2c/theta` laws |
| Capture | pin/pad/latch clearance, retained pose, axial/lateral/moment capacity | separate visual clearance from contact/load evidence |
| Load path | rigid form lock, compliant mating lock, pad-only control, compliant base | pair states and seeds; record reaction loads when available |
| Chain condition | reset station 0–8 and recorded chain handoff | initial-state SHA-256 is the pairing key |
| Controller | learned insertion v24 and guarded insertion | identical state, seed, budget, predicates and settling check |

Held fixed within a comparison: source commit, checkpoint SHA-256, seed,
initial-state digest, phase budget, success predicate, estimator path and 0.70 s
settling requirement. A criterion change and controller change are never one
arm.

## Baselines

1. The as-built analytical configuration and its nominal simulation arm.
2. Analytical prediction alone versus simulation outcome alone; neither may be
   called validated without the comparison.
3. Guarded insertion versus learned v24 insertion on identical states.
4. Isolated v24 certification versus reset-station and chain-handoff results.
5. Rigid versus compliant form lock on identical states and seeds.
6. Fixed base versus a base spring that is actually in the robot load path.
7. Visual latch-clearance checks versus contact-enabled retention/load tests.

The v24 checkpoint is frozen at SHA-256
`47AA9EFB60F7794BE5CDD1EBD0AD5EC0E94CE00345BCF975D83AE9418D9A1B9F`.

## Metrics and decision rules

Primary metrics:

- end-to-end settled success rate, episode count and Wilson 95% interval;
- success and terminal axial/lateral/orientation errors per boundary point;
- analytical-versus-simulation support at every tested point;
- insertion success per reset station and recorded handoff;
- learned-minus-guarded success for every paired condition and pooled;
- skill-alone minus chain-conditioned success;
- tool-to-module translation/rotation during carry;
- peak/impulse contact force and moment, joint break/saturation events, and
  base deflection only when those signals are in the modeled load path;
- instability, non-finite state, safety abort and termination-reason counts.

Frozen boundary rule: a simulated loss is separated only when that point's
Wilson-95 upper bound is below the nominal Wilson-95 lower bound. A feasible
point supports the model only without a separated loss; an infeasible point
supports it only with a separated loss. Otherwise it is a mismatch. The
predeclared analytical-law ratio gate is `[0.85, 1.15]`. These rules are in
`scripts/validate_serviceability_boundary.py`; do not relax them after seeing a
result.

Frozen controller rule: replace guarded insertion only if v24 is not worse at
every paired condition and is not worse pooled. Both arms remain in the JSON.

## Experiment matrix

| ID | Comparison | Fixed arms | Episodes / replication | Current state |
| --- | --- | --- | --- | --- |
| E0 | CPU geometry currency | current constants vs preserved simulator configurations | 8 recorded configurations plus exact queried geometry | passes |
| E1 | Rack clearance | 6 mm, 11.065 mm nominal, 16 mm; then bracket 10.350 and 11.781 mm bounds on both sides | existing 16/arm is diagnostic; qualification target 3 seeds × 32/arm | mismatch; retain all three arms |
| E2 | Module section | 120×16, 130×20 nominal, 140×26 mm; add near-boundary sections selected by the closed form | existing 16/arm is diagnostic; qualification target 3×32/arm | one outside arm supports, one contradicts |
| E3 | Robot base | nominal, x = −0.70 m, y = nominal +10 mm; bracket the base-y failure | 3×32/arm | +10 mm is analytically feasible but has a separated simulation loss: mismatch |
| E4 | Entry attitude | recorded relief sweep and unchanged-checkpoint throat intervention | preserve existing arms; repeat on 3 seeds for qualification | supported in simulation only |
| E5 | Capture geometry/load | analytical latch clearances, contact-enabled axial/lateral/moment pull, pad-only control | 3 seeds × 32 in simulation; 3 specimens × 10 cycles/direction on rig | analytical only; hardware absent |
| E6 | Load-path type | rigid and compliant form lock on identical state digests and seeds | 3×32/arm with reaction-load telemetry | existing arms are unpaired and idealized |
| E7 | Base compliance | fixed root control and compliant mount connected in the articulation load path | 3×32/arm; sweep derived stiffness/damping, not arbitrary wobble | excluded: current root is fixed |
| H0 | Reset stations | stations 0–8 × seeds 1070/2070/3070 × guarded/v24 | 64 episodes per arm, 3,456 total | complete: v24 786/1,728, guarded 0/1,728; v24 is zero at stations 0–3 |
| H1 | Real chain handoffs | seeds 4070/5070/6070 × guarded/v24 | 32 episodes per arm, 192 total | complete: guarded 94/96, v24 0/96 |
| H2 | Certificate agreement | isolated v24 certificate versus H0/H1 | report all strata and pooled | complete: isolated pooled success does not predict the caller's state |

The minimum physical test is an instrumented capture-interface coupon, not a
full orbital mock-up: three independently assembled specimens, ten
engage/load/release cycles per axial, lateral and closing-axis-moment direction,
calibrated six-axis load measurement, camera-observed slip, and the exact
derived service load. Simulation force telemetry may select cases; it may not
stand in for this measurement.

## Evidence and commands

The current boundary result is
`evidence/serviceability_boundary_validation_v2.json`: `not_qualified`. Only
entry attitude is supported; rack clearance and module section mismatch, base
offset also mismatches, capture is analytical-only, and load-path evidence is
idealized/unpaired. This negative result is the starting state, not a gate to
reinterpret.

The completed controller comparison is
`evidence/insertion_conditioned_controller_v3.json`. Its raw simulations are
bound to clean commit `daa53d6` and its aggregation to clean commit `e6d841d`.
The 27 reset pairs and three chain-handoff pairs retain both controllers and
their terminal physical errors.

CPU checks:

```powershell
.\.venv\Scripts\python.exe scripts/validate_serviceability_boundary.py --output <new-versioned-evidence-path>
.\.venv\Scripts\python.exe scripts/build_evidence_manifest.py --check
.\.venv\Scripts\python.exe -m pytest -m "not isaac and not camera and not benchmark"
```

The validator exits non-zero while the configuration is not qualified. Generate
a new versioned filename after adding evidence; never overwrite v1.

Exact GPU matrix, from a clean committed worktree with the frozen checkpoint
available:

```powershell
& .\scripts\run_conditioned_insertion.ps1 -IncludeChainHandoffs `
  -OutputRoot artifacts\conditioned-insertion\reproduction `
  -EvidencePath evidence\insertion_conditioned_controller_v4.json
```

## Evidence gates before writing

- [x] Core CI imports without optional FastAPI and the CPU suite passes.
- [x] Boundary protocol, tolerances and losing-arm retention are executable and
  versioned.
- [x] H0 and H1 are complete on a clean commit; all 60 arms, state digests and
  losing controllers are retained.
- [ ] E1–E3 are repeated at qualification counts and boundary contradictions are
  resolved or reported as exclusions.
- [ ] E5 has contact/load evidence; otherwise claims remain geometric only.
- [ ] E6 is paired with reaction-load telemetry.
- [ ] E7 puts compliance in the load path; otherwise base-compliance claims are
  excluded.
- [ ] Every quoted run has recoverable committed source and reachable checkpoint
  provenance.
- [ ] At least the capture/load boundary has the physical coupon validation
  above; otherwise the contribution is explicitly simulation-only.

Until these boxes close, report the method and blockers; do not draft a
manuscript or promote the system beyond simulation-guided qualification.
