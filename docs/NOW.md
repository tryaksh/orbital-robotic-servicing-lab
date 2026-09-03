# Now

Verified repository state. Evidence status is mechanical in
[`evidence/MANIFEST.json`](../evidence/MANIFEST.json); bounded tasks are in
[`NEXT_WORK.md`](NEXT_WORK.md). Last verified: 2026-09-02 on
`paper/serviceability-qualification`, based on `main` at `bccce6d`.

Everything is simulated. Nothing has run on hardware.

## Trust snapshot

| Item | Verified state |
| --- | --- |
| Evidence | 46 canonical, 11 retracted, 158 historical; quote only canonical |
| Source provenance | 13 reports carry runtime source bindings; two match the working source, one is mechanically recovered, and ten older reports remain lost because they used uncommitted code |
| Current completion result | 22/24, **91.67%**, after visible rack retention engages, both robot-side supports release, and the rack alone holds for at least 0.70 s |
| Boundary decision | **not qualified**; only entry attitude is supported in simulation |
| Live RGB-D service | fail-closed, and now complete: one continuous episode grasps, extracts, carries, seats, releases both robot supports and is held by the rack alone for 0.733 s, with 1,772/1,772 detections |
| CI architecture | core modules and CPU tests do not require optional FastAPI imports |
| Checkpoints | reports contain hashes, but weights under `logs/` and `checkpoints/` are absent from a clone |
| Hardware claim | none |

T0 remains open for the ten source-bound reports whose exact uncommitted code
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
| Release | visible rack pawls engage only after measured seating; simultaneous hand and compliant-latch release is followed by a 0.70 s rack-only recheck |

The robot rail indexes the world-fixed robot base and does not model its own load
path. A phase is labelled learned only when policy actions actually step it.

## Measured state

### Chain and skills

The current strict chain with destination retention scores **91.67%**: 22/24
fixed-cohort episodes over three held-out seeds, Wilson 95% **[74.2%, 97.7%]**.
It still fails the unchanged 95% full-chain gate. The three seed results are
6/8, 8/8 and 8/8. There were no non-finite episodes.

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
| Guarded insert, real chain handoff | 94/96 under the legacy supported-settle criterion | selected; strict completion with rack retention is 22/24 |

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

The current-source no-rack control exactly reproduces the strict baseline at
**17/24 (70.83%)**. Enabling only the visible rack capture raises the identical
fixed cohorts to **22/24 (91.67%)**, a gain of five episodes / 20.83 percentage
points. All **22/22** episodes that reach the unchanged measured-seating
predicate engage the rack, survive the rack-only recheck and record 0.0 m / 0.0
rad maximum Rack-to-module drift. The remaining two fail upstream and never
engage, so the full-chain 95% gate remains failed rather than being attributed
to load transfer.

The mechanism is two visible 2.5 x 20 x 20 mm rack-owned pawls, with 2.5 mm
rear-face overlap, 0.5 mm no-snap face clearance and an 81.633 mm open half-gap.
Their simulated load path is a disclosed 600 N / 30 N-m break-rated fixed joint
from `Rack` to `SpareBlade`, enabled only after the live seating predicate. It is
not a world constraint and never writes the module pose. Visual pawl contact is
not simulated; the idealized joint carries the load.

### RGB-D perception

The former passing certificate used a tilted tag floating 90 mm above the
current module and is retracted. The datum remains flush with the module top
face. At the current 640 px resolution the single centred datum detected in
937/1,024 held-out frames (**91.50%**) and 683/683 critical-bay frames, with
position p95 1.19 mm, orientation p95 10.93 mrad and exact occupancy under
unchanged gates. The prior camera and decoder arms remain as preserved losers.

**The continuous blocker was geometric, and it is closed.**
[`rack_sightline_occlusion_v1.json`](../evidence/rack_sightline_occlusion_v1.json)
derives, without a simulator, that the destination bay's own vertical lead-in --
an 80 x 60 x 18 mm plate at 12 degrees over the bay centre line, hanging 25 mm
above the module's top face -- covers a centred flush datum for **154 mm of the
529 mm seating stroke** from both fixed cameras. It is a roof: clearing it means
looking under an 82 mm span through 25 mm of headroom, which foreshortens the
marker cell below the resolution the estimator needs, so no camera placement
fixes it. The derivation validates itself against the recorded loss depth of the
dual-camera run before it reports.

The one change is the datum, not the camera: **two flush plates on the same
plane**, ArUco 23 aft and ArUco 15 forward at module-frame x = ∓0.115 m. The
separation is derived -- it must exceed the lead-in's 203 mm shadow, and each
plate must stay in frame, which leaves ∓[0.1025, 0.1275] m; this is that
interval's centre. Marker size, quiet zone, plate plane, camera placement, lens,
resolution and every estimator gate are unchanged.
[`rack_sightline_datum_pair_v1.json`](../evidence/rack_sightline_datum_pair_v1.json)
reports no depth of the stroke where both plates are unreadable, and that holds
on the primary camera alone.
[`servicing_camera_geometry_v4_datum_pair.json`](../evidence/servicing_camera_geometry_v4_datum_pair.json)
keeps 64/64 workflow-envelope poses covered with 8.62 px minimum marker cell
against the unchanged 8.0 px requirement; neither plate covers the envelope alone
(36/64 and 32/64), which is the measurement that says why there are two.

A logic defect was also fixed earlier: missed detections could propagate the
module as if attached to the moving tool before capture. Reset drains the tiled
camera's blank startup buffers, and a complementary fixed RGB-D view was added
for rack entry. The estimator still holds the last observation until physical
capture and fails closed when no current camera can see a datum.

### The continuous demonstration

**One complete continuous episode now exists, on clean source.**
[`rgbd_strict_rack_retention_datum_pair_seed6070.json`](../evidence/rgbd_strict_rack_retention_datum_pair_seed6070.json),
commit `7a82db2`, tracked worktree clean:

| Stage | What the run recorded |
| --- | --- |
| Capture | PPO capture policy |
| Extraction | PPO extraction policy |
| Transit | robot-carried on the form lock; 1.05 mm and 3.27 mrad maximum tool-to-module drift |
| Insertion | guarded advance, 563 advancing control steps, terminal axial target 0.676 m -- the derived seated plane -- reached at module centre 0.6763 |
| Seating | all seven insertion conditions true, including axial depth; still true after the 0.70 s supported settle |
| Release | both robot-side supports released; `all_conditions_including_released_gripper` true |
| Rack only | pawls engaged after the unchanged measured-seating predicate at step 1720; 0.733 s rack-only recheck observed with 0.0 m and 0.0 rad drift |
| Perception | **1,772/1,772 detections, zero failures, zero consecutive failures**; forward plate carried 1,232 and aft plate 540 |

No hidden movement, no simulator-known module position, no teleport, and no
tolerance was changed: the guarded advance ran on the deployed estimator's own
2 mm / 15 mrad bounds throughout, which is eight times tighter than the entry
flare's catch and is now reported as the tolerance that applied.

**This is n = 1.** It is a demonstration, not a rate. The pooled RGB-D chain
certification is T1 and is the next measurement.

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
| Load path | destination transfer supported in 22/22 eligible simulations; both robot- and rack-side joints remain idealized |
| Base compliance | excluded; the fixed robot root prevents the authored spring from deflecting |

Every losing arm is retained. No tolerance was widened. The envelope is **not
qualified**.

## Claim limits

- The robot-side latch geometry is visual. Its rigid fixed joint and compliant
  spring-damper are idealized simulation load paths.
- The rack-side pawls are visible geometry without contact colliders. Their
  600 N / 30 N-m `Rack`-to-module fixed joint is an idealized simulation load
  path; its reaction magnitude is not exposed.
- Simulator force probes are diagnostics, not hardware load ratings.
- The robot root is fixed to the world; spacecraft reaction and compliant-base
  tolerance are not modeled.
- The robustness sweep ranks sensitivities but is not a qualified tolerance band.
- Every learned policy comes from one training seed.
- The continuous RGB-D demonstration is one episode at one seed. The pooled
  perception-in-the-loop rate is not measured.

## Reproduce and continue

```powershell
# CPU trust gate
.\\.venv\\Scripts\\python.exe scripts/check_criterion_currency.py
.\\.venv\\Scripts\\python.exe scripts/check_source_provenance.py --depth 200
.\\.venv\\Scripts\\python.exe scripts/build_evidence_manifest.py --check
.\\.venv\\Scripts\\python.exe scripts/build_script_index.py --check
.\\.venv\\Scripts\\python.exe -m pytest -m "not isaac and not camera and not benchmark"

# Geometry that needs no simulator, including the sight lines
.\\.venv\\Scripts\\python.exe scripts/check_rack_sightlines.py
.\\.venv\\Scripts\\python.exe scripts/check_servicing_camera_geometry.py

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

Destination transfer is closed with a narrowed claim: 22/22 eligible episodes
hold, while the full chain remains 22/24 and below 95% because two fail before
seating. The flush-tag camera gate now passes; the next gate is the strict RGB-D
chain and a recording. Learned insertion remains a separate interface-transfer
problem; do not spend more GPU until its reset and real-handoff distributions
are the same by construction.
