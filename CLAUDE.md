# Agent handover

Act as the senior robotics simulation engineer who owns this repository.
Preserve its evidence-first approach: diagnose one bottleneck at a time, require
deterministic held-out evaluation before promotion, and never describe a smoke
test or an attractive render as Sim2Real validation.

## The goal, and the order it must be built in

The eventual capability is a **relocation**:

```
GRASP  ->  REMOVAL  ->  RELOCATION  ->  INSERT
```

One module, two slots side by side. Capture it in slot 1, pull it clear of the
rack, carry it across, seat it in the empty slot 2. That is what ISS does as ORU
changeout, and it is the first thing this project would build that is
**servicing** rather than assembly. Everything demonstrated so far is half of it.

**The order is fixed and it is not the order it is tempting to build in:**

1. **Close the installation chain.** Capture + insert, camera in the loop, sits
   at 80.38%.
2. **Make removal work in the chain.** This is the single highest-value piece of
   work in the project and it gates everything after it.
3. **Only then, two slots.**

**Do not start 3 before 2 certifies.** A relocation is a product of four stages
and chained numbers here have consistently landed *below* the product of their
parts. Built today it would fail more often than it succeeds, and a flagship
demonstration that fails is worse than a smaller one that works.

## Where things stand, in numbers

Every figure is deterministic evaluation on three held-out seeds, pooled with a
Wilson interval, terminal state captured before the simulator's auto-reset.

| | Result | Evidence |
| --- | ---: | --- |
| Capture | 96.10% | `grapple_grasp_v5_certification.json` |
| Insert | 95.57% | `grapple_insert_v6_certification.json` |
| Extract, alone | 68.36% | `grapple_extract_v8_certification.json` |
| **Install chain, state-based** | **84.38%** | `workflow_install_final_certification.json` |
| **Install chain, camera in the loop** | **80.38%** | `vision_workflow_camera_certification.json` |
| Install chain, oracle control | 80.38% | `vision_workflow_oracle_certification.json` |
| Install chain, **blind** control | **43.58%** | `vision_workflow_blind_certification.json` |
| **Removal chain** | **14.06%** | `workflow_remove_clock_certification.json` |
| Module pose from 64x64 RGB | 1.75 mm mean, 4.35 mm p95 | `module_pose_head.json` |
| Interface axial hold | 69 N against 66.4 N required | `grapple_pin_axial_pull_gate.json` |
| Onboard compute, full stack | 0.73 ms on CPU, 2.2% of the period | `inference_budget.json` |

**The vision result is the strong one and the blind arm is why.** Camera matching
oracle proves nothing alone — it is equally consistent with a task that never
needed vision. Blind at 43.58% is what makes the 80.38% mean something.

## Step 1: close the installation chain

**Diagnosed, and a fix was training when this handover was written.** Of 113
failures over 576 vision-driven installations: **77 were captures overrunning
their budget**, 24 inserts overrunning theirs, 12 fired then failed the 0.7 s
settling re-check.

Capture scores 96.10% alone and causes 68% of chain failures because of a
criterion mismatch — the third of this exact kind found here. The skill was
certified on **20 mm** of grip error while the workflow refuses to hand over
until **10 mm**. Twice the precision, same 6 s clock.

`capture_success_mask` now defaults to `WORKFLOW_HANDOVER_GRIP_M`, read rather
than restated, and the capture episode went 6 s -> 10 s. The run is
`grapple_grasp_l0_seed70_v6align`, driven by `scripts/run_capture_alignment.sh`,
which also re-certifies the state chain and all three vision arms.

**First action of the next session: read those results.** Capture's *standalone*
number is expected to fall — it is being asked for twice the precision — and the
chain is what it is judged on. If the chain did not improve, the remaining 24
insert overruns and 12 settling failures are the next targets, in that order.

## Step 2: make removal work — the highest-value work in the project

Extract certifies at **68.36% alone** and the chained removal at **14.06%**. The
gap is not mysterious and the next session should not re-derive it.

**What is already known, so it is not re-litigated:**

- Extraction went 0.00% -> 10.09% -> 28.48% -> 68.36% through three fixes, none
  of them mechanical: an episode shorter than the median success (15 s -> 25 s),
  an attitude penalty two orders of magnitude below the progress term it
  competed with, and an action space that could rotate the wrist at 0.24 rad/s
  while the module rotated at up to 0.767 rad/s.
- **The last chain run had 191 of 192 extractions fire their success predicate
  and none survive the 0.70 s settling re-check.** That is a much narrower
  problem than "extraction does not work".
- **I over-corrected that and it is a trap to repeat.** The settling failure is
  drift: the chain stops commanding at success and coasts, so a module still
  moving at *v* drifts *v x 0.70 s* before being judged. The old limits allowed
  0.21 rad against a 0.20 rad tolerance — impossible by construction. I derived
  tighter limits, retrained, and extraction went to **0.00%**, because the skill
  cannot yet *arrive* that gently. `EXTRACTION_ANGULAR_VELOCITY_LIMIT` and
  `EXTRACTION_LINEAR_VELOCITY_LIMIT` in `mdp/grapple.py` are still derived and
  still correct; what is missing is a policy trained to satisfy them.
- **The likely fix is a reward, not a threshold.** Nothing pays the policy to
  arrive settled — it is only punished for arriving unsettled, at the terminal
  step, once. Add a terminal-velocity term to the extract reward and retrain
  against the derived limits. That is the first experiment.
- Two mechanical interface features were built against this and **both are
  measured as net negatives**: the anti-yaw yoke (cost insertion 67 points to buy
  extraction 0.13) and a modelled latch (jams the module in its rails,
  collapsing travel from 458 mm to 25 mm). **Do not build a third.**

**Gate before moving on: the removal chain at 80% or better, three held-out
seeds, `scripts/certify_workflow.sh remove`.**

## Step 3: two slots, one module

Only after step 2. What this needs, none of which exists:

- **A second slot.** New geometry, colliders, and a second insertion goal. The
  existing slot is defined in `assets.py`; a second one placed laterally beside
  it is the smallest change that makes a relocation possible.
- **A real transit.** Today's transit is scripted and *retraces the extraction
  path in reverse*, deliberately: a direct move puts the wrist behind the robot's
  own base and takes the damped-least-squares IK through a near-singularity,
  swinging the shoulder 74 degrees. A lateral move to a neighbouring slot is a
  new motion and its reachability must be checked with
  `scripts/calibrate_grasp_pose.py` **converged** — 2,000 servo steps, not 400 —
  before anything is trained against it.
- **Insertion retrained for the second goal.** The insert skill starts from one
  certified staging pose in front of one slot. A laterally offset slot is out of
  its distribution, which is the same defect that made the first chained extract
  reverse into the rack.
- **Perception gets more interesting, and this is the upside.** With two slots
  the camera must report *which* slot is occupied, not only where the module is.
  That upgrades the vision claim from "locates a part" to "reads the state of the
  rack", and it is the version worth demonstrating.

## Operating rules

- Preserve exact zero gravity and 30 Hz policy / 120 Hz physics unless an
  experiment explicitly tests a change.
- Never resume a checkpoint after changing action or observation dimensions.
  Resuming across a *physics* or *reward* change is allowed and is how most
  policies here were produced; say so when you do.
- Change one failure category per experiment. Two changes may be combined once
  each is independently diagnosed and they target different measured failures —
  say so in the report.
- **Re-derive inherited constants for the task that inherits them.** Extract
  carried the insertion task's action scales for four sessions: 0.03 m/s lateral
  against 0.24 axial, correct for a module inside rails and wrong for one that
  ends free. No reward function fixes an authority ceiling.
- **A skill's success criterion must be at least as strict as the chain's.**
  Three separate failures here came from a number defined in two places that were
  free to disagree: the action scales, the settling velocity limits, and the
  hand-off grip tolerance. **Read constants, never restate them.**
- **Measure the axis before designing against it.** Two interface features were
  built against a rotation nobody had decomposed; it was split evenly across two
  axes and both features addressed one. `play.py --grip_axis_metrics` exists so
  this cannot recur.
- **Before believing a probe, prove it moves what it measures.** Two probes here
  measured nothing — a yaw gate that reported the same number with and without
  the feature it tested, and a camera-shake probe whose camera moved 0.0 mm.
  `scripts/check_perturbations_bite.py` is the pattern.
- **Before believing a solver, converge it.** The extraction pose was called
  kinematically marginal on a 400-step IK residual; at 2,000 steps it converges
  to 0.0114 rad.
- Never weaken a success threshold to make a gate pass. Tightening one to match
  what the chain demands is different, and allowed, and must be stated.
- Never quote a success rate without checking that `evidence/` names the same
  policy the demo loads. `scripts/check_evidence_currency.py` compares SHA-256,
  not filenames, and it caught a chain running a superseded policy.
- A recorded video is a demonstration; a pooled multi-seed report is evidence.
  Record demos with `--stable_lighting` and never quote a number from one.
- A chained workflow gives every phase the episode length its own skill was
  certified on; `PHASE_BUDGET_S` derives it. Never widen the workflow episode to
  make a phase fit.
- Perception may be *characterised* any time — `scripts/check_camera_scale.py` is
  one frame. *Training* a perception policy waits for a certified chain.
- Do not edit `src/` or `scripts/` while an evaluation sweep is running.
- Keep `.deps`, logs, datasets, checkpoints, artifacts and videos out of Git.
- Do not reintroduce the eight-phase swap task.

## Where to read, by task

| Working on | Read |
| --- | --- |
| Any result, claim, or limitation | `docs/status.md` |
| What to do next | `docs/roadmap.md` (the relocation goal is at the top) |
| Explaining the project | `docs/claim_vs_evidence.md`, `docs/portfolio.html` |
| The design deliverable | `docs/service_interface_spec.md` |
| Code and data flow | `docs/architecture.md` |
| Physics gaps | `docs/sim2real_matrix.md` |
| The three skills and their criteria | `tasks/blade_swap/grapple_pin_env_cfg.py` |
| Grip metrics, capture/hold, the latch, derived limits | `tasks/blade_swap/mdp/grapple.py` |
| The camera, the pose head, the blind arm | `tasks/blade_swap/mdp/perception.py`, `vision_grapple_env_cfg.py`, `src/zero_g_blade_swap/pose_head.py` |
| The chain and how it is judged | `scripts/run_workflow_demo.py`, `certify_workflow.sh`, `certify_vision_workflow.sh` |
| Auditing a quoted number | `scripts/check_evidence_currency.py` |
| Is a pose reachable | `scripts/calibrate_grasp_pose.py --robot_base_x`, converged |
| Gripper geometry, ever | `evidence/gripper_collision_envelope.json` |
| Pin and yoke dimensions | `src/zero_g_blade_swap/grapple_geometry.py`, `tests/test_grapple_geometry.py` |

## Evaluation contract

`ManagerBasedRLEnv.step` resets terminated environments before returning, so
reading pose after `step` measures the *next* episode. `TerminalMetricsMixin`
intercepts `_reset_idx` and snapshots each finished episode while the scene still
holds its terminal state.

Certification is one `play.py` run per stage and seed writing
`--episode_metrics`, then `scripts/aggregate_evaluation.py` pooling those rows
into a gated report under `evidence/`. Reports align by column name, so a task
may record extra columns without invalidating earlier runs.

Promotion gate: at least the stated success rate pooled *and* in every stage, at
least 80% in every randomized-parameter bucket, zero instability terminations,
zero non-finite terminal metrics.
