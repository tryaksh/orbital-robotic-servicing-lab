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

**The promoted configuration is capture v5 + insert v6.** Both certify scripts
default to it, nothing from 2026-08-16 is promoted, and every number below is
that configuration unless it says otherwise.

| | Result | Evidence |
| --- | ---: | --- |
| Capture | 96.10% | `grapple_grasp_v5_certification.json` |
| Insert, on its own reset | 95.57% | `grapple_insert_v6_certification.json` |
| **Insert, on the states the chain hands it** | **~80%** | derived from `workflow_install_final`; see step 1 |
| ~~Extract, alone~~ | **retracted, see below** | `grapple_extract_v10_certification.json` is 0.00% |
| **Install chain, state-based** | **84.38%** | `workflow_install_final_certification.json` |
| **Install chain, camera in the loop** | **80.38%** | `vision_workflow_camera_certification.json` |
| Install chain, oracle control | 80.38% | `vision_workflow_oracle_certification.json` |
| Install chain, **blind** control | **43.58%** | `vision_workflow_blind_certification.json` |
| ~~Removal chain~~ | **retracted, see below** | — |
| Module pose from 64x64 RGB | 1.75 mm mean, 4.35 mm p95 | `module_pose_head.json` |
| Interface axial hold | 69 N against 66.4 N required | `grapple_pin_axial_pull_gate.json` |
| Onboard compute, full stack | 0.73 ms on CPU, 2.2% of the period | `inference_budget.json` |

**Every extraction and removal figure this project published is retracted.** They
were all measured before the settled-enough velocity limits were derived and
tightened on 2026-08-15 (commit `3851fa0`, 14:58) — extract v8's certification was
written at 13:21, v9's at 14:46, and every removal chain run including the quoted
14.06% predates it. Read against the limit now in the code, **0 of extract v8's
6,156 counted successes qualifies**, and the fastest-settling of them is 3.1×
over, so there are no borderline cases and this is arithmetic rather than an
inference needing a re-run. Do not spend GPU re-confirming it. The only
extraction ever measured under the current criterion is v10, at **0.00%**.

**The vision result is the strong one and the blind arm is why.** Camera matching
oracle proves nothing alone — it is equally consistent with a task that never
needed vision. Blind at 43.58% is what makes the 80.38% mean something.

**A caveat on that, found 2026-08-16 and worth carrying.** Under capture v5 the
camera and oracle arms score 80.38% *each, to the digit*, and two estimators that
cannot be told apart mean the task is not asking either for much. With the
criterion-aligned capture v6 the three arms separate into a monotonic ladder —
oracle 79.17%, camera 69.97%, blind 59.90%, with non-overlapping oracle/camera
intervals — which is the first non-zero cost of perception this project has
measured. A sharper capture needs a better pose estimate, so alignment work and
the vision claim are coupled; expect the margin to move when either changes.

## Step 1: close the installation chain

**The bottleneck is measured and two fixes for it are refuted. Do not repeat
either.** The chain is still 84.38% and the target is still 95%.

**What the bottleneck is.** Insert scores 95.57% on its own reset and roughly
**80%** on the states the chain actually hands it. That gap is the whole of the
chain shortfall, it was invisible in every per-skill certification, and it has
been there since the chain was first built. `run_workflow_demo.py
--handoff_trace` measures it and `scripts/analyse_handoff.py` pools it; nothing
measured it before 2026-08-16, which is how a rule this project has now broken
four times kept being breakable.

The mechanism: insert's reset drew ±0.020 rad of joint noise around one nominal
pose, and the chain hands it a pose **0.157 rad** away on its worst axis at the
median — overwhelmingly `wrist_1`, p95 0.284, max 0.383. Even the fifth-percentile
hand-off is three times the widest value that reset could draw on any joint.

**Refuted fix 1 — widen the per-joint reset noise.** Insert v6 under it scores
**0.00%**, all 534 episodes losing the grip at reset with 65.6 mm of terminal grip
error. The hand-off has a large joint deviation *and* a 12.5 mm grip error
together, because the capture **servoed** there; independent noise gives both
errors at once and the fingers close on nothing.

**Refuted fix 2 — sample measured hand-off arm poses (`handoff_poses.py`).**
Insert v6 scores 26.32% on it against ~80% in the real chain, and 1,000 epochs of
fine-tuning made it *worse* — 24.33%, chain 76.74% → 69.10%, camera 69.97% →
55.21%. Reward converged and oscillated with no trend, so this is a mis-specified
task and not a short budget.

**Refuted fix 3 — pair the module pose with the arm pose.** Built, and **gated
before any GPU was spent**, which is the check that should have preceded fixes 1
and 2. `mdp.reset_from_handoff_bank` draws both from one index:

| Reset built to reproduce the hand-off | insert v6, unchanged |
| --- | ---: |
| Per-joint noise box | 0.00% |
| Measured arm poses, module left at nominal | 26.32% |
| Measured arm *and* module poses, paired | **47.17%** |
| *The real chain's insert phase* | *~80%* |

Pairing recovered half the remaining gap and did not close it, so **nothing was
trained on it**. Run that gate before any future attempt: it costs five minutes
and it would have saved the run that cost the chain 8 points.

**Do not build a fourth reset. The hand-off is not a pose.** The chained driver
latches the holding closure at hand-over — `TwoStageRobotiqAction.hold_latch`,
set per environment by `run_workflow_demo.py` — so the grip cannot relax for the
rest of the workflow, and **no training task sets it**, deliberately, so every
policy sees the gripper behaviour its certification was produced under. The
module in the chain is carried under a *different gripper controller* than the
module in the insert task, and no reset distribution can express that. Three
successively better approximations of the hand-off as a pose returned 0.00%,
26.32% and 47.17%.

**So the next thing to try is the training loop, not the reset.** Train insert
*inside the chain*: run the capture, latch the grip, hand over to the policy
being trained. That is the honest end state of "train across the states your
predecessor produces", and every cheaper approximation of it is now measured.

The insert task's reset is **back to the original** and verified — insert v6
scores 96.29% on it against its certified 95.57%, so the task in the file is the
one its certification describes. The bank machinery stays implemented and
regenerable; enabling it means setting `self.events.reset_handoff`.

**Capture v6align is not promoted and is not the problem.** Aligning capture to
the chain's 10 mm hand-off (`capture_success_mask` now reads
`WORKFLOW_HANDOVER_GRIP_M`, episode 6 s → 10 s) removed 38 of 39 capture-phase
overruns exactly as designed. It cost the chain 8 points only because insert
cannot absorb its hand-off, and it moved the hand-off by 12–20%, not by the
factor of three the previous handover assumed. The "4.31 mm hand-off" in that
handover is capture's *standalone* terminal state; in the chain, hand-over fires
at the 10 mm gate and a one-second seat follows, so v5 and v6 both deliver about
12.5 mm. **Re-promote it once insert can take it.**

## Step 2: make removal work — the highest-value work in the project

Extract certifies at **68.36% alone** and the chained removal at **14.06%**. The
gap is not mysterious and the next session should not re-derive it.

**What is already known, so it is not re-litigated:**

- Extraction went 0.00% -> 10.09% -> 28.48% -> 68.36% through three fixes, none
  of them mechanical: an episode shorter than the median success (15 s -> 25 s),
  an attitude penalty two orders of magnitude below the progress term it
  competed with, and an action space that could rotate the wrist at 0.24 rad/s
  while the module rotated at up to 0.767 rad/s.
- **Those three fixes are real but the 68.36% they add up to is not.** All three
  were measured under the loose limits. See the retraction at the top.
- **The "over-correction" recorded in the previous handover was not one, and
  believing it is the trap.** That handover said the derived limits were too
  strict for a working skill, because retraining against them gave 0.00%. The
  arithmetic says otherwise: extract v8 never satisfied them either — 0 of its
  6,156 counted successes does, the best by a factor of 3.1 — so v10's 0.00% was
  the *first honest measurement* and the 68.36% it was compared against was the
  artefact. `EXTRACTION_ANGULAR_VELOCITY_LIMIT` and
  `EXTRACTION_LINEAR_VELOCITY_LIMIT` are derived, correct, and **must not be
  loosened**; the gap they expose is simply larger than the record admitted.
- **The retraction is now confirmed by measurement, not just arithmetic.** The
  identical v8 checkpoint re-evaluated under current code scores **0.00%**
  (`grapple_extract_v8recert_certification.json`), 8,990 of 9,005 ending on
  `extraction_failed`. Use *that* as the extract baseline; v8's stored terminal
  metrics are not a valid control for anything, because its successes terminated
  the instant they satisfied the old v ≤ 0.10 limit while later runs go to
  failure, so the two sample different moments.
- **The fix is a reward, it was applied on 2026-08-16, and it is the most
  informative result of that session.** `mdp.extraction_settling_penalty` reads
  the derived limits, is zero below them, and ramps over the last 60 mm. Trained
  at two weights, both ~9,000 held-out episodes under identical code:

  | settling weight | 0 | −0.5 | **−2.0** |
  | --- | ---: | ---: | ---: |
  | terminal linear velocity | 0.0961 | 0.2244 | **0.0202** m/s |
  | **module** orientation error | 1.1728 | 1.5163 | **0.3696** rad |
  | cycle time | 12.23 | 11.17 | **7.47** s |
  | **grip** attitude | 0.0613 | 0.0861 | **0.3538** rad |
  | success | 0.00% | 0.00% | 0.00% |

  **At −2.0 the extraction is better on everything about the payload and fails
  only on the gripper** — 4.8× slower arrival, 3.2× straighter module, 1.6×
  faster — and 70.6% of episodes end at the 0.350 rad grip-attitude limit.
  **The wrist rotates, not the module. Do not read this as the pin failing**, and
  do not build a third interface feature: the module the pin holds comes out
  straighter under this reward.

  **The response is not proportional** — −0.5 is worse than no penalty at all —
  so the weight selects between qualitatively different policies rather than
  tuning a trade. Do not expect intermediate weights to give intermediate results.
- **The precondition that killed the force-shaping work is satisfied here, and it
  was checked rather than assumed.** That work failed because a position-
  controlled policy could not act on a force it could sense. Extraction is not in
  that position: `blade_velocity` is already in the extract observation, the
  action space directly commands the motion being regulated, and no dimension
  changes — which is what makes a fine-tune legitimate rather than a retrain.
- Two mechanical interface features were built against this and **both are
  measured as net negatives**: the anti-yaw yoke (cost insertion 67 points to buy
  extraction 0.13) and a modelled latch (jams the module in its rails,
  collapsing travel from 458 mm to 25 mm). **Do not build a third.**

**Do these three before training extraction again, in this order. The first two
are CPU-cheap and both can invalidate the third.**

1. **Sweep the axial hold against grip attitude** in `grasp_diagnostics.py`. The
   69 N gate was only ever measured *at* the head-on attitude. Every extraction
   above is failed at 0.35 rad of grip attitude while the module is straight,
   still gripped at 19.1 of 30 mm, and arriving at a fifth of the speed — so
   whether that criterion protects anything is unmeasured. **Do not move the
   threshold; measure whether it is load-bearing.** If the hold survives 0.354
   rad, the criterion is the thing to revisit and extraction may already be much
   closer than 0.00% suggests.
2. **Decompose the terminal wrist pose** with `play.py --grip_axis_metrics` and
   the arm's joint angles at the end of the pull. This says whether the wrist
   *must* re-orient to brake through the wedge or merely learned to. No training
   run can separate those two.
3. Only then retrain. The next lever is roadmap item 7, an action space that can
   command compliance — measured three times now as the missing capability.

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
- **A reset distribution is a joint distribution, and marginals do not
  reconstruct it.** Two attempts to reproduce the chain's hand-off failed in
  opposite directions because each preserved one correlation and broke another:
  per-joint noise randomised the arm against the module, and the pose bank
  randomised the arm against a module that did not move with it. A hand-off is a
  point on a manifold. **Before training on a reconstructed distribution, run the
  unchanged predecessor's successor on it and check it scores what it scores in
  the real chain.** Five minutes; it would have caught both.
- **A retraction is arithmetic when the recorded metrics allow it.** Every
  extraction figure here was retracted without spending a single GPU-minute, by
  reading the published runs' own terminal velocities against the criterion now
  in the code. Check whether the evidence already answers the question before
  re-running it — and check the *timestamps* of a report against the commits that
  changed what it measures.
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
- **Check for an orphaned `kit` process before starting anything.** One from a
  crashed run survived five hours on 2026-08-16, held 0.79 GB, and competed with
  the overnight pipeline the whole time; evaluation runs went from about 7
  minutes to about 4 once it was killed. `Get-Process kit` and
  `nvidia-smi --query-compute-apps=pid --format=csv` before every launch.
- **512 environments costs about 0.9 GB of 12 GB, measured 2026-08-16**, not the
  near-ceiling this project's notes imply. If a run needs to be faster, 1024 is
  very likely safe — but verify with `nvidia-smi` in the first minutes, because
  an OOM that kills an overnight run costs more than the speedup is worth.
- **`train.py`'s stdout is block-buffered when redirected**, so `train.log` can
  lag minutes behind. Judge progress from the `summaries/` tfevents file and the
  `nn/` checkpoint mtime, not from the log.
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
| Auditing a quoted number | `scripts/check_evidence_currency.py`, **and the report's timestamp against `git log -S` on the criterion it uses** |
| What a chain actually hands each skill | `run_workflow_demo.py --handoff_trace`, `scripts/analyse_handoff.py` |
| Rebuilding the hand-off pose bank | `scripts/build_handoff_pose_bank.py`, on training-side seeds only |
| Per-reward-term training diagnosis | the run's `summaries/` tfevents; `Episode/Episode_Reward/<term>` separates which term a policy is actually optimising |
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
