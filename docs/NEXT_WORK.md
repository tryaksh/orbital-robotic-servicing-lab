# Next work

Every known weakness, exposed defect, unverified assumption and scalability
limit in this repository, as a bounded task. Prioritised: **T0 first** — it is
about whether any of the other numbers can be checked at all.

Each task states the evidence it starts from, the code it touches, how to run
it, what would count as done, and roughly what it costs. Read
[`NOW.md`](NOW.md) first — it is the canonical state and these tasks assume it.

**Two tracks.** `T0`–`T11` are the engineering backlog, ordered by what the
repository most needs. [`Publication track`](#publication-track) at the end is the
same work seen from a paper deadline: which of these a reviewer will insist on,
in what order, and the few things only a submission needs. They do not conflict —
`P1` *is* `T0`, `P2` *is* `T3` — the paper section only adds sequencing and the
claims worth making.

**Two rules are non-negotiable, and they are why some of these tasks are shaped
the way they are.** Never widen a tolerance to make a gate pass; if a criterion
is wrong, replace it with one derived from the parts. And when a success or
failure criterion changes, re-run the *previous* checkpoint under both criteria,
so that a criterion change and a policy change are never quoted as one number.
`play.py --legacy_grip_ball_m` and `--legacy_unbounded_reset` exist for exactly
that.

Compute figures assume the measured machine: RTX 5070 Ti Laptop, 12 GB. A
1024-environment PPO run fits alongside a small evaluation process; two full
training runs do not.

## Index — pick one, then read only that section

| # | Task | Cost | Blocks |
| --- | --- | --- | --- |
| **T0** | No certification is reproducible from committed code | CPU + 1 cert batch | everything; the paper outright |
| **T1** | Certify the chain on the vision task | hours, 1 batch | the strongest claim about perception |
| **T2** | Insert skill: wedged at 84.6 mrad against a 20.5 mrad channel | ~1 h train + 45 m certify | a learned seating phase |
| **T3** | Three training seeds, so numbers carry a spread | 4+ training runs | any claim about a *method* |
| **T4** | Exercise robustness levels 1–4 | evaluation only | a degradation curve |
| **T5** | Randomize the variables the sweep is sensitive to | retrain + re-certify | a tolerance band, not a point |
| **T6** | Grasp and extract miss the 95% gate | cheap to attribute | the skill numbers |
| **T7** | The live service runs the superseded w65 policy set | small if folded into T1 | the demo's credibility |
| **T8** | One epoch, two filenames, two provenance hashes | <1 h, CPU | nothing; a latent trap |
| **T9** | The insert skill's load path still differs from the chain's | half a day + certify | T2's transfer |
| **T10** | Test-suite portability | **done 2026-08-25** | — |
| **T11** | No recording shows the certified chain | ~8 min a clip | the media, and any release |
| **P1–P7** | [Publication track](#publication-track) — the same work on a submission deadline | see section | Frontiers, 2026-11-09 |

---

## T0 — No certification is reproducible from committed code

**Found by this audit, and it outranks everything below it, because it is about
whether the other numbers can be checked at all.**

**Where it stands.** Nine reports record `runtime_source_bindings`: the SHA-256 of
each source file *as it was on disk when the run happened*. That is a strong
provenance record, and nothing had ever verified it. Verified now, against 200 of
this repository's 266 commits:

```
9 reports carry source bindings; 9 cannot be fully recovered from git.
```

Every one. Including `robot_carried_full_chain_pin.json`, the single end-to-end
run of the chain that carries the headline 97.92%. For the pooled certification,
four of the six recorded bindings — `run_workflow_demo.py` (the chain driver,
which owns the phase budgets and the settled re-check), `fiducial.py`,
`perception.py`, `scene_cfg.py` and `grapple_pin_env_cfg.py` — match **no commit
in the repository's history**. Only `assets.py` does.

All three certified seeds agree on the same source hashes and on the same
`policy_set_sha256`, so this is genuinely the certified state and not a mislabelled
artifact.

**Read this precisely.** It does **not** mean 97.92% is wrong. The run happened,
the episodes are the episodes, and the arithmetic is unchanged. It means the run
is **not reproducible from this repository**, and nobody can say what differed
between the code that produced the number and the code that is committed.

**And the difference is not safely assumed cosmetic.** The natural hypothesis is
that the previous session ran its measurements and then wrote explanatory comments
before committing — this repository comments heavily. But the commit that followed
the certification, `7b3e719`, changed `FIDUCIAL_TAG_CENTER_M` from
`(0.0, -0.015, 0.100)` to a flush top-face plate and changed
`FIDUCIAL_TAG_SIZE_M`. Those are geometry constants, not commentary. The chain's
pooled number runs on the *state* task, where the module pose comes from the
simulator and the fiducial plate is not in the measurement path, so that
particular change is very likely irrelevant to the 97.92% — but "very likely
irrelevant" is an assumption, and this project's rules exist because assumptions
of that shape have been wrong five times.

**This is systemic, not one lapse.** All nine reports fail, across three sessions.
The working habit — measure, then write up and commit — guarantees the committed
bytes differ from the measured bytes every time. The provenance field has been
recorded faithfully and has never been usable.

**Code.** `scripts/check_source_provenance.py` (new, this audit) is the checker.
It classifies each binding `recovered` / `working` / `lost`, handles the
CRLF-versus-LF difference between a Windows checkout and git's storage, and proves
that conversion on every binding that does match.

**Run it.**

```bash
python scripts/check_source_provenance.py --depth 200
```

**Recommended action, in order.**

1. **Change the habit, cheaply.** Commit the code *before* running a
   certification. Nothing else in this list is worth anything until a number can
   be tied to a commit.
2. **Record the anchor in the report.** Have `run_workflow_demo.py` write
   `git rev-parse HEAD` and the dirty/clean status alongside
   `runtime_source_bindings`. A hash that matches nothing is a puzzle; a commit
   SHA plus "working tree dirty" is an answer. This is additive report metadata
   and cannot move a number — but it edits the chain driver, so it was left as a
   task rather than done mid-audit without a GPU to test it.
3. **Re-run the chain certification on committed HEAD** and publish it beside the
   current one. If it reproduces 97.92% within its Wilson interval, the provenance
   gap is closed and the number is confirmed. If it does not, that is a finding
   and the difference must be published, not reconciled.

**Done when.** `check_source_provenance.py` reports `recovered` for the chain's
certification, and the driver records a commit SHA so the next one cannot drift.

**Cost.** Steps 1 and 2 are minutes and CPU only. Step 3 is one certification
batch — the same cost as the run being reproduced, and **naturally combined with
T1**, which re-runs the chain anyway.

---

## T1 — Certify the chain on the vision task

**The highest-value missing *measurement* in the project** (T0 outranks it as a
matter of provenance, not of measurement).

**Where it stands.** The pooled 97.92% runs on
`Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0`, the *state* task: the module
pose comes from the simulator, and the guarded advance's "deployed estimate" is
the deployed code path reading ground truth. Perception is certified separately
on 1,024 rendered frames. The RGB-D chain has been run end to end at **one seed**
(`evidence/full_chain_rgbd_service_seed4070.json`) and not since the changes that
produced the current rate. So the two strongest claims in the repository are
measured on different inputs and have never been combined at scale.

**Evidence.** `evidence/workflow_robot_carried_m130pin_guarded_certification.json`
(state, pooled), `evidence/fiducial_rgbd_service_plate.json` (perception, frames),
`evidence/full_chain_rgbd_service_seed4070.json` (both, n=1).

**Code.** `scripts/run_robot_carried.sh` — the `certify` stage runs `$STATE_TASK`
and the `rgbd` stage already runs `$VISION_TASK`
(`Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0`) with
`--perception_backend fiducial_pnp`. The two have never been crossed.

**Run it.** The `certify` stage takes `CHAIN_EXTRA`, so no new script is needed:

```bash
CERT_TAG=m130pin_vision \
CERT_TITLE="Robot-carried relocation, driven by RGB-D fiducial perception" \
CHAIN_EXTRA="--task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 --perception_backend fiducial_pnp" \
scripts/run_robot_carried.sh certify
```

Check first that `CHAIN_EXTRA` reaches the driver after `--task` (the `chain()`
helper passes `--task "${TASK:-$STATE_TASK}"` itself, so passing `--task` twice
may need `TASK=` instead — verify on one environment before spending the batch).
Rendering costs VRAM: if 32 environments will not fit, drop to 16 and raise the
episode count so the pooled n stays at 96.

**Done when.** A pooled rate over three held-out seeds with a Wilson interval,
written to `evidence/`, published *beside* the state number rather than instead
of it — the gap between them is the cost of perception and must be reported as
such. If it falls below the 95% gate, that is the result; do not retune
perception to reach the gate.

**Cost.** Hours, not minutes. Rendering makes each episode substantially slower
than the state task. Budget one overnight batch and time-box it.

---

## T2 — The insert skill is wedged, not creeping

**Where it stands.** The learned insert skill has certified at **0.00%** for this
project's entire history — 1,536 episodes on the current checkpoint
(`evidence/grapple_insert_v20chain_certification.json`), a median of **204 mm
short** against tolerances of 2.5 mm and 52.4 mrad. Seven of the eight ways its
task disagreed with the chain's seating phase are closed, it loses the grip in
**0** of 128 held-out episodes, and its mean reward went positive for the first
time in this project.

**The creep reading is refuted.** For a long time the failure was read as the
policy *creeping* — still moving at 3.65 mm/s when the clock stopped, against
120 mm/s of authority — and the fix that followed was a time cost sized so a full
clock costs 12, below the 15 that failing costs. Trained to convergence at 1,400
epochs and evaluated on 512 held-out episodes, it changes nothing:

| | median short | terminal speed | clock used |
| --- | ---: | ---: | ---: |
| v20chain, time cost −0.10 | 203.6 mm | 3.60 mm/s | 900 / 900 |
| v21time, time cost −0.40, converged | 202.2 mm | 3.98 mm/s | 900 / 900 |

The cost is being **paid, not avoided**. `evidence/insert_attitude_diagnosis.json`.

**What the same episodes actually show.** A rigid part of length *L* entering a
channel with *c* of relief per side fits only while its tilt stays under `2c/L`,
which for the shipped relief is `SERVICE_DELIVERED_ATTITUDE_RAD` = **20.5 mrad**.
The module arrives at:

```
orientation at the end    p5 56.1    median 84.6    p95 119.7 mrad
channel admits                             20.5 mrad
```

**Not one episode in 512 ends inside the angle at which the module could enter at
all.** It is not creeping toward a seat it might reach; it is wedged, and the
remaining axial command does nothing. v20chain sits at 84.3 mrad, so this was
never about the time cost.

**Ruled out rather than assumed.** Not the reset —
`evidence/insert_reset_bank.json` reports `attitude_residual_rad` of 0.0 at every
station, so the module starts square and the episode takes it to 84.6 mrad. Not
grip slip — tool-to-handle holds at 12.2 mm with a p95 of 12.48, which is the
pin's own measured 12 mm feed.

**The mechanism is this project's own thesis.** Two flat pads on a pin cannot
resist a moment about the closing axis; the chain carries the module on a form
lock for exactly that reason, and the skill trains without one. What let it go
unnoticed for so long is that `insertion_misalignment_penalty` normalised
orientation by **0.15 rad** — the *seated* tolerance, which only applies once the
channel is already holding the module square. At that scale an 84.6 mrad module
costs 0.08 a step against the 0.50 the same episode pays for 7.1 mm of lateral,
so the objective ranked a fatal attitude below a survivable offset.

**What was changed.** The insert task now normalises orientation by the rack's own
admittance, `SERVICE_DELIVERED_ATTITUDE_RAD`, derived rather than chosen. The
function default stays 0.15 so every previously published insertion number is
bit-identical. `tests/test_skill_chain_agreement.py` holds both halves.

**Code.** `mdp/insertion.py::insertion_misalignment_penalty` (the
`orientation_scale_rad` parameter); `grapple_pin_env_cfg.py::InsertRewardsCfg`
(the `misalignment` term); `scripts/play.py --latch_enabled` (added here, and how
the lock-on arm was measured).

**Run it.**

```bash
"C:/isaac-sim/python.bat" scripts/train.py --headless \
  --task Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0 \
  --num_envs 1024 --seed 70 --robustness_level 0 \
  --max_iterations 1400 --run_name grapple_insert_l0_seed70_v22attitude

SKILL=Insert CKPT=<highest epoch> TAG=insert_v22attitude \
scripts/certify_grapple_skills.sh
```

**Done when.** Orientation at the end drops below the channel's 20.5 mrad
admittance for a substantial fraction of episodes, and the skill is certified the
same way every other skill is — three curriculum stages, three held-out seeds —
with the rate published beside v20chain's 0.00%. **If the attitude comes down and
the rate does not, publish that too**: it would mean entry angle was necessary and
not sufficient, which is a further result rather than a failure to report.

Then verify it *in the chain*, which is the standard extraction is held to:
`--insert_controller policy` runs the learned seating head to head against the
scripted guarded advance on the same workcell. A skill that certifies alone and
loses in the chain is the failure mode this repository has paid for most.

**Cost.** ~1 hour training at 1024 environments, ~45 minutes for the full
three-stage certification, ~30 minutes for the chain comparison.

**Note the ordering against T9.** The load path is still pads-only in the skill
and a compliant form lock in the chain. If T9 is done first, T2's retraining
happens under the load path the chain actually uses; if not, this measures a
policy trained on a strictly harder problem than it is deployed into. Either is
defensible, and which one was done must be stated.

---

## T3 — Give the numbers a spread: three training seeds a skill

**Where it stands.** Every policy in this repository is **one PPO training seed**.
The evaluation seeds are held out, so the published rates are honest — but
training repeatability is untested and no number carries a spread. "This policy
scores 85.69%" is not the same claim as "this method scores 85.69% ± something",
and only the second one tells a reader whether a 2-point difference between two
checkpoints means anything.

This matters retroactively: the extract ladder in
`evidence/extract_attribution.json` attributes differences of 1–7 points to
specific task changes, all on single-seed policies. Without a training spread
there is no way to say which of those exceed run-to-run noise.

**Code.** `scripts/train.py --seed`; `scripts/run_grapple_skills.sh`.

**Run it.** Retrain grasp and extract at two further seeds each (e.g. 71, 72)
with everything else identical, then certify each with
`scripts/certify_grapple_skills.sh` and report mean and range per stage.

**Done when.** Each skill's headline number carries a spread across three
training seeds, and `NOW.md` §2 quotes it that way. If the spread turns out to be
wider than the differences the attribution ladder credits to task changes, say so
plainly — that would qualify a published conclusion, and qualifying it is the
point of measuring.

**Cost.** Four training runs. Extract is the long one (~12,600 epochs to the
current checkpoint). Batch them; do not run two at once on 12 GB. Budget several
overnight sessions, or reduce scope to extract only and say that is what was done.

---

## T4 — Exercise robustness levels 1–4

**Where it stands.** Every certification in this repository is at **robustness
level 0**. Levels 1–4 exist, are implemented, and are unexercised by any
published number.

The profiles are cumulative (`robust_insertion_env_cfg.py::configure_robustness`,
and each grapple skill overrides it):

| Level | Adds |
| --- | --- |
| 0 | arm reset noise (0.001, 0.002, 0.004) by stage |
| 1 | wider arm reset noise (0.003, 0.006, 0.012) |
| 2 | + randomized module mass |
| 3 | + slot and guide friction, stiction |
| 4 | + compliant base mount and base wobble |

**The level-4 caveat is already known and must be carried forward.** The
satellite base compliance is authored and **not in the load path** — the robot
spawns with a fixed root, so the declared spring has nothing to deflect and the
measured deflection is 0.000000 on every step. A level-4 number that does not say
this would imply a mount compliance that is not being simulated.

**Code.** `--robustness_level` on `scripts/train.py` and `scripts/play.py`.

**Done when.** At minimum, the *certified* checkpoints are re-certified at levels
1, 2 and 3 with no retraining, producing a degradation curve — that is the honest
first question ("how much does the current policy lose to these perturbations"),
and it costs evaluation time only. Level 4 either gets the fixed-root defect fixed
first or is published with the caveat stated in the report's own scope block.

**Cost.** Evaluation only, no training: three certifications per skill, each
comparable to an existing `certify_grapple_skills.sh` run. Hours, not days.

---

## T5 — Randomize during training the variables the sweep says the chain is sensitive to

**Where it stands.** `evidence/chain_robustness_sweep.json` ranks what breaks the
chain, and training randomizes **none** of it. The two that dominate:

- **module cross-section** — 120 × 16 mm takes the chain from 93.75% to **0.00%**;
- **where the robot parks across the bay** — a **10 mm** error takes it to **6.25%**.

That second one is the rail's indexing accuracy, and nothing in this project had
ever put a number on how good it has to be. A point certification at one base
position is not a tolerance band, and the sweep says the band is narrow.

**Code.** The sweep drives these as *evaluation* flags on the workflow driver —
`--robot_base_y`, `--robot_base_x`, `--module_cross_section_m`,
`--rack_lateral_clearance_mm`, `--module_mass_kg` (`scripts/sweep_chain_robustness.sh`).
Making them training-time randomization means adding events to the skill tasks'
`EventsCfg`, in the pattern `mdp/randomization.py` already uses for module mass.

**Do the geometry first.** `scripts/check_workcell_geometry.py` derives which
module sections the rack accepts at all and the window the lateral clearance must
lie in. Randomizing across a range the geometry rejects trains a policy on
episodes no policy can win — the exact defect that cost extract 39% of its hardest
cases before the reset was bounded. Sample **inside** the derived envelope.

**Done when.** Base lateral position is randomized during training over at least
the ±10 mm the sweep shows the chain cannot currently absorb, and the chain is
re-certified and re-swept at that variable. Success is a **flatter sweep**, not a
higher nominal rate; if the nominal rate drops and the band widens, that is a
win and should be reported as the trade it is.

**Cost.** Retraining grasp and extract (this changes the training distribution,
so the certified checkpoints do not carry over), then a re-certification and a
re-sweep. The largest task on this list. Sequence it after T3, so the spread
exists to judge the result against.

---

## T6 — Grasp and extract miss the 95% gate

**Where it stands.** Grasp certifies at **85.69%** pooled (worst stage 78.68%),
extract at **87.75%** (worst stage 84.08%), against a 95% gate. Extract is no
longer the binding skill; **grasp's worst stage is now the lower of the two.**

**Do not read this as a training-budget problem.** Extract's ladder is the
strongest evidence in the repository about where points come from: 900 epochs
moved it 1.4 points, 2,000 more moved it **0.0**, and three task corrections moved
it **13** on an unchanged checkpoint (`evidence/extract_attribution.json`). Rule:
check the geometry before spending the GPU.

Grasp has had no equivalent audit. The obvious first question is which stage-2
failures are geometric rather than behavioural — the curriculum's stages differ in
how much of the module the rails still hold, and extract's rate fell monotonically
with the freedom left.

**Done when.** Either grasp's failures are attributed the way extract's were, one
change a row on an unchanged checkpoint, with the ladder published; or the gate is
argued to be the wrong gate for a phase that hands over on the *next* phase's
precondition — with the argument written into
`docs/service_interface_spec.md` §10 and the number left where it is. **Not by
widening the tolerance.**

**Cost.** The attribution ladder is evaluation-only and cheap. Any retraining that
follows is not.

---

## T7 — The live service runs the superseded policy set

**Where it stands.** `src/zero_g_blade_swap/service/presets.py` — which CLAUDE.md
describes as "what the live service actually runs" — pins:

```
GRASP               grapple_grasp_l0_seed70_v6w65   ep 2400
EXTRACT             grapple_extract_l0_seed70_v16w65 ep 9700
INSERT_W65_TWO_SLOT grapple_insert_l0_seed70_v12w65  ep 7100
```

That is the **w65 set, two promotions behind** the checkpoints the 97.92% was
measured on (v7m130 / v18pin / v13m130). The service is internally consistent —
its provenance evidence was produced with those weights — so it is not *wrong*, it
is *describing a superseded chain*. A visitor running the live demo sees a chain
two promotions old.

**This was found by the same coverage gap as the chain runner's defaults.**
`scripts/promote_checkpoints.py` was written specifically to stop defaults drifting
behind the promoted set, and it does not cover `presets.py` either.

**Why this is a task and not an edit.** Changing these paths changes what the
service runs, which moves `evidence/full_chain_rgbd_service_seed4070.json` and
`evidence/fiducial_rgbd_service_plate.json` out of agreement with the code. Under
the project's own rule, a refactor that could move a published number must re-run
the affected certification — and that is a GPU run.

**Done when.** The preset names the certified set, the service's RGB-D full-chain
evidence is re-run and re-hashed against it, and `presets.py` is covered by
`promote_checkpoints.py` (or by a test in the shape of
`tests/test_reproduction_path.py`, which is the cheaper and more durable option).
**Natural to fold into T1** — that task already re-runs the vision chain.

**Cost.** Small if bundled with T1. A separate RGB-D service run otherwise.

---

## T8 — Checkpoint provenance: one epoch, two filenames

**Where it stands.** Extract epoch 12600 exists under two rl-games naming
conventions:

```
last_..._ep_12600_rew_172.70488.pth     1341301 bytes  sha ADC247AB...  <- certified
last_..._ep_12600_rew__172.70488_.pth   1341477 bytes  sha A83D3CAC...
```

Their **weights are byte-identical** — the same 17 tensors, verified equal — so
this changes no behaviour. But a report's `checkpoint_sha256` is a *file* hash, so
the two produce different provenance for the same policy, and
`scripts/check_evidence_currency.py` can be made to disagree with itself. The
`m130pin_check` run in `artifacts/` recorded the other hash and therefore a
different `policy_set_sha256` than the certification.

`promote_checkpoints.py` breaks this tie by `(file size, name)`, which selects the
**double-underscore** file — *not* the one the current certification used. Its
docstring's claim that "every certification in evidence/ was produced from"
the double-underscore form is no longer true.

**Recommended action.** Make the tie **refuse** rather than guess: print both and
require an explicit choice. A tool whose job is to prevent silent drift should not
resolve an ambiguity silently. `tests/test_reproduction_path.py` pins the correct
file for the current set in the meantime.

**Cost.** Under an hour, CPU only, no re-certification — the weights are equal, so
no published number moves.

---

## T9 — The insert skill's load path still differs from the chain's

**Where it stands.** Seven of the eight ways the insert skill's task disagreed
with the chain's seating phase are closed and held by
`tests/test_skill_chain_agreement.py`. **The load path is the one still open, and
it is measured rather than overlooked.**

The chain seats with the form lock softened to a bounded spring-damper. The skill
cannot simply switch that on: the lock's joint is authored between the wrist and
the module at their **spawn** poses, and this task's reset writes the module
anywhere along 436 mm of stroke, so PhysX resolves the disagreement by snapping
them together. With the lock as the only change, same checkpoint and seed:

| | Dead inside ten control steps | Roll about the pin |
| --- | ---: | ---: |
| Lock on | 125 of 128 | 247.6 mrad |
| Lock off | 0 of 128 | 9.4 mrad |

**Code.** `mdp.GrappleLatch` must re-anchor after a reset — code, not a
configuration value. The chain's mating numbers are declared on the task next to
the measurement, so it is a one-line change at the call site once the re-anchor
exists.

**Why it matters beyond tidiness.** This class of defect — a skill trained on a
problem the chain does not hand it — is **the failure mode that has cost this
project the most**. The insert skill certified at 0.00% while holding the grip
perfectly, because it was being asked for something the geometry forbids. The
remaining disagreement is the last place that can still be true.

**Done when.** `GrappleLatch` re-anchors on reset, the skill runs with the lock in
the chain's configuration, and the eighth row of
`tests/test_skill_chain_agreement.py` asserts equality like the other seven.
Re-certify the insert skill afterwards — the load path is not a cosmetic change
and the 0.00% was measured without it. **Do this before T2's retraining if both
are being done**, or T2 trains against a load path the chain does not use.

**Cost.** Half a day of implementation. Re-certification is ~20 minutes.

---

## T10 — Test suite portability (**done 2026-08-25, recorded here so it is not redone**)

Three test modules imported optional dependencies at module scope, so a missing
package was a **collection error** that took the whole suite down rather than a
skip: `test_fiducial.py` (cv2) and `test_pose_head.py` (torch) failed in the CI
environment, which installs neither, and `test_service_api.py` (httpx via
Starlette's TestClient) failed under the simulator's interpreter, which does not
have it. CI runs `pytest -m "not isaac and not camera and not benchmark"` on
ubuntu with numpy, pyyaml and h5py only, so **CI could not have been green.**

Fixed with `pytest.importorskip` guards. The suite now collects and passes under
both interpreters: 238 passed / 4 skipped under the CI-like environment, 242
passed / 1 skipped under Isaac's Python.

---

## T11 — No recording shows the certified chain

**Where it stands.** Audited 2026-08-25 by checking every clip against the report
of the run that produced it rather than against its filename. Every video in the
repository is from superseded checkpoints, pre-fix geometry, or both, and **none
achieved settled seating**:

* `1_grasp_and_extract.mp4` and `2_carry_across_on_the_rail.mp4` run the w65
  checkpoints, two promotions behind the certified set, in a run that ended at
  `reached_phase: transit` with 43.2 mm of final lateral error;
* `3_full_chain_seated.mp4` is **misnamed** — its run reports
  `lateral_alignment: false`, 4.62 mm against a 2.5 mm tolerance;
* the perception clips are from 2026-08-15, predating the workcell move, the
  130 x 20 mm module and the derived rack.

The 4.62 mm failure is the blocker that deriving both lead-ins closed, so those
clips are an honest record of the problem and a dishonest record of the solution.

**Why it was not fixed in the audit.** Producing honest media is a GPU run, and
the audit deliberately spent no GPU time on anything but the one training resume
it was asked to carry.

**Run it.**

```bash
scripts/run_robot_carried.sh rgbd    # ~8 min, 1 env, RGB-D active, writes a report
```

**Done when.** Three or four clips exist whose runs report
`seated_conditions_still_held_after_settling: true`, covering the learned skills,
the robot-carried transit, the complete seating chain and perception — attached to
a GitHub Release rather than committed, since `*.mp4` stays gitignored and the
repository stays ~21 MB. `docs/DEMOS.md` holds the full detail and the caption
each clip needs.

**Cost.** Minutes per clip. The check that matters costs nothing: read
`seated_conditions_still_held_after_settling` in the run's report before
publishing the clip.

---

---

# Publication track

**Target: a submission in 10–12 weeks.** This section exists because the work is
close to publishable and the gap is *measurement discipline*, not results. It does
not replace T0–T11; it says which of them a reviewer will insist on, and adds the
ones only a paper needs.

| Venue | Fit | Deadline / speed | Notes |
| --- | --- | --- | --- |
| **Frontiers in Robotics and AI — Space Robotics** | Best topical match: the collection explicitly invites learning-based control, manipulation, simulation and experimental validation for on-orbit servicing | **2026-11-09** — about 11 weeks | First choice. The collection framing matches this project's actual contribution almost exactly. |
| **Aerospace** (MDPI) | Good, if framed as a servicing problem | ~18.5 days median to first decision | The fastest path. **Must not read as an Isaac Lab demonstration** — lead with the servicing problem and the interface specification, not the simulator. |
| **IJARS — Service Robotics** | Solid fallback; scope covers space exploration, design, control, simulation and validation | Rolling; 6–12 pages, double-anonymised | The length cap is the binding constraint: this project has more evidence than fits, so the selection has to be deliberate. |

## What the paper actually claims

The temptation is to lead with 97.92%. **That is the weakest available framing** —
a success rate on one simulated workcell, with no hardware, invites the reviewer
to ask what it generalises to, and the honest answer is "not measured".

The defensible contribution is what this project did that is unusual:

1. **The binding constraint in robotic servicing of modular hardware is the
   mechanical interface, not the controller — and here it is quantified.** 6 N of
   holding force against 66.4 N demanded, a factor of eleven, with tightening the
   grip measured to make it *worse*. Then a redesign that closes it, with the
   losing arm kept: on finger pads alone, 0 of 16 environments retain the
   transform and the module travels 913 mm while the tool travels 168.
2. **An RL objective must be scaled against the constraint that binds, and getting
   that wrong produces a policy that fails geometrically rather than
   statistically.** The insert skill spent this project's entire history at 0.00%
   while its objective normalised orientation by the *seated* tolerance (0.15 rad)
   when the binding constraint was *entry* at `2c/L` = 20.5 mrad. The policy
   converged to 84.6 mrad — wedged, not slow — and every diagnosis that read it as
   creep proposed a time cost, which is now measured to change nothing. This is a
   transferable lesson about reward design in contact-rich assembly, and it is the
   most novel thing here.
3. **Skills trained in isolation silently describe a different problem than the
   chain that runs them.** Eight dimensions differed between the insert skill's
   task and the chain's seating phase, and the skill certified at 0.00% while
   holding the grip perfectly. The mitigation — a source-level agreement test that
   runs without a simulator on every commit — is a methodological contribution
   rather than a bug fix.
4. **Design-for-serviceability requirements derived from manipulation
   measurements** rather than chosen: the module cross-section envelope, the
   two-sided bound on rack clearance, and the lateral indexing accuracy the rail
   needs.

Claims 2 and 3 are the paper. Claim 1 motivates it. Claim 4 is the deliverable
that makes it matter to a spacecraft designer. The 97.92% is *evidence for* the
architecture, reported with its limits — not the headline.

## P1 — Close the provenance gap before writing a word

**This is T0, and for a paper it is not optional.** A reproducibility statement
that says "the code that produced these numbers is not in the repository" is not
publishable. Every number in the paper must trace to a commit.

Do T0 steps 1–3, then re-run every certification the manuscript quotes on
committed code. **Sequence everything else after this** — a number re-measured
later at a different commit costs a second re-run.

**Gate.** `check_source_provenance.py` reports `recovered` for every report cited.

## P2 — Three training seeds, because one is not a result

**This is T3, and it is the single most likely cause of rejection.** "We trained
one policy and it scored X" does not support a claim about a *method*. Every
headline number needs a mean and a spread over at least three training seeds.

It also decides whether claim 2 survives review. The attitude-scale correction has
to beat the old scale by more than training noise, or it is an anecdote. **Run
both arms at three seeds each** — the corrected scale and the 0.15 rad original —
as a controlled ablation on one changed parameter.

**Gate.** Every rate in the manuscript carries a spread, and the attitude
ablation's effect exceeds the seed spread — or the claim is weakened to match what
was measured.

**Cost.** Six training runs for the insert ablation, plus two further seeds each
for grasp and extract. The largest line item here. Start it first and batch it
overnight.

## P3 — The ablation table the paper is built around

One table, one changed thing per row, all on held-out seeds. Most rows already
exist and need only re-running at the committed commit and at three seeds:

| Row | Status |
| --- | --- |
| Passive finger grip vs robot-side form lock, for transit | **have** — `robot_carried_interface.json` |
| Rigid vs compliant mating stroke | **have** — `robot_carried_rigid_mating_refuted.json` |
| Insert reward: orientation scaled by seated tolerance (0.15 rad) vs channel admittance (`2c/L`) | **new, P2** — the paper's central ablation |
| Insert time cost −0.10 vs −0.40 | **have** — `insert_attitude_diagnosis.json`, a negative result |
| Learned insert vs scripted guarded advance, head to head in the chain | **have**, needs re-running |
| Skill-task/chain agreement across 8 dimensions, before and after | **have** — `test_skill_chain_agreement.py` plus the certifications either side |
| Module cross-section and rack clearance sweep vs the closed-form envelope's prediction | **have** — `chain_robustness_sweep.json` |

The last row deserves its own figure: a closed-form CPU check that predicts every
simulated cross-section outcome *before* the simulator runs is a strong result for
a design-tool paper.

## P4 — Perception in the loop, at scale

**This is T1.** A space-robotics reviewer will not accept a manipulation result
whose object pose comes from the simulator while perception is validated
separately on still frames. Either report the chain on the vision task, or state
the split so plainly it cannot be mistaken — and expect to be asked why the
measurement was not made.

Report the state-task and vision-task rates side by side. The gap *is* the cost of
perception, and it is a result in itself.

## P5 — Robustness as a curve, not a point

**T4 and T5 together.** Levels 1–3 re-certified on unchanged checkpoints give a
degradation curve for evaluation cost only, which is the cheapest figure in this
plan. Randomising the sensitive variables *during* training (T5) is the expensive
one; if time runs short, publish the degradation curve and name the randomisation
as future work rather than doing it badly.

Carry the level-4 caveat explicitly: the base compliance is authored and not in
the load path, so a level-4 number would imply a mount compliance that is not
being simulated.

## P6 — Say what sim-to-real would take

Required for a space-robotics venue and currently absent. Not experiments — an
honest, specific analysis:

- what the contact model does and does not represent (contact forces are a
  relative damage proxy, not an absolute budget);
- the jaws carry no collider, so pad-on-pin contact is not simulated;
- the base is fixed, so reaction into the servicing spacecraft is not in the load
  path — a real free-flyer or a compliant mount changes the problem;
- no connector mating, cabling, thermal or vacuum effects;
- which hardware experiment would falsify the interface specification most
  cheaply.

Reviewers reward a paper that bounds its own claims. This project already has the
discipline; it has never written it as a section.

## P7 — Make the artifact citable

- **Checkpoints are not in the repository** (`logs/` and `checkpoints/` are
  gitignored). A reproducibility statement needs them somewhere permanent —
  Zenodo, with a DOI, is the usual answer and integrates with GitHub releases.
- Tag the commit the paper describes.
- `evidence/MANIFEST.json` is already close to a machine-readable artifact index;
  cite it directly.
- The demonstration videos (T11) belong with the artifact, and must show the
  certified chain rather than the current footage.

## Ten-week shape

| Weeks | Work |
| --- | --- |
| 1–2 | P1 provenance. Start P2 seed runs immediately — they are the long pole, and everything else can proceed while the GPU is busy. |
| 3–5 | P2 completes. P3 ablation table assembled from re-run evidence. |
| 4–6 | P4 vision-chain certification (overlaps P3; different GPU sessions). |
| 6–7 | P5 degradation curve. T11 media. |
| 7–9 | Write. P6 sim-to-real section. Figures from `evidence/`. |
| 9–11 | Internal review against the two non-negotiable rules, then submit. |

**If the schedule slips, cut P5's training randomisation and T2's further insert
work before cutting P1 or P2.** Provenance and seed spread are what make the paper
reviewable; an extra ablation only makes it stronger.

## What not to claim

- No hardware result, no flight readiness, no TRL claim.
- Not "97.92% success at on-orbit servicing" — it is one simulated workcell, one
  module geometry, one rack, at robustness level 0.
- Not that the learned skills meet the project's own gate. They do not, and the
  chain exceeding them is explained rather than glossed.
- Never quote a skill rate and a chain rate as though they measured the same
  thing.

---

## Not tasks: things already settled

Do not spend GPU hours re-deriving these. `NOW.md` §4 lists them with their
evidence — the compliant mating stroke, the rail carrying the robot rather than
the module, the refuted depth-dependent attitude envelope, the module
cross-section result, and the two dead ends (widening the channel, shortening the
module). Each was measured, and each is preserved with its losing arm.
