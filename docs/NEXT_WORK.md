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
| **T2** | Insert: attitude is the interface's, not the reward's — **diagnosed, work moved to T9** | done | — |
| **T3** | Three training seeds, so numbers carry a spread | 4+ training runs | any claim about a *method* |
| **T4** | Exercise robustness levels 1–4 | evaluation only | a degradation curve |
| **T5** | Randomize the variables the sweep is sensitive to | retrain + re-certify | a tolerance band, not a point |
| **T6** | Grasp and extract miss the 95% gate | cheap to attribute | the skill numbers |
| **T7** | The live service runs the superseded w65 policy set | small if folded into T1 | the demo's credibility |
| **T8** | One epoch, two filenames, two provenance hashes | <1 h, CPU | nothing; a latent trap |
| **T9** | **The insert blocker.** Give the skill the chain's *mating compliance*, not just "enable the lock" | half a day + ~2 h GPU | a learned seating phase |
| **T10** | Test-suite portability | **done 2026-08-25** | — |
| **T11** | No recording shows the certified chain | ~8 min a clip | the media, and any release |
| **T12** | Grasp and extract are certified on the rack before the clearance was derived | ~45 min a skill | the two skill numbers |
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

## T2 — Insert: the attitude is the interface's, not the objective's

**Status: diagnosed, and the work moved to [T9](#t9--the-insert-skills-load-path-still-differs-from-the-chains).**
Two candidate fixes have now been tried and measured; neither is the answer, and
what they rule out is worth more than either would have been.

**Where it stands.** The learned insert skill has certified at **0.00%** for this
project's entire history — 1,536 held-out episodes
(`evidence/grapple_insert_v20chain_certification.json`), a median of **204 mm
short** against tolerances of 2.5 mm and 52.4 mrad. Seven of the eight ways its
task disagreed with the chain's seating phase are closed, and it loses the grip in
**0** of 128 held-out episodes.

### Refuted 1: it is not creeping

The long-standing reading was that the policy *creeps* — still moving at
3.65 mm/s when the clock stops, against 120 mm/s of authority — and the fix was a
time cost sized so a full clock costs 12, below the 15 that failing costs.
Trained to convergence at 1,400 epochs:

| | median short | terminal speed | clock used |
| --- | ---: | ---: | ---: |
| v20chain, time cost −0.10 | 203.6 mm | 3.60 mm/s | 900 / 900 |
| v21time, time cost −0.40, converged | 202.2 mm | 3.98 mm/s | 900 / 900 |

The cost is **paid, not avoided**.

### Refuted 2: it is not the objective's angular scale

The module ends at ~**84.5 mrad** against `INSERTION_ORIENTATION_TOLERANCE_RAD`
= **52.4 mrad**, with only 2–4% of episodes inside. A 7× stronger orientation
penalty was tried — and briefly published on a mis-read constant, now in
`evidence/RETRACTED.md`. Trained 400 epochs it moved the angle by **0.03 mrad**.

| Objective | Orientation | Inside the 52.4 mrad tolerance |
| --- | ---: | ---: |
| baseline time cost | 84.26 mrad | 2.3% |
| 4× time cost, converged | 84.61 mrad | 2.7% |
| 7× orientation penalty | 84.58 mrad | 3.9% |

**Three objectives, 0.4 mrad apart.** An angle that does not move when the reward
is changed three different ways is not the reward's to give.

### What is left, and it is T9

Not the reset — `evidence/insert_reset_bank.json` reports `attitude_residual_rad`
of 0.0 at every station, so the module starts perfectly square and the episode
takes it to 84.5 mrad. Not the grip — tool-to-handle holds at 12.2 mm with a p95
of 12.48, the pin's own measured feed.

What remains is the load path, and it is this project's own thesis: **two flat
pads on a pin cannot resist a moment about the closing axis.** The chain reaches
**46 mrad** at the identical seating phase, and the only difference is that the
chain carries the module on a form lock while the skill trains without one.

Naively switching the lock on does not work either — engaged on this task the
module is flung, 100% dead inside ten control steps at 313.6 mrad and 589.9 mm/s,
because the latch anchors its transform before the reset writes the module along
the stroke and then fights the difference at up to its 1 kN cap. `play.py
--latch_enabled` exists now to reproduce that in one command.

**So the ordering is settled: do T9 first.** Retraining the insert skill before
the load path matches the chain trains a policy on a strictly harder problem than
the one it is deployed into, which is the failure this whole line of work exists
to stop.

**Evidence.** `evidence/insert_attitude_diagnosis.json` holds all four arms and
regenerates from recorded rows with `python scripts/report_insert_attitude.py`.

**When T9 lands, verify both halves** — the standard extraction is held to and
insertion never has been:

```bash
CKPT=<the retrained checkpoint> TAG=insert_v23lock scripts/verify_insert_skill.sh
```

That certifies the skill on three stages and three held-out seeds, then runs the
same checkpoint inside the full chain against the scripted guarded advance. **The
chain keeps the scripted advance unless the chain arm beats it on the same
seeds**; a skill certification alone does not move the seating phase.

**Cost.** ~1 hour training at 1024 environments, ~45 minutes for the skill
certification, ~25 minutes for the chain arm.

**If the attitude comes down and the rate does not, publish that too.** It would
mean squareness was necessary and not sufficient, which is a further result rather
than a failure to report.

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

## T9 — The insert blocker: the skill's load path is not the chain's

**This is now the insert skill's blocking task, not a tidiness item.** T2 tried
the two cheap explanations and measured both away: the policy is not creeping,
and the objective's angular scale is not the defect. What is left is the load
path, and there is direct evidence for it — the module ends at ~84.5 mrad against
a 52.4 mrad tolerance, and *the angle does not move* when the reward is changed
three different ways.

**Why it is the load path.** Two flat pads on a pin cannot resist a moment about
the closing axis. This project has measured that four independent ways, and it is
the entire reason the chain carries the module on a form lock. The chain reaches
**46 mrad** at the identical seating phase; the skill reaches 84.5. The one
structural difference is that the chain has the lock and the skill does not.

**Three ways of switching it on that do not work, all measured**
(`evidence/insert_attitude_diagnosis.json`, and `play.py --latch_enabled` makes
each reproducible in one command):

| Arm | Dead ≤ 10 steps | Orientation |
| --- | ---: | ---: |
| lock off (as trained and published) | 0% | 84.6 mrad |
| lock on, engaged on the first qualifying step | 100% | 313.6 mrad |
| lock on, engaged after 5 control steps | 96.1% | 308.4 mrad |
| lock on, engaged after 20 control steps | **0%** | **325.1 mrad** |

The last row is the informative one. Deferring engagement **completely removes
the early-death mode** — which confirms the anchoring timing was a real defect,
and `engage_after_steps` now exists for it, defaulting to 0 so nothing published
moves. But the attitude is *worse than with the lock off*. **The timing was real
and it was not the cause.**

**The actual reason, and it is already in this repository's own evidence.** The
insert task's reset places the module **inside** the destination channel, anywhere
along a 436 mm stroke, so the rails are constraining it at the moment the latch
engages. A restoring wrench on a module the rails hold fights the rack rather than
the drift it was built for. `mdp.GrappleLatch`'s own docstring records the same
effect on extraction: a latch engaged on capture "never moved the rotation it was
aimed at and collapsed extraction travel from 465 mm to about 25 mm".

The chain never does this. It arms the lock only once the driver says the module
is **clear of the rails**, and then softens it to the remote-centre *mating
compliance* at the mouth — soft in rotation, with the centre at the part's own
tip — precisely so the lead-ins can still walk the module square. A stiff lock
cannot be reoriented by the channel it is entering, which is the other half of the
jam (`_configure_latch`, and specification §9.6).

**So the work is not "enable the lock", and reaching the chain's load path took
three things, all now in the tree** (`--latch_mating_compliance` on both
`train.py` and `play.py` sets all three):

1. **Anchor after the reset settles.** `engage_after_steps`, defaulting to 0 so
   nothing published moves.
2. **`joint_mode` must be `"fixed"`, not `"compliant"`.** With `"compliant"` the
   load path is the explicit wrench and the mating joint is *never installed*, so
   softening re-anchors a transform that engagement set one line earlier —
   measured byte-identical to not softening at all. The chain runs the lock
   `fixed` for exactly this reason: a fixed joint carries the transit, and
   `soften()` disables it and hands the load path to the remote-centre mating
   joint.
3. **`replicate_physics` must be off.** PhysX copies only the first
   environment's procedurally authored joint, so envs 1..N get the prim and no
   usable joint, and the run dies with `Fixed release latch is missing at
   /World/envs/env_1/...`. `configure_base_rail` records the same defect and
   turns replication off for it. **The skill tasks run replication ON for
   throughput — that is the structural reason the chain's load path was never
   reachable from them.** It costs environments: this trains at 512, not 1024.

**Zero-shot, with all three in place:** episodes survive — 0% dead inside ten
control steps against 100% on the transit lock — and the module gets **20 mm
further in**, 182.2 mm short against 202.2 with no lock. Attitude is worse,
113.3 mrad against 84.6, which is expected and is *not* a result: the policy was
trained on pad contact alone and is being evaluated on a load path it has never
seen. **The measurement that matters is a policy trained under it.**

**Code.** `src/zero_g_blade_swap/tasks/blade_swap/mdp/grapple.py` (`GrappleLatch`,
the `_mating_joints` path around the `joint_mode != "compliant"` early return);
`grapple_pin_env_cfg.py::ZeroGBladeGrapplePinInsertEnvCfg` (`latch_enabled`,
`latch_joint_mode`, the mating caps); `scripts/run_workflow_demo.py` for how the
chain sequences arm → soften → release.

### Result, 2026-08-25: the load path was the depth blocker, and the rack is the rest

`grapple_insert_l0_seed70_v23lock` trained 1,400 epochs at 512 environments under
this configuration, and it moves the skill further than anything since the task was
built. Evaluated on 256 held-out episodes at stage 0, with the same configuration
it trained under:

| | no lock (v21time) | trained on the mating compliance (v23lock) |
| --- | ---: | ---: |
| median shortfall | 202.2 mm | **98.6 mm** |
| episodes reaching seated depth (≤12 mm) | ~0% | **35.5%** |
| median lateral | 7.10 mm | **4.51 mm** |
| median orientation | 84.6 mrad | 102.8 mrad |
| success | 0.00% | 0.00% |

**The load path was the blocker for depth, and T9 is confirmed in that respect.**
A third of episodes now drive the module home, where none did before.

**Orientation is the sole remaining failure, and it is not the policy's either.**
Among the 91 episodes that reach seated depth, orientation **floors** at
56.033 mrad — p5 is 56.035, the minimum is 56.033, and the tail runs to 86.5.
A floor is a surface the module cannot get past, and `2c/L` on the channel's
unrelieved lateral throat is **56.396 mrad**: the floor is 0.994 of it.

`INSERTION_ORIENTATION_TOLERANCE_RAD` is **52.36 mrad**. So **the angle at which
this throat holds a module is 4.04 mrad outside the angle its own acceptance
criterion demands.** A module that merely rests in it cannot pass. The chain
reaches 46 mrad because its form lock holds the module squarer than the rack
does — not because the rack squares it.

> **Two corrections to the mechanism named above, both from
> `evidence/destination_channel_geometry.json`, and the conclusion survives
> both.** This section originally read the band as the module "resting
> corner-to-corner against the channel walls". It is not the walls: the
> destination bay is relieved, so they admit 76.90 mrad of yaw, and these runs
> went through `play.py --latch_enabled`, which applied the relief a *second*
> time and opened them to 97.40. What holds the module is the **lead-in throat**
> at the mouth, which is authored from the rail face and does not move with the
> relief. And a wedge against a wall would be a ceiling; this is a floor. Same
> constant either way — `GUIDE_CENTER_OFFSET_Y` places the guides and the flares
> are derived from the rail face — so the rack change below is unchanged.
> `evidence/RETRACTED.md`.

### Done, 2026-08-25: the rack was re-derived and the floor moved with it

**(1) `GUIDE_CENTER_OFFSET_Y` is derived from the seated criterion, 86.689 →
85.065 mm.** The pads' bound, 12.689 mm, is no longer the binding one:

```
lower  2c/L >= 46.00 mrad delivered   ->  c >= 10.350 mm
upper  2c/L <= 52.36 mrad accepted    ->  c <= 11.781 mm
(pads  hypot(c, 8.00) <= 15.00 mm     ->  c <= 12.689 mm, superseded)
```

Placed at **11.065 mm**, the midpoint *in attitude*, which is the clearance that
maximises the smaller of the two margins — 3.18 mrad on each side. Not on a bound,
because both values this project has used sat on one and each cost a training run.
`tests/test_workcell_geometry.py` pins the derivation and the equal margins.

**(2) Tested by moving it, before spending an epoch on it.** Same checkpoint
(`v23lock`, trained on the old rack), same seed, same 256 episodes:

| throat | `2c/L` | attitude floor | median | success |
| ---: | ---: | ---: | ---: | ---: |
| 12.689 mm | 56.40 mrad | 56.03 mrad | 56.92 mrad | 0.00% |
| 11.065 mm | 49.18 mrad | **45.75 mrad** | 46.85 mrad | **18.85%** |

The first non-zero insert-skill success this project has recorded, and it is
zero-shot on the wrong training rack. Among episodes reaching seated depth,
orientation passes 94.6% where it passed 0%, and the binding condition moves to
**lateral alignment** at 54.8% — 2.464 mm at the median against a 2.5 mm
tolerance, which is the policy's to close and is what the retrain is for.
`evidence/insert_attitude_wall_moved.json`.

**(3) The chain was re-certified at both clearances and is identical seed for
seed** — 97.92% pooled either way, 93.75 / 100 / 100 either way. The chain never
used the headroom that moved.
`evidence/workflow_robot_carried_m130pin_guarded_c11065_certification.json`.

**(4) A defect found on the way, and it is its own finding.** The channel relief
was applied as an increment by a method that runs once from `__post_init__` and
again from anything re-selecting the robustness level — so `train.py
--robustness_level` and `play.py --latch_enabled` each applied it twice. **The
insert skill trained in a 21.91 × 17.23 mm channel and was certified in a
17.30 × 12.61 mm one**, on both axes, for as long as both paths have existed.
Every insert number taken before 2026-08-25 carries it. Fixed by writing absolute
poses; `scripts/check_destination_channel.py` reports the applied relief as a
multiple so it cannot drift back.

**(5) The load path is the task's, not a flag's.** `verify_insert_skill.sh`
passes no `--latch_mating_compliance` to `play.py`, so a checkpoint trained on
the lock would have been certified on pad contact alone — the same shape of
mismatch as the relief. `ZeroGBladeGrapplePinInsertTwoSlotEnvCfg` carries
`latch_enabled`, `joint_mode fixed`, soften-on-engage, engage-after-5 and
replication off. That is the eighth row of
`tests/test_skill_chain_agreement.py`, equal now rather than named as a gap.

**Cost, as spent.** ~25 min for the chain at the new clearance, ~6 min for the
zero-shot probe that gated the training run, ~100 min of training, ~110 min to
verify.

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

## T12 — Grasp and extract are certified on the rack before the clearance was derived

**Where it stands.** `GUIDE_CENTER_OFFSET_Y` moved 86.689 → 85.065 mm on
2026-08-25, which narrows the channel by 1.624 mm per side in **both** bays —
the constant places the source bay's guides as well as the destination's. Grasp
(85.69% pooled) and extract (87.75% pooled) were certified in the wider one.

**What is and is not known.** The chain was re-certified at both clearances and
reproduces **seed for seed**, 93.75 / 100 / 100 either way
(`workflow_robot_carried_m130pin_guarded_c11065_certification.json`). That runs
both skills in situ, so it bounds the effect at the chain level. It says nothing
about the skill level, where the certifications pool three curriculum stages and
the chain runs one.

**The predicted direction is favourable, and it is worth stating in advance so
the measurement can contradict it.** The failure mode extraction has at stage 0
is the grip losing the pin: 65 of 92 stage-0 failures on v17m130 end with the
grip more than 13.5 mm across the pin, against a channel corner of
`hypot(lateral, vertical)`. That corner was **15.000 mm** — exactly
`GRIP_MAX_TRANSVERSE_M`, the offset at which a pad keeps half its face on the pin
— and is now **13.654 mm**. The rack can no longer take the module anywhere the
pads cannot follow. If extraction does *not* improve, that is a result: it would
mean the corner was not the mechanism, and `evidence/extract_attribution.json`
would need a fifth row.

**Run it.**

```bash
SKILL=Extract CKPT=logs/rl_games/zero_g_blade_insertion_contact/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth \
  TAG=extract_v18pin_c11065 scripts/certify_grapple_skills.sh

SKILL=Grasp CKPT=logs/rl_games/zero_g_blade_insertion_contact/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth \
  TAG=grasp_v7m130_c11065 scripts/certify_grapple_skills.sh
```

**Done when.** Both are re-certified on the derived rack with the **unchanged**
checkpoints — one variable, the rack — and `docs/NOW.md` §2 quotes the new
numbers beside the old ones rather than replacing them. If either moves by more
than its Wilson interval, it belongs in `evidence/extract_attribution.json` as a
row of its own.

**Cost.** About 45 minutes a skill: nine runs of 128 environments each.

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
   while it was held by two flat pads that cannot resist a moment about the
   closing axis. The policy
   converged to 84.5 mrad against a 52.4 mrad tolerance and *stayed there* under
   three different objectives — a baseline time cost, a 4× time cost trained to
   convergence, and a 7× orientation penalty — all within 0.4 mrad. The lesson is
   sharper than a reward-shaping one: an angle that does not move when the reward
   is changed three ways is set by the interface, and no objective buys what the
   gripper cannot deliver. That is a transferable result about contact-rich
   assembly and it is the most novel thing here.
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
| Insert: three objectives (baseline, 4× time cost, 7× orientation penalty) all landing at 84.5 mrad | **have** — `insert_attitude_diagnosis.json`; the paper's central ablation |
| Insert with the form lock vs without, once T9 lands | **T9** — the arm that should move the angle |
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
