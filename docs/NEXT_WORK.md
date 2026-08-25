# Next work

Every known weakness, exposed defect, unverified assumption and scalability
limit in this repository, as a bounded task. Prioritised: **T0 first** — it is
about whether any of the other numbers can be checked at all.

Each task states the evidence it starts from, the code it touches, how to run
it, what would count as done, and roughly what it costs. Read
[`NOW.md`](NOW.md) first — it is the canonical state and these tasks assume it.

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

## T2 — Finish the insert time-cost run; it is untested, not refuted

**Where it stands.** The learned insert policy certifies at **0.00%** over 1,536
episodes (`evidence/grapple_insert_v20chain_certification.json`), stopping a
median of **204 mm short** with the whole clock spent, against tolerances of
2.5 mm and 52.4 mrad. It is no longer a task-specification problem: seven of the
eight ways the skill's task disagreed with the chain's seating phase are closed,
the policy loses the grip in **0** of 128 held-out episodes, and the mean reward
went **positive for the first time in this project** (−80 → +13.7), with lateral
error 20.7 → 7.9 mm and orientation 128 → 86 mrad.

It is **creeping, not jamming**: still moving at 3.65 mm/s when the clock stops,
against the 120 mm/s its action scale allows and the 60 mm/s the scripted advance
uses to cover the same stroke in nine seconds. Creeping is what the objective paid
for — progress is potential-based, so covering the stroke pays the same however
long it takes, and dawdling cost 3 over a whole episode against a success worth 30.

**The fix is already in the tree and has not been given a fair test.**
`elapsed_time_penalty` is weighted −0.40 in `InsertRewardsCfg`, so a full clock
costs 12 — deliberately *below* the 15 that failing costs, because a time penalty
larger than the failure penalty makes giving up early the cheaper option. Run
`grapple_insert_l0_seed70_v21time` had reached only **300 epochs**, and the run
before it took 800 before its behaviour settled. 300 epochs is too early to read.

**Code.** `src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py` ~line
1205 (the weighting and the reasoning); `scripts/train_insert_stroke.sh`.

**Run it.** Resume rather than restart — the run is mid-flight and its reward was
still climbing (−5.3 → −2.8 → −0.84 at epochs 100/200/300). `--max_iterations` is
an **absolute** epoch, not a count of further epochs:

```bash
"C:/isaac-sim/python.bat" scripts/train.py --headless \
  --task Isaac-ZeroG-Blade-GrapplePin-InsertTwoSlot-v0 \
  --num_envs 1024 --seed 70 --robustness_level 0 \
  --max_iterations 1400 --run_name grapple_insert_l0_seed70_v21time \
  --checkpoint <the highest-epoch checkpoint under that run>/nn/last_...pth
```

Then certify it the same way v20chain was, so the two are comparable:
`SKILL=Insert CKPT=<path> TAG=v21time scripts/certify_grapple_skills.sh`.

**Done when.** Either the median shortfall drops materially from 204 mm and the
terminal speed rises from 3.65 mm/s toward the 60 mm/s the scripted advance
achieves — in which case publish the new rate beside v20chain's 0.00% — or it
does not, in which case **publish that too**. A time cost that does not move the
creep is a real finding about the objective and belongs in `evidence/` next to
the negative result it failed to overturn. Either way the chain keeps the
scripted guarded advance until a policy beats it head to head.

**Cost.** ~1,100 epochs at 1024 environments ≈ 5.5 min per 100 epochs ≈ **1 hour**
of GPU, plus ~20 minutes to certify. This is the cheapest open task with a real
chance of changing a headline claim.

> **Status note, 2026-08-25 audit.** The run was resumed from epoch 300 to
> **1400** and the checkpoints are on disk under
> `logs/rl_games/zero_g_blade_insertion_contact/grapple_insert_l0_seed70_v21time/nn/`.
> **It was not evaluated**, deliberately: the audit's scope was the repository,
> not the policy, so the measurement is left here rather than half-done.
>
> The reward trace does not look like a policy that learned to hurry. Best mean
> reward by epoch: 300 → −0.84, 700 → +1.98, 900 → +3.46, 1000 → −26.6,
> 1200 → +2.15, 1300 → −0.06. It oscillates around zero rather than climbing.
> **Do not read that as a refutation** — the reward functions differ, so v21time
> and v20chain are not comparable on reward. A full clock costs 12 more under
> v21time, and v20chain's best of +13.7 minus 12 is +1.7, right inside the band
> v21time oscillates in. That arithmetic is consistent with the time cost being
> *paid* rather than *avoided*, which would mean the creep did not move — but it
> is an inference from reward, not a measurement of speed.
>
> **Measure it directly**, the way the published figures were produced. The two
> numbers are per-episode fields, not derived quantities:
>
> ```bash
> STAGES=0 SKILL=Insert TAG=insert_v21time \
> CKPT=logs/.../grapple_insert_l0_seed70_v21time/nn/<highest epoch>.pth \
> scripts/certify_grapple_skills.sh
> ```
>
> Then pool `axial_error_m` and `blade_linear_velocity_mps` from the `.npz`
> rows. That method reproduces v20chain's published figures exactly — median
> 203.64 mm short and 3.6 mm/s terminal speed over its 1,536 stage-0 episodes —
> so the comparison is like for like. Roughly ten minutes at stage 0 on three
> seeds; the full three-stage certification is nine runs.

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

## Not tasks: things already settled

Do not spend GPU hours re-deriving these. `NOW.md` §4 lists them with their
evidence — the compliant mating stroke, the rail carrying the robot rather than
the module, the refuted depth-dependent attitude envelope, the module
cross-section result, and the two dead ends (widening the channel, shortening the
module). Each was measured, and each is preserved with its losing arm.
