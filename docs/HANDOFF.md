# Handoff — session of 2026-09-05, closed 2026-09-06

Branch `research/correction-aware-handoffs`, 41 commits from `3dced19`. Working
tree clean, no jobs running, GPU free. 505 CPU tests pass, ruff clean.

**Read in this order:** [`METHOD.md`](METHOD.md) for what the project is now,
[`DECISION_REPORT_prediction_test.md`](DECISION_REPORT_prediction_test.md) for
where it stands, [`NEXT_WORK.md`](NEXT_WORK.md) for what to do. `CHARTER.md`
holds the investigation that produced them. `NOW.md` is verified state.

---

## The one-paragraph version

The project began the session trying to predict which handoff errors a fixture
corrects. That was falsified in an afternoon — the module jams entering the bay
zero times in 192, and pre-handoff state predicts the terminal residual no
better than noise. Underneath it was a scoping error: **73 of 142 evidence
reports were run with the destination rack's retention pawls absent**, split
systematically between campaign families that were being quoted side by side.
Fitting the pawls takes the nominal point from 110/192 to **187/192**. That
reframed the project into a method — *given existing measurements and a proposed
part or fixture change, predict which phase will fail and how* — which was built,
tested prospectively on two never-run configurations, and produced a mixed
result with a specific, fixable cause.

---

## What is established

**The retention scoping error.** `destination_rack_retention.enabled` is false
in 73 of 142 reports, including every `robustness64*` boundary cohort, the whole
gravity sweep, and the `install` second workflow — while the factorial, RGB-D,
dose and gate-ablation certifications all have it. Fitting it at the nominal
point: 110/192 → 187/192, 77 gained, 0 lost, p = 1.3e-23. The 77 is exactly the
count of terminal-gate failures. Precision given delivery goes 0.588 → 1.000 and
post-release drift goes to zero. `scripts/report_retention_configuration.py`
makes the check one command.

**The boundary verdicts are rescoped, not retracted.** With pawls fitted the
module-section axis is *entirely delivery*: precision given delivery is 1.000 at
every seed of both cross-sections, and pass rates track delivery alone (0.974
nominal, 0.911 at 140×26, 0.792 at 120×16). The "not qualified" decision was
reading a capture failure as a clearance failure.

**The prescription refutation survives at a quarter of its size.** With the
fixture fitted the design library's prescribed clearance costs 12 episodes in
192, not 46 — 0 gained, 12 lost, p = 4.9e-04. Those 12 never seat: predicate
never fires, retention never in their path, residuals bit-identical with and
without the pawls.

**The jam is a yaw**, and it is not close. All 12 jams rest at −87.76 mrad about
the vertical axis; all 362 seated episodes rest within 3 mrad. Roll and pitch do
not separate them.

**Clearance is not a monotone tolerance** — the best finding of the session.
Peak-yaw p95 runs 72.88, 87.08, 60.24 mrad at 15.678, 11.065 and 6.065 mm of
clearance. Clearance both permits the wedge and supplies the room it needs, so
jam risk peaks at intermediate clearance. The design library's prescription
moved this channel from a safe clearance to the worst available one.

**A statistical defect, fixed.** `one_sided_p` was computed from the smaller
discordant tail and carried no direction: 14-0 and 0-14 both returned 6.1e-05.
`rack_prescription_paired_n192` had published a 74-to-28 *loss* as p = 2.95e-06.
Replaced by `improvement_p`/`deterioration_p` with regression tests; nine reports
corrected, originals preserved under `one_sided_p_direction_blind`.

## What was tested and did not survive

* The handoff-gate thesis. The failure it guards has count zero.
* A pre-handoff predictor. Largest of 24 correlations 0.135 against a null
  median of 0.159; held-out Brier gain +0.002 against a declared +0.10.
* The data-efficiency claim. A rank test on residuals beat counting on one
  configuration comparison and lost on another. **Keep this closed** unless
  genuinely new evidence appears.
* The lognormal residual fit. 0.650 against a counted 0.573, interval excluding
  the truth.

## Where the method stands

Four checks composed into a six-step procedure in `METHOD.md`, implemented as
`src/handoff_qualification/change_prediction.py` with a JSON corpus and
`scripts/predict_configuration_change.py` as the entry point. Seven tests, most
asserting refusals.

**The prospective test, scored against a rule fixed beforehand:** on the wider
module every predictor failed including the simple clearance rule. On the
thicker module the simple rule came last, the tool produced the most accurate
single number and named the wrong phase, and the hand prediction got phase,
direction and delivery rate right. The method's *reasoning* beat the simple
alternative; the tool's *implementation* did not.

**The decision survived the wrong numbers.** The method said skip the wider
candidate and run the thicker one. The wider one changed nothing; the thicker
one is where the failure was.

## The single next action

**Soften the refusal rule to a joint-sensitivity report and re-score the same
two configurations.** No GPU: both cohorts and the corpus exist, the change is
to `change_prediction.py` alone.

The tool refused thickness because the corpus never varied it alone. The hand
prediction ignored that and was right. The repair is to report the *joint*
sensitivity with its confound named — "a part that is wider-and-thicker costs
extraction, and this corpus cannot say which" — which would have flagged the
risk while still declining to attribute a cause. The pre-registration named this
outcome in advance as the fixable one.

If it recovers the missed prediction without licensing a guess, that is a result
about the method. If not, phase attribution rests on a human reading a
confounded corpus and cannot be automated — narrow the contribution to the
diagnosis tooling and say so.

## Then, in order

1. **R1b** — finish the clearance 2×2. `artifacts/clearance_factorial/` is
   partial (seven of twelve, unpaired) and labelled; finish from there rather
   than pairing against `robustness64*`, whose scope flag is unrecorded.
2. **R2** — re-run the gravity sweep with pawls. Sixteen cohorts, none with
   retention; a module released free under gravity with nothing holding it falls
   out, so that sweep has not yet tested the interface it claims to.
3. **R3** — the `install` workflow, all three seeds with pawls and traces.
4. **Deliverable D** — Forge peg-insert, scripted controller. Surveyed, not
   started: 57 µm clearance, `engage_threshold`/`success_threshold` map onto
   delivery/precision, `UsdFileCfg.scale` gives a geometry axis, and uniform
   scaling reproduces the rack's width/thickness confound. It cannot test phase
   attribution — it has one phase.

## Two things needing a decision from the owner

**Engineer feedback.** `docs/ENGINEER_EXAMPLE.md` is a one-page case with the
outcome deliberately blank, ready to send. I will not contact anyone without a
named channel and direction. Until someone reacts, "the problem matters" stays
labelled unvalidated.

**A learned baseline on Forge.** No weights ship with the task. I chose a
scripted controller and stated the scope limit — two controller–fixture systems,
no claim about learned skills. Reversible if a trained baseline is wanted.

## Process rules this session paid for

* **Verify a kill by the surviving processes and their output paths**, not by
  the absence of an error. `pkill -f` did not kill a supervisor, which then ran
  70 minutes alongside another campaign, halving throughput on a run described
  as time-boxed.
* **Verify a chained job by its process, not its log.** A log can be empty
  because the job just started. Checking too early nearly produced two Isaac
  processes writing one `--episode_metrics` path — six seconds from corrupting
  the pre-registered experiment.
* **Commit predictions before the data and verify the artifact directory is
  absent at commit time.** Done at every step here; git is the record.
* A run records the commit it *finished* at. `git_source_revision` now also
  reports `code_dirty`, which ignores `evidence/`, `docs/` and `artifacts/` so a
  batch of reports writing each other does not read as an unreproducible run.

## Corrections made mid-session, for the record

Four, all committed: the pawls do **not** lock in bad seatings (those 12 never
seat); undelivered episodes are **not** all capture (they split capture/extract
and invert with the part); the clearance factorial **was** still running and
**had** written episodes; and S1 credit on the wider module rests on one episode
and is not credit — the scorer now says so itself.
