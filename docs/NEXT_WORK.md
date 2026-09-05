# Next work

Bounded tasks for the correction-aware handoff pivot. The question, what is
settled, and the day-14 decision are in [`CHARTER.md`](CHARTER.md); read that
first. Current state is [`NOW.md`](NOW.md).

The pre-pivot backlog — thirty-one tasks, most of them closed, refuted or
absorbed — is
[`archive/NEXT_WORK_pre_pivot_2026-09-05.md`](archive/NEXT_WORK_pre_pivot_2026-09-05.md).
It was retired on 2026-09-05, not deleted: three of its tasks became the pivot's
evidence and the rest record why lines were abandoned. Nothing below restates it.

**What carried over, and why only these.** A task survives the pivot only if the
day-14 decision needs it. T0 (source provenance) survives because a result whose
code cannot be recovered cannot be published. T21 (the delivered attitude)
survives as *done* and load-bearing: the 46 mrad premise is measured at 8.19 mrad
median, and every geometry number derived from the old constant is suspect. The
rest are archived.

---

## H1 — Join pre-handoff state to outcomes that vary (**done 2026-09-05**)

The traced nominal cohort landed at 13:29, three seeds, 23 minutes of GPU.
**The control passes exactly**: 35/64, 36/64, 39/64 — 110/192, the published
number to the episode — with all 192 episodes agreeing individually and a
maximum per-episode residual difference of 0.000000 mm. The trace is
non-perturbing and 187 of 192 episodes carry a pre-handoff row.
`evidence/trace_non_perturbing_v1.json`.

## H2 — Fit the pre-handoff predictor (**falsified 2026-09-05**)

Largest of 24 feature correlations with the terminal residual: **0.135**,
against a null whose largest-of-24 at n = 187 has a median of 0.159. Held out by
seed, the model's Brier is 0.2407 against the base rate's 0.2427 — a gain of
+0.002 where the declared margin was +0.10.
`evidence/pre_handoff_predictability_v1.json`, verdict `falsifies_h2`.

**Do not read this as "the fixture erases incoming error".** The delivered
distribution is narrow — 2.131 mm of lateral spread at handoff, 0.124 mm of
latch relative position error — against a 3.931 mm terminal residual spread. The
variance that decides the outcome is generated *during* the contact interval,
not inherited. A correction model has nothing to condition on because the
conditioning variable barely moves.

## H2b — Widen the handoff distribution on purpose (**the one experiment worth running**)

The single recommended next experiment, and the only way to learn whether a
correction model could work at all. Inject lateral and attitude offsets at the
transit-to-insert boundary, well beyond the 2 mm the transit naturally delivers,
and find where the fixture stops absorbing them.

**Why it is the right next step.** H2 did not show that incoming error is
uninformative; it showed that this cohort never varied it. The boundary the
whole pivot is about has never been sampled.

**Machinery that already exists.** `scripts/solve_insert_reset_bank.py` solves
paired arm and module poses along the seating stroke in closed form, and
`play.py --legacy_unbounded_reset` widens a reset distribution.

**Falsified if** the pass rate stays flat across the full injected range — which
would mean the fixture's acceptance region is wider than anything reachable, and
the qualification question is answered trivially — or if it collapses the moment
any offset is injected, which would mean the delivered distribution is already at
the boundary and the margin is zero.

**Cost.** The traced cohort was 23 minutes for 192 episodes. A five-point
injection sweep at three seeds is under two hours.

## H3 — Transfer the criterion curve between configurations

The criterion curve is the advantage that survived day one. Test whether it
transfers: fit on two of `nominal`, `section_120x16`, `section_140x26`, predict
the third's pass rate at a criterion it was not run at.

**Done when** the prediction and its interval are compared against the
third cohort's measured rate, with the interval declared before the third is
read.

**Falsified if** the predicted rate misses by more than ± 0.10 absolute or the
interval fails to cover.

This needs no new GPU. The three cohorts exist.

## H4 — Finish the second workflow

`install` is the second system and it is two thirds unrun. Seed 4070 scored
0/16 with residuals 4.05–10.41 mm; seeds 5070 and 6070 died in Isaac startup
when the session closed at 12:34 on 2026-09-05, and
`supervise_secondtask.sh` logged `exit=0` for one of them anyway — the log line
was written for a process that had produced no `.npz`.

**Do not restart it until H1 is off the GPU.** Then re-run both seeds with
`TRACE=1`, and verify by the artifact rather than the log line.

**Why it matters more than a Franka port.** Same frozen controller, same bay,
different approach path, and the residual moves by a factor of 2.4 in required
tolerance. That is a two-configuration result already; a Franka port would cost
a baseline training campaign to reach the same point.

## H5 — Why a run exits 0 having written nothing (**diagnosed, mostly closed**)

`supervise_secondtask.sh` reported `exit=0` for two runs that wrote no episodes.
Audited 2026-09-05: **the `rc=$?` rule is intact and there is no hole in it.**
The script captures the status correctly and, more importantly, builds its
aggregation list from `[ -f "${out}.npz" ]` and refuses to aggregate below three
seeds. No certification was ever at risk. The log was the only thing that lied,
and it now names a missing artifact instead of printing a bare `exit=0`.

What remains open is the smaller and stranger half: **`run_workflow_demo.py`
exited 0 after 36 seconds having written no `.npz`.** Both logs truncate at the
same byte, 13.7 s into scene setup, and both carry `Disabling key-value database
because another kit process is locking it` — a second Isaac process was running.
The session closed at 12:33 and took them with it, so the likely story is a kill
that surfaced as a zero status through Git-Bash. That is a guess.

**Done when** either the driver is shown to return non-zero on a scene-setup
abort, or the story above is confirmed and the exemption written down. Low
priority: the artifact check above already prevents it from producing a wrong
number.

## T0 — Source provenance for results that remain in scope (carried over)

Ten older reports were produced by uncommitted code and cannot be reproduced.
The pivot narrows what must be recovered: only the reports the day-14 decision
cites. Everything else may stay lost and labelled.

**Done when** every report cited by `CHARTER.md` carries a runtime source
binding that resolves, or is dropped from the citation list.

---

## Two rules this list does not get to bend

**Never widen a tolerance to make a gate pass.** H3 reports what tolerance a
configuration would need. That is an input to the interface specification, not
permission to move `INSERTION_LATERAL_TOLERANCE_M`. The criterion is unchanged
and every rate here is quoted against 2.5 mm.

**Never quote a criterion change and a policy change as one number.**

Compute figures assume the measured machine: RTX 5070 Ti Laptop, 12 GB. A
1024-environment PPO run fits alongside a small evaluation process; two full
training runs do not.
