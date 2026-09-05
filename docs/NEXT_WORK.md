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

## H1 — Join pre-handoff state to outcomes that vary

**The blocker.** The cohort that varies has no trace; the cohorts with traces do
not vary. `artifacts/robustness192_section/nominal.npz` is 192 episodes at
110 successes and was run without `--handoff_trace`. Pooled over all eighteen
trace-bearing cohorts, four have both outcomes present and hold six failures
between them.

**Running.** `artifacts/campaign/supervise_traced_nominal.sh`, launched
2026-09-05 13:06. Three seeds, 64 environments, `TRACE=1`, nothing else changed.

**Its own control.** The existing cohort scored 110/192. If the traced run
reproduces that within its interval, the trace is non-perturbing and its
pre-handoff state may be joined to outcomes already published. If it does not,
that is the more important result and it is reported before anything is fitted.

**Done when** 192 traced episodes exist whose handoff rows join to their
outcomes by environment index, and the reproduction check is written as
evidence either way.

**Cost.** ~18–31 minutes a seed measured on the untraced runs, so roughly one to
one and a half hours — but wall clock on this machine is not stable and the
schedule is planned in seeds, not hours.

## H2 — Fit the pre-handoff predictor, and try to fail

Once H1 lands. Predict the terminal residual from state available *before* the
contact interval, and the pass/fail that thresholds it.

**Admissible features only.** The handoff row at `to_phase == INSERT`. Never a
terminal variable: `lateral_error_m` at judgement time defines the failure and
using it to predict that failure is the trap `_freeze` exists to make visible.

**The named alternatives, at matched budget:**

| alternative | question it answers |
| --- | --- |
| the cohort base rate | does any feature beat guessing? |
| nominal geometry / clearance rule | does measured state add anything beyond arithmetic? |
| direct binary predictor on the same features | does modelling the residual beat predicting the outcome? |
| residual regression, thresholded | the candidate |

**Falsified if** no predictor beats the base rate by ≥ 0.10 absolute in held-out
Brier score. At n = 64 the best pre-handoff correlate with terminal residual in
the gate-ablation cohort was ρ = +0.32 across 29 features, which is what the
largest of 29 correlations looks like under the null. Expect this to fail; run
it because the traced cohort is the first sample where it *can* succeed.

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
