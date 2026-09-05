# Research charter — correction-aware handoff qualification

Branch `research/correction-aware-handoffs`, opened 2026-09-05 from
`paper/serviceability-qualification` at `3dced19`. One charter, one experiment
plan. When this disagrees with an older plan, this wins and the older text goes.

## The question, after the evidence moved it

The pivot brief proposed: *can we predict which errors at the start of an
assembly step will be corrected by the robot and fixture, and which will remain
dangerous?* One day of reading existing episodes says that question, on this
system, has no events to fit.

In the 192-episode nominal cohort:

| failure mode | count | what predicts it |
| --- | ---: | --- |
| lost before delivery | 5 | the grip criterion |
| lost in transit | 0 | — |
| **jammed in the bay** | **0** | the entry criterion |
| missed the terminal gate | 77 | nothing |

Every episode reached the final phase. The module never once failed to enter.
Contact correction into the bay does not fail here, so a model of *when
correction fails* has nothing to learn. Pooled across all eighteen trace-bearing
cohorts on disk, only four have both outcomes present, and those four hold six
failures between them.

What separates the 110 successes from the 82 failures is one continuous
quantity: the lateral error the module comes to rest at, cut by
`INSERTION_LATERAL_TOLERANCE_M = 0.0025`. Seventy-seven of the 82 failures land
between 2.51 mm and 5.73 mm; the successes run up to 2.451 mm. The distribution
is continuous across the line, and **the threshold reproduces the recorded
success label for 192 of 192 episodes** — the binary outcome *is* the residual,
dichotomised. Five failures sit at 216 mm to 1213 mm and are a different mode
entirely: the module never left the source bay.

So the question this branch tests is narrower and answerable:

> **Given a frozen controller and fixture, does modelling the residual the
> handoff leaves — rather than the rate it passes — support a qualification
> decision the rate cannot, and does it hold on a configuration it was not fitted
> to?**

## What is already settled, and what it cost

Measured 2026-09-05 from episodes already on disk, no GPU:

* **The reframing is verified, not assumed.** 192/192 label agreement.
* **Criterion transfer works and is the advantage that survives.** From one
  cohort the pass rate at any tolerance: 45.8% at 2.0 mm, 57.3% at 2.5 mm, 89.1%
  at 4.0 mm, 96.4% at 5.0 mm. The nominal chain needs **4.41 mm** to reach 95%.
  A counted rate answers at one criterion and no others, at any sample size.
* **It pays most where the rate says nothing.** The `install` workflow — same
  frozen controller, same bay, different approach path — scored **0 of 16**, a
  rate whose interval is [0, 0.194]. Its residuals: every episode arrived and
  seated, median 6.54 mm, max 10.41 mm. It needs **10.41 mm** for 95%, a factor
  of 2.4 on the relocation chain. That is a quantitative distance to
  qualification from a cohort with zero successes.

Two things were tested and failed, and they are kept because they bound the
claim:

* **The lognormal fit is wrong here.** 0.650 against a counted 0.573, bootstrap
  interval excluding the truth, KS 0.110. Retained only for reaching past the
  sampled range, never reported without its distance.
* **Reading the residual does not reliably buy statistical power.** Against
  `section_120x16` it gained 1.2–1.5× at n ≥ 48; against `section_140x26` it
  gained nothing and lost slightly at n = 96. Neither reading reaches 80% power
  at 96 episodes an arm. **The data-efficiency half of the candidate claim does
  not survive its own test.** The advantage that survives is categorical —
  answering a question the rate cannot answer — not statistical.

## The blocker, stated exactly

**The cohort that varies has no trace; the cohorts with traces do not vary.**

Everything above is *descriptive*: the residual is measured at the end of the
episode, so none of it predicts an outcome before the episode runs, and none of
it is a runtime gate. The predictive step needs pre-handoff state joined to
outcomes that vary, and `robustness192_section/nominal.npz` — the one large
cohort that varies — was run without `--handoff_trace`.

`artifacts/campaign/supervise_traced_nominal.sh` re-runs that point at the same
three seeds with `TRACE=1` and nothing else changed. It carries its own control:
the existing cohort scored 110/192, and if the traced run reproduces that, the
trace is non-perturbing and its pre-handoff state may be joined to outcomes
already published. If it does not reproduce, that is the more important result.

**Seed 4070, 2026-09-05 13:14: the control passes exactly.** 35/64 against
35/64, all 64 episodes agreeing individually, and a maximum per-episode residual
difference of 0.000000 mm. Recording the trace changes nothing, and at this
seed, environment count, commit and machine the chain is bit-reproducible —
measured rather than assumed, and claimed no further than that.
`scripts/check_trace_is_non_perturbing.py`, and it exits non-zero when the two
runs are not the same run.

**And the first read of the joined data goes against H2.** On seed 4070's 62
arrived episodes, the largest of 24 pre-handoff feature correlations with the
terminal residual is **0.181**, against a null in which the largest of 24 noise
correlations at n = 62 has a median of **0.285**. The strongest apparent signal
is weaker than chance typically produces. That is one seed and the remaining two
are running; if it holds at 192 episodes, the fixture is not merely correcting
the incoming error but erasing it, and there is nothing for a pre-handoff
predictor to learn.

## Day-14 decision, with the margin declared before the data

The primary endpoint is **held-out criterion-curve accuracy**: fit on two
geometry configurations, predict the pass rate of a third at a criterion the
third was not run at, and compare against the two named alternatives.

| | practical margin, declared 2026-09-05 |
| --- | --- |
| **GO** | The traced cohort reproduces 110/192 within its interval; a pre-handoff predictor beats the cohort's base rate by **≥ 0.10 absolute** in held-out Brier score; and the criterion curve fitted on two configurations predicts the third's rate within **± 0.10 absolute** with its interval covering the truth. |
| **NARROW** | The measurement path is trustworthy and the criterion curve transfers, but no pre-handoff predictor beats the base rate. Ship the descriptive tool and the two-workflow tolerance result as an engineering study. **This is the current default.** |
| **STOP** | The traced cohort does not reproduce 110/192, or the criterion curve fails to transfer between configurations. |

NARROW is the default rather than a fallback, because the two negatives above
have already removed the data-efficiency claim, and what remains — criterion
transfer — is useful engineering rather than a new method. Promoting it to GO
requires the pre-handoff predictor to work, which nothing yet shows.

Two held-out configurations remain two independent transfer cases regardless of
how many episodes are rolled out. No number of episodes converts them into a
general claim about geometry.

## Rules this branch does not get to bend

1. **Never widen a tolerance to make a gate pass.** The criterion curve reports
   *what tolerance a configuration would need*. That is a design input for the
   interface specification, not permission to move `INSERTION_LATERAL_TOLERANCE_M`.
   The 2.5 mm criterion is unchanged and every rate in this document is quoted
   against it. This is the rule this work sits closest to and the one it must be
   read against.
2. **Never combine a criterion change and a policy change in one result.**
3. **Never claim a learned capability whose checkpoint is unreachable.**
4. A run records the commit it *finished* at. Merge only what cannot alter a
   running job while a cohort is in flight.

## Scope this branch explicitly does not take

The brief proposed a public Franka insertion task as a second system. This
repository already has one — the `install` workflow, same controller and
fixture, different approach path — and it produced the sharpest result of the
first day at 16 episodes. A Franka port is not started until the rack result
earns it: it would cost a baseline training campaign to reach the point the
`install` workflow is already at.

Autonomous recovery, camera redesign, policy retraining and adaptive sampling
are outside this branch.
