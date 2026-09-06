# Research charter — correction-aware handoff qualification

Branch `research/correction-aware-handoffs`, opened 2026-09-05 from
`paper/serviceability-qualification` at `3dced19`. One charter, one experiment
plan. When this disagrees with an older plan, this wins and the older text goes.

## The scoping fact underneath all of it, found 2026-09-05

Every workflow report carries a `destination_rack_retention` block. Nothing had
ever read it across cohorts, and reading it splits this repository's evidence
into two families that measure physically different workcells:

| family | cohorts with pawls | without |
| --- | ---: | ---: |
| `robustness64*` sweeps, `relief0*`, `envcount*` — the boundary evidence | **0** | **30** |
| `campaign/gravity` — the whole gravity sweep | **0** | **16** |
| `campaign/secondtask` — the `install` second workflow | **0** | **2** |
| `campaign/factorial`, `rgbdcohorts`, `dose`, `gate_ablation`, `guardbounds`, `noisedchain` | **49** | 0 |

142 reports carry the block; 69 have the destination bay's retention pawls
fitted and **73 do not**. The split is not random — it is the sweep family
against the certification family, and the two have been quoted side by side.

This matters because the success definition re-checks the module after a
**0.70 s free-module window**. With no pawls, in zero gravity, a module released
with any residual velocity coasts and nothing brings it back. Where the pawls
are fitted and engage, the worst recorded module drift while the rack alone
holds is **0.001 mm**.

So the boundary verdicts — the module-section points, the clearance axis, "not
qualified", and the rack-prescription refutation at 110/192 against 64/192 —
were all measured on a destination bay with no retention mechanism at all. That
does not make any of those numbers wrong. It makes their scope much narrower
than the prose citing them has assumed, and it is the most likely reason the
sweep family and the certification family have never cohered.

### The paired arm landed, and it is the whole failure mode

Same point, same three seeds, same checkpoints, `--rack_retention` and nothing
else changed. The success definition is byte-identical between the two arms, so
this is one variable, not a criterion change wearing a policy change's clothes.

| | retention absent | pawls fitted |
| --- | ---: | ---: |
| pass rate | 110/192 = **0.5729** | 187/192 = **0.9740** |
| module delivered to the bay | 187/192 | 187/192 |
| precision given delivery | 110/187 = 0.588 | **187/187 = 1.000** |
| residual median / p95 / max | 2.160 / 4.275 / 5.734 mm | **0.731 / 0.977 / 1.874 mm** |
| drift after release | median +1.187 mm, max +5.234 | **zero, at every episode** |

**77 gained, 0 lost, McNemar two-sided p = 1.3e-23.** The 77 is not a
coincidence: it is exactly the count of `missed_the_terminal_gate` failures
found at the start of the day. Every one of them was the absent fixture. The
five remaining failures are byte-identical across both arms — 384.6, 216.5,
364.9, 1212.5 and 833.7 mm — modules that never left the source bay, which
retention at the destination cannot and should not touch.

The mechanism is confirmed, not inferred. With pawls the lateral error is the
same number at the last held sample, at release and at the terminal check:
0.731 mm median, 1.874 mm worst. Release velocity is 0.000 mm/s. Retention
engaged in 187 of 187.

**This inverts the day's earlier headline.** The criterion curve read on the
retention-absent cohort said the chain needs 4.41 mm to reach 95%. Read on the
cohort with its actual fixture it needs **0.95 to 1.87 mm** depending on seed —
the chain has between 1.3x and 2.6x margin against the unchanged 2.5 mm
criterion, rather than a 1.76x deficit. Both numbers are correct about the
configuration they measured. Only one of them is about the workcell that was
designed.

`evidence/rack_retention_paired_n192_v2.json`,
`evidence/release_drift_retention_v1.json`.

### R1's first test: the published refutation, resolved

The only published *refutation* in the boundary set said the design library's
prescribed channel clearance makes the chain worse — 110/192 to 64/192, 28
gained against 74 lost, p = 5.9e-06 — and the conclusion drawn was that the
library's seated-rest upper bound does not govern. Both arms ran without pawls.
Crossing the two factors settles what each was doing:

| channel relief | retention | pass | precision given delivery | residual median / max |
| --- | --- | ---: | ---: | ---: |
| 4.61 mm (shipped) | absent | 110/192 | 0.588 | 2.160 / 5.734 mm |
| 0.00 mm (prescribed) | absent | 64/192 | 0.342 | 3.192 / 32.928 mm |
| 4.61 mm (shipped) | **pawls** | **187/192** | 1.000 | 0.731 / 1.874 mm |
| 0.00 mm (prescribed) | **pawls** | **175/192** | 0.936 | 0.770 / 32.928 mm |

Where the delivered modules come to rest tells the story the rates hide:

| cell | passes | near miss 2.5–6 mm | 6–10 mm | gross 10–50 mm |
| --- | ---: | ---: | ---: | ---: |
| shipped, no pawls | 110 | 77 | 0 | 0 |
| prescribed, no pawls | 64 | 109 | 2 | **12** |
| shipped, pawls | 187 | 0 | 0 | 0 |
| prescribed, pawls | 175 | 0 | 0 | **12** |

**The refutation survives at a quarter of its size, and its mechanism is now
named.** With the fixture fitted the prescription costs 12 episodes in 192 —
0 gained, 12 lost, two-sided p = 4.9e-04 — not 46. The other 62 episodes the
original comparison attributed to the prescription were the missing fixture.

The 12 are not near-misses pushed across a line, and they are not bad seatings
that the pawls lock in. **They never seat at all.** Checked per episode: they
reach the insert phase, do not time out, and the insertion success predicate
never fires — `predicate_fired = 0`, no settle rows written, retention never
engaged. The module stops 32.0 to 32.9 mm off centre during insertion and the
episode ends there. They are the same environments with the same residuals in
both relief-0 arms — env 3, 8 and 42 at seed 4070 read 32.763, 32.901 and
32.052 mm with and without pawls, because the pawls are never in the path.
Deterministic geometry, not scatter.

That mode has a name in this repository, and its count at the shipped relief is
zero. `jammed_in_the_bay` — reached insert and stopped short of the seated plane
— is the mode the **entry** criterion predicts, and the entry bound is the half
of the library's window that survived the original analysis. Narrowing the
channel to the design point takes away the clearance the module uses to find
centre and produces, in about one episode in sixteen, exactly the failure the
library's entry criterion exists to forecast.

So the upper bound is not refuted by a rate, and the two halves of the window
are doing different jobs: the seated-rest upper bound was never what governed
those 62 episodes — the missing fixture was — while the entry bound is visibly
load-bearing in the tightening direction.

### What the relief is actually for, and it is not lateral

The insert traces name the mechanism, and the 32.5 mm lateral is a consequence
rather than a cause.

**Every insertion swings.** In both configurations the module's orientation
error peaks around 76 mrad partway along the stroke — median **75.71 mrad** with
the relief and **75.76 mrad** without. The swing is the normal motion of this
interface and the channel accommodates it. The relief does not change the
typical swing at all. It bounds the **tail**:

| arm | median | p95 | max | stopped short |
| --- | ---: | ---: | ---: | ---: |
| relief 4.61 mm (shipped) | 75.71 mrad | 77.07 | **80.01** | **0 of 187** |
| relief 0.00 mm (prescribed) | 75.76 mrad | 87.34 | **95.08** | **12 of 187** |

The separation is clean. Across 187 episodes the shipped relief never exceeds
80.01 mrad; every one of the 12 that wedge exceeds 83.6 mrad. And they stop in
the same place — x = 0.220 to 0.2203 m against a seated plane at 0.676 m, about
ninety millimetres into a half-metre stroke — with the guard reading
`clear_to_advance = 0` and holding, which is the guard working rather than
forcing a jammed module.

So the relief is not slack to be designed out. **It is what keeps the entry
swing below the angle at which the module wedges**, and the library's upper
bound — derived from where a *seated* module may rest — cannot see that function
because it is not about the stroke. The design rule this implies is a bound on
peak entry swing, which nothing in the geometry checker currently derives.

`evidence/entry_swing_v1.json`, `scripts/report_entry_swing.py`.

`evidence/prescription_factorial_v1.json`,
`evidence/rack_prescription_retained_paired_n192.json`.

### R1's second test: the module-section axis is a delivery problem, not a seating one

`section_120x16` and `section_140x26` are the two points the "not qualified"
boundary decision names. Both re-run with the pawls fitted, three seeds, paired
against the retention-absent arms that already existed:

| cross-section | pass, no pawls | pass, pawls | gained / lost | p |
| --- | ---: | ---: | ---: | ---: |
| 120 x 16 mm | 88/192 | **152/192** | 64 / 0 | 1.1e-19 |
| 140 x 26 mm | 78/192 | **175/192** | 97 / 0 | 1.3e-29 |

**With the fixture fitted, precision given delivery is 1.0000 at every seed of
both cross-sections.** Every module that reaches the bay seats inside the
criterion. The whole axis reduces to how often the module is captured and
delivered at all:

| cross-section | delivery | precision given delivery | pass |
| --- | ---: | ---: | ---: |
| nominal | 0.974 | 1.000 | 187/192 |
| 140 x 26 mm | 0.911 | 1.000 | 175/192 |
| 120 x 16 mm | 0.792 | 1.000 | 152/192 |

So the boundary decision was reading a capture failure as a clearance failure.
The cross-section does not degrade seating; it degrades grasping, and the
clearance analysis built on top of it was answering the wrong question.

**This is also the strongest validation the residual method has produced**, and
it was a prediction rather than a fit. This morning's delivery/precision split —
computed on the retention-absent cohorts, before any of these runs existed —
said `120x16` had no precision deficit (0.579 against nominal's 0.588) and lost
everything upstream, while `140x26` delivered well and lost precision. It
therefore predicted that fixing the seating would help `140x26` more. It gained
97 episodes against `120x16`'s 64. And the delivery rates measured here, 0.792
and 0.911, are the same numbers to four figures as the retention-absent arms,
which is the internal consistency check: pawls act after seating and cannot
change delivery.

`evidence/section_section_120x16_retained_paired_n192.json`,
`evidence/section_section_140x26_retained_paired_n192.json`.

### The jam is a yaw, and the reference workcell reproduces exactly

Instrumented re-run of three arms on one fixed configuration, with the module's
full quaternion recorded during insertion rather than only the magnitude of its
orientation error.

**The reference reproduces to the episode.** 62/64, 61/64, 64/64 — 187/192,
every episode agreeing individually, maximum residual difference 0.000000 mm
against the established arm. Adding the four quaternion columns perturbed
nothing, so the reference workcell is genuinely fixed.
`evidence/reference_workcell_reproduction_v1.json`.

**The mechanism, at rest, median Euler components in mrad:**

| arm | group | n | roll | pitch | yaw |
| --- | --- | ---: | ---: | ---: | ---: |
| reference (relief 4.61 mm) | seated | 187 | 0.02 | 2.18 | 0.71 |
| prescribed (relief 0.00 mm) | seated | 175 | −0.27 | 2.59 | 1.46 |
| prescribed (relief 0.00 mm) | **jammed** | **12** | −0.81 | −6.57 | **−87.76** |

The separation is total. All twelve jams rest near 88 mrad of yaw; all 362
seated episodes rest within 3 mrad. Roll and pitch are unremarkable in both
groups and do not distinguish them.

So the wedge is a **rotation about the vertical axis, between the side guides**.
Lateral clearance governs it; vertical clearance does not. That kills the
lead-in explanation and makes the discrepancy with the library's own `2c/theta`
bound sharper rather than softer: the bound is about lateral clearance, it is
the right bound for this failure, and it permits 250.7 mm of engagement where
the module wedges at 88.0.

`evidence/jam_mechanism_v1.json`, `scripts/report_jam_mechanism.py`.

### What this obliges, and what it does not

It does **not** retract a number. Every cohort measured what it measured.

It does oblige rescoping, and three published lines need re-running before they
can be quoted as properties of the workcell rather than of a bay with no pawls:

* the **boundary verdicts** — module-section points, clearance axis, "not
  qualified" — all from the `robustness64*` family, 30 cohorts, none with pawls;
* the **gravity sweep**, 16 cohorts, none with pawls. A module released free
  under gravity with nothing holding it falls out of the bay, so the reported
  0/48 at lunar, Mars and Earth is not yet evidence about the interface;
* the **`install` second workflow** at 0/16, and with it this morning's
  "needs a 10.41 mm tolerance" figure.

The residual reading built this morning is unaffected as a method and was right
about where to look. It was applied to a configuration missing its fixture.

## The finding that reframes the project, 2026-09-05

Every framing this project has tried — a handoff gate, a correction model, a
tolerance prescription, a learned seating policy — asked which *upstream*
variable explains the terminal pose. None of them asked *when* the failing error
appears. The settle trace answers that directly, because it records lateral
error on both sides of the hand release.

**The module is placed correctly in every episode. It is released while still
moving, and drifts out of tolerance with nothing holding it.**

| stage | median lateral error | worst | inside the 2.5 mm criterion |
| --- | ---: | ---: | ---: |
| worst sample while the robot holds it | 1.058 mm | 1.909 mm | **186/186** |
| at the instant of release | 0.902 mm | 1.941 mm | **186/186** |
| terminal | 2.144 mm | 5.734 mm | **110/186** |

The guarded advance does its job. The error that fails the task is created
entirely in the unsupported settling window, and it is created by translation:
the module leaves the gripper carrying about 2.4 mm/s and coasts. Drift
predicted as velocity times free-window duration tracks observed drift at
**rho = +0.84**, and `rack_retention_engaged` is false for all 186 episodes in
this configuration, so nothing arrests it. This is Newton's first law, measured.

Selecting episodes by release velocity recovers the whole pass rate — at
2.0 mm/s every admitted episode passes — and in 104 of 186 episodes a slower
release moment was available while the robot was still holding on. That is a
selection over recorded episodes, not proof that waiting would work, and the
report says so; but it locates the entire 41% failure rate in one controllable
decision.

It also explains the noise that has dogged every comparison here. If the outcome
depends on the phase of a residual oscillation at the moment of release, then
the same cell run twice scoring 7/24 and then 4/24 is not an instrument problem
to be averaged away — it is the release phase, and it is fixable rather than
irreducible.

`evidence/release_drift_v1.json`, `scripts/report_release_drift.py`.

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

**Resolved 2026-09-05 13:29: the control passes exactly, on all three seeds.**
35/64, 36/64 and 39/64 — **110/192, the published number to the episode** — with
all 192 episodes agreeing individually and a maximum per-episode residual
difference of 0.000000 mm. Recording the trace changes nothing, and at these
seeds, this environment count, commit and machine the chain is bit-reproducible.
Measured rather than assumed, and claimed no further than measured.
`scripts/check_trace_is_non_perturbing.py` exits non-zero when the two runs are
not the same run. 187 of 192 episodes carry a pre-handoff row.

## H2 is falsified, and the reason matters more than the verdict

On 187 arrived episodes with pre-handoff state joined to outcomes:

* the largest of 24 feature correlations with the terminal residual is **0.135**,
  against a null in which the largest of 24 *noise* correlations at n = 187 has a
  median of **0.159** and a 95th percentile of 0.220. The strongest apparent
  signal is weaker than chance typically produces;
* leave-one-seed-out, the fitted model scores a Brier of **0.2407** against the
  base rate's **0.2427**. A gain of +0.002 where the declared margin was +0.10.

So no pre-handoff predictor exists at this sample size, and the charter's
NARROW condition is met rather than GO.

**But the null has two possible causes and they call for opposite next steps**,
and this is the finding to carry forward. Either the contact interval destroys
the information the incoming state carried, or the incoming state never varied
enough to carry any. The delivered spread says it is closer to the second:

| quantity | p5–p95 spread |
| --- | ---: |
| module lateral position at handoff | 2.131 mm |
| grip error at handoff | 2.029 mm |
| latch relative position error at handoff | 0.124 mm |
| **terminal lateral residual** | **3.931 mm** |

The transit delivers the seating step a narrow, self-similar distribution, and
the residual that decides the outcome is roughly twice as variable as the widest
thing handed in. **The variance that determines success is generated during the
contact interval, not inherited from the handoff.** A correction model has
nothing to condition on because the conditioning variable barely moves.

That is not the same as "correction-aware qualification cannot work". It is
"this experiment never tested it". The experiment that would is to widen the
handoff distribution deliberately — inject lateral and attitude offsets at the
transit-to-insert boundary and find where the fixture stops absorbing them —
which has never been run here and which `solve_insert_reset_bank.py` already has
the machinery for. That is the single recommended next experiment, and it is
cheap: the cohort above took 23 minutes of GPU for all three seeds.

## Day-14 decision, with the margin declared before the data

The primary endpoint is **held-out criterion-curve accuracy**: fit on two
geometry configurations, predict the pass rate of a third at a criterion the
third was not run at, and compare against the two named alternatives.

| | practical margin, declared 2026-09-05 |
| --- | --- |
| **GO** | The traced cohort reproduces 110/192 within its interval; a pre-handoff predictor beats the cohort's base rate by **≥ 0.10 absolute** in held-out Brier score; and the criterion curve fitted on two configurations predicts the third's rate within **± 0.10 absolute** with its interval covering the truth. |
| **NARROW** | The measurement path is trustworthy and the criterion curve transfers, but no pre-handoff predictor beats the base rate. Ship the descriptive tool and the two-workflow tolerance result as an engineering study. |
| **STOP** | The traced cohort does not reproduce 110/192, or the criterion curve fails to transfer between configurations. |

### The decision, reached on day 1 rather than day 14: **NARROW**

Every clause resolved on 2026-09-05, against margins declared before the data
existed:

| clause | result |
| --- | --- |
| traced cohort reproduces 110/192 | **yes**, to the episode, 0.000000 mm residual difference |
| pre-handoff predictor beats base rate by ≥ 0.10 Brier | **no**: +0.002 |
| criterion curve is unbiased within a configuration | **yes**: \|bias\| < 0.002, coverage 0.86–0.99 |
| curve orders three configurations consistently | yes, and the scope block says why that is near-arithmetic |

The measurement path is trustworthy and the descriptive tool works. The method
claim does not survive: neither of its two halves — data efficiency, and a
pre-handoff predictor — beat the alternatives they were declared against.

**What NARROW means here, concretely.** Ship the residual reading as an
engineering contribution: the delivery-against-precision decomposition, the
criterion curve, and the two-workflow tolerance result. Do not spend the
remaining six weeks on a correction-aware method paper. Spend one bounded
experiment — the widened handoff distribution described above — on the question
of whether the method *could* work, because the day-1 answer is that it was
never tested, not that it failed.

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
