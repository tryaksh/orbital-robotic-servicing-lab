# Adapting an assembly workflow to a changed part or fixture

The project's goal, set 2026-09-05: a reusable method that tells a robotics
engineer, before they spend a campaign finding out, **whether an existing
controller will handle a change, where it will fail, and what evidence should
guide the fix.**

This document is the method and its evidence plan. [`CHARTER.md`](CHARTER.md)
holds the investigation that produced it; [`NEXT_WORK.md`](NEXT_WORK.md) holds
bounded tasks. Every investigation supports one of three deliverables or is
deferred:

1. **A tool another engineer can run** — takes finished episodes and a changed
   configuration, returns where it will fail and what to measure.
2. **Evidence of advantage** — predictions made before running, compared against
   straightforward testing, measured in experiments saved.
3. **A focused manuscript.**

## Why this is the method, and not the one I started with

Everything below earned its place by being tested on this repository today.
Nothing is here because it sounded like a contribution.

The original framing — predict which handoff errors get corrected — was
falsified: the module jams entering the bay zero times in 192, and pre-handoff
state predicts the terminal residual no better than noise (largest of 24
correlations 0.135 against a null median of 0.159; held-out Brier gain +0.002
against a declared +0.10). What replaced it is smaller and survived contact
with four separate tests in one day.

### The four checks that make up the method

**1. Configuration audit — is the cohort measuring the workcell you think it is?**
Reading `destination_rack_retention` across 142 reports found 73 run with the
destination bay's retention mechanism absent, systematically split by campaign
family, and quoted side by side with the ones that had it. Fitting it moved the
nominal point from 110/192 to 187/192. This check costs one command and was
worth more than every model fitted today.

**2. Phase attribution — which step actually fails?**
Undelivered episodes split between capture and extract, and which dominates
inverts with the part: nominal 3 and 2, a thinner module 33 and 7, a thicker one
1 and 16. An engineer told "delivery got worse" fixes the wrong phase half the
time.

**3. Delivery against precision — did it never arrive, or arrive badly?**
These need opposite fixes and a pooled rate reports them identically.
`section_120x16` and `section_140x26` both scored worse than nominal; the first
seated as precisely as nominal and lost 40 modules before delivery, the second
delivered well and lost precision.

**4. The criterion curve — how far from qualifying, and at what tolerance?**
The pass rate at any criterion, read off residuals from episodes already run. A
cohort that scores 0 of 16 has a rate interval of [0, 0.194] and says nothing;
its residuals say it needs a 10.41 mm tolerance against the reference chain's
4.41 mm.

### What is available when — prediction is not diagnosis

The paper can accidentally claim the second while demonstrating only the first,
so the stages are separated here and every result must say which one it is.

| stage | what exists | what may be claimed |
| --- | --- | --- |
| **Prediction** | reference cohorts, cohorts that varied *some* dimensions, the proposed change as deltas. **No episode of the changed configuration.** | which phase will fail, delivery or precision, direction and a bounded magnitude |
| **Verification** | the above, plus the changed configuration's episodes, read once against a prediction recorded beforehand | whether the prediction held |
| **Diagnosis** | the changed configuration's failed runs, read freely | what happened, and why — a mechanism |

Diagnosis is where the yaw finding came from and it is genuinely useful, but it
is retrospective: the twelve jams had already happened. A result that reads a
changed configuration's failures and explains them is a diagnosis however
sharp it is, and may not be presented as prediction. The pre-registration file
exists to make that distinction checkable in git rather than asserted.

### The procedure, written so it does not need this rack

The contribution is not the four checks — those are good practice. It is that
they compose into a repeatable procedure taking **existing measurements plus a
proposed geometry change** and returning a phase-level prediction. An engineer
follows it without knowing anything about this workcell.

1. **Audit the reference cohort.** Read every fixture and mechanism flag its
   report records and confirm they are the workcell you intend to deploy. Half
   this repository's evidence failed this check.
2. **Attribute the reference's own failures by phase.** That is the baseline
   phase budget the change will perturb.
3. **Split each phase's failures into delivery and precision** — did the part
   never arrive, or arrive out of tolerance. They take opposite fixes.
4. **Build the dimension–phase sensitivity table from cohorts that already
   varied a dimension.** This is the step that removes the need for local
   knowledge: it is read from data, not from an engineer's memory of the cell.
   On this rack, from three existing cohorts — thickness up drives *extract*
   failures, width and thickness down drive *capture* failures, lateral
   clearance down drives *insert* jams.
5. **Project the proposed change through that table**, one dimension at a time,
   and emit direction plus a bounded magnitude per affected phase.
6. **Refuse where no cohort has varied the relevant dimension.** The procedure
   is only as wide as its sensitivity corpus, and saying so is the difference
   between a tool and a guess.

Step 4 is what a clearance rule cannot do. `2c/theta` knows about lateral
clearance and insertion; it has no representation of a grasp or an extraction,
so it is structurally silent on the phases that carry most of this workcell's
remaining failures.

### The advantage claim, stated so it can fail

Not data efficiency — that was tested and did not survive (a rank test on
residuals beat counting successes on one configuration comparison and lost on
another; neither reached 80% power at 96 episodes an arm). The claim is:

> Given episodes from a reference configuration and a proposed change, the
> method predicts **which phase will fail and whether the failure is delivery or
> precision**, and does so before the changed configuration is run.

**It has already made one such prediction and been right.** Computed on
retention-absent cohorts before the fixture runs existed, the decomposition said
`120x16` had no precision deficit and lost everything upstream while `140x26`
lost precision — so repairing the seating should help `140x26` more. It gained
97 episodes against `120x16`'s 64, and the delivery rates measured afterwards
matched the prediction's inputs to four figures. That is one prediction. The
plan below is about making enough of them to matter.

## The comparator

Every claim is measured against what an engineer would otherwise do: **run the
changed configuration and count successes.** That is the honest baseline, it is
what this repository did for its whole history, and the metric is the number of
episodes needed to reach a correct decision about where to spend engineering
effort.

## Plan

| | work | deliverable |
| --- | --- | --- |
| **A. Foundation** *(in flight)* | Finish the jam investigation: which axis the module wedges about, does the conclusion depend on the idealised pawls, and does the prediction hold on untested geometry | Evidence of advantage |
| **B. Tool** | Fold the four checks into one entry point that takes a reference cohort and a proposed change and returns a written prediction | Tool |
| **C. Prospective tests on the rack** | Predict before running, on configurations never run here. Each prediction recorded before the GPU starts | Evidence |
| **D. Second system** | One standard ground-based assembly task | Evidence, manuscript |
| **E. Manuscript** | Only once C and D have produced predictions that could have failed | Manuscript |

## Two things the plan needs from outside the repository

**Engineer feedback cannot be gathered by me.** Establishing that the problem
matters through practising engineers requires contacting people, which I will
not do without explicit direction and a named channel. What I can do instead,
and will unless told otherwise: ground the problem statement in published
assembly-qualification practice and prepare the material an engineer would be
asked to react to. The claim that the problem matters stays labelled as
unvalidated until someone actually reacts to it.

**The standard ground task has no shipped weights.** `Isaac-Forge-PegInsert-Direct-v0`
is registered in the pinned Isaac Lab and its config is present, but no
checkpoint ships with it, so a competent learned baseline costs a training
campaign of unknown length.

*Implementation choice, taken rather than escalated:* start with a **scripted**
insertion controller on the Forge task. The method qualifies a controller's
response to a changed part; it does not care how the controller was obtained,
and the rack's own guarded advance — the controller all of today's results are
about — is scripted too. Train a policy only if a result depends on the
controller being learned.

**The scope sentence this obliges, to be carried wherever the second-system
result appears:** demonstrating the procedure on the rack and on Forge shows it
applies across **two controller–fixture systems**. It says nothing about
learned skills in general. Any broader claim — that the procedure transfers
across policies, across training runs, or to controllers obtained differently —
needs evidence this project does not have and will not have from these two
systems.
