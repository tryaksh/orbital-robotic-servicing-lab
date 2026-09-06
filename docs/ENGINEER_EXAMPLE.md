# One worked case, for a practising engineer to judge

Two minutes. The question is whether you have this problem, not whether the
method is clever.

## The situation

A robot cell moves a rectangular part from one bay of a rack to another and
seats it. The part is 130 x 20 mm in section, 450 mm long. The controller is
fixed and works: 187 of 192 attempts succeed.

Someone now wants to run a **different part** through the same cell. Two
candidates:

* **A** — 140 x 20 mm. Ten millimetres wider, same thickness.
* **B** — 130 x 26 mm. Same width, six millimetres thicker.

You have to decide where to spend engineering effort before you build fixtures
or book cell time. The ordinary answer is to run both and count. Each costs a
few hundred attempts.

## What was available beforehand

Three configurations had already been run for other reasons, and none of them
was either candidate:

| already run | what it varied against the reference |
| --- | --- |
| 120 x 16 mm part | narrower **and** thinner |
| 140 x 26 mm part | wider **and** thicker |
| narrowed channel | fixture clearance only |

Note the trap: the two part cohorts each moved width *and* thickness together.
Nothing on hand separates them. That is normal — campaigns vary a part, not a
dimension.

## The advance prediction

Made before either candidate was run, recorded in git first.

| | candidate A (wider) | candidate B (thicker) |
| --- | --- | --- |
| where it fails | insertion | insertion |
| how often | 13.0% of attempts | 6.25% of attempts |
| grasping | unchanged, 1.6% | unchanged, 1.6% |
| pulling clear | unchanged, 1.0% | unchanged, 1.0% |

The reasoning, in one line each. **A** halves the clearance the part has to fit
into, so insertion failures roughly double against the narrowed-channel cohort.
**B** does not touch that clearance at all, so insertion should look exactly
like the narrowed channel — and the procedure explicitly declines to say
anything about the thickness change, because nothing on hand varied thickness
by itself.

**The engineer running the tool and the engineer who knew the cell disagreed.**
The person who had spent a day inside this workcell predicted **B** would lose
parts while *pulling them clear*, reasoning from the 140 x 26 cohort where most
losses were extraction failures. The tool refused to make that inference,
because that cohort also changed the width. Both predictions are on record.

## What happened

*(This section is filled in when the runs land. It is deliberately left empty in
the version circulated for comment, so the reader judges the setup and not the
outcome.)*

## The decision it supports

If the prediction holds, the engineer does not run both candidates. They run
**B**, because it is the one the procedure could not fully answer, and its
result also de-confounds thickness for every future part. **A** needs no cell
time at all: its failure mode and rate were readable from a cohort that already
existed for another purpose.

That is the claim worth your judgement — not the accuracy, which one case cannot
establish, but whether **"which of these two changes do I actually need to
test?"** is a question you face, and whether an answer of this shape would
change what you do.

## What this is not

It does not tell you the part will work. It tells you where it will fail and how
often, from configurations you already ran. It is silent on any dimension your
existing cohorts never varied, and it says so rather than guessing — on the case
above it declined the thickness question outright.

All results are simulated. No hardware.

## The three things worth telling us

1. Do you make this decision, and how do you make it now?
2. Is "which phase fails, and at what rate" the output you would want, or would
   you want something else?
3. Would a refusal — "your existing runs cannot answer this, run X" — be useful
   or annoying?
