# Decision report: did the procedure predict unseen failures better than the simple alternative?

Two configurations never run in this repository, 192 episodes each, three
predictors whose claims and scoring rule were both committed to git before any
episode existed. `evidence/prediction_scorecard_v1.json`,
`docs/PREREGISTRATION_jam_prediction.md`.

## What happened

Reference: 187/192, capture 0.0156, extract 0.0104, insert 0.

| | `wider_140x20` | `thicker_130x26` |
| --- | --- | --- |
| delivery | 0.9844 | **0.9115** |
| capture | 0.0156 → 0 | 0.0156 → **0.0260** |
| extract | 0.0104 → 0.0104 | 0.0104 → **0.0625** |
| insert | 0 → 0.0052 | 0 → **0.0469** |
| phase that rose most | insert, **on one episode** | **extract** |

`wider_140x20` did not meaningfully change. `thicker_130x26` did: delivery fell
six points and extraction failures rose six-fold.

## The scorecard

| predictor | `140x20` S1 / S2 / S3 | `130x26` S1 / S2 / S3 |
| --- | --- | --- |
| `2c/theta` clearance rule | – / **no** / – | – / **no** / – |
| the tool | yes\* / **no** / 0.1250 | **no** / yes / **0.0156** |
| hand prediction | yes\* / **no** / 0.2500 | **yes** / **yes** / 0.0265 |

\* S1 credit on `140x20` rests on a single episode inside the reference's own
interval and is not credit. The scorer says so itself.

## Answer to the first question

**Partly, and not in the way I expected.**

On `wider_140x20` **every predictor failed**, including the simple alternative.
All three said it would be worse; it was unchanged. That is a clean, shared
failure and it is the more interesting half of the result — see below.

On `thicker_130x26` the ranking is clear and the simple alternative is last. The
clearance rule predicted no change at all, and delivery fell six points. The
tool's *rate* on insert was the most accurate number any predictor produced
(0.0625 against 0.0469 observed, error 0.0156) but it named the wrong phase. The
hand prediction named the right phase, got the direction right, and put delivery
at 0.85–0.92 against an observed **0.9115** — inside its stated range.

So the method's **reasoning** beat the simple alternative on the discriminating
case. The **tool's implementation of it did not**, and the reason is specific
and fixable.

## The refusal rule is too strict, and that was pre-registered as the finding

The tool refused to say anything about thickness because the corpus never varied
thickness alone — the two section cohorts moved width and thickness together. My
hand prediction ignored that and attributed `140x26`'s extraction losses to
thickness anyway. **The hand prediction was right.**

The pre-registration named this outcome in advance: *"If P3 holds while the tool
refused, the confounded corpus carries usable signal the procedure is throwing
away, and the refusal rule is too strict — a concrete, fixable finding."*

That is what happened. The correct repair is not to abandon the refusal but to
soften it: when a dimension is confounded, report the **joint** sensitivity with
its confound named, rather than declining outright. `140x26` moved width and
thickness together and lost episodes to extraction; the honest output is "a
part that is wider-and-thicker costs extraction, and this corpus cannot say
which of the two is responsible" — which would have flagged `130x26` as risky
while still refusing to attribute the cause.

## Why everyone failed on the wider module

Diagnosis, after the fact. Peak yaw during insertion:

| clearance/side | median | p95 | reach the wedging band | jams |
| ---: | ---: | ---: | ---: | ---: |
| 15.678 mm | 56.02 | 72.88 | 0/187 | 0 |
| 11.065 mm | 56.12 | **87.08** | 13/187 | 12 |
| 6.065 mm | 57.14 | 60.24 | 0/190 | 1 |

**Clearance is not a monotone tolerance.** It both permits the wedge and
supplies the room the wedge needs, so jam risk peaks at intermediate clearance.
Every predictor assumed monotonicity, which is why every predictor was wrong.

## Answer to the second question

**Yes, and the decision survived the wrong numbers.**

The decision the method supported was: *do not spend cell time on A, run B,
because B is the one the procedure cannot answer and its result also
de-confounds thickness for every future part.* Observed: A changed nothing and
would have been wasted cell time; B is where the failure was. **The method got
the decision right on both configurations while getting the rate wrong on one
and the phase wrong on the other.**

That is the honest form of the claim, and it is narrower than "the method
predicts failures". It predicts *where to look*, and on this test that was worth
one of two campaigns.

## Recommended next action, and only one

**Soften the refusal to a joint-sensitivity report, then re-score these same two
configurations against it.** No new GPU: both cohorts now exist, the corpus
exists, and the change is to `change_prediction.py` alone. If the softened rule
would have flagged `130x26` while still declining to attribute the cause, the
procedure recovers the one prediction it missed without acquiring the licence to
guess — and that is a result about the method rather than about this rack.

If it would not have, the phase-attribution claim rests on human reading of a
confounded corpus and cannot be automated, which is a finding worth publishing
plainly and a reason to narrow the contribution to the diagnosis tooling.

## What is proven, and what is not

**Proven.** The jam is a yaw. Clearance is non-monotonic for it. Thickness costs
extraction and delivery. Both configurations were predicted in advance from
committed claims under a committed scoring rule, and the reference reproduces
bit-identically.

**Not proven.** That the procedure beats a clearance rule in general — two
configurations is two cases, and on one of them every predictor failed. That the
refusal rule can be softened without becoming a guess: that is the next
experiment, not a conclusion. And nothing here has run on hardware.
