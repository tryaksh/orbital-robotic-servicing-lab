# What the extract skill was actually failing, and what the rack was doing to it

Written during the session that found it. It supersedes
`final_session_handoff.md`, which is kept because its reasoning is what
produced this state and because one of its conclusions is refuted here.

Read this before quoting any number about the grapple skills.

## The question it started from

`final_session_handoff.md` ends with: extract gained nothing from 900
epochs of fine-tuning — 1.4 points pooled, and −2.7 on its worst stage — while
the same treatment moved grasp 23 points. Its last word on it was "the obvious
next move is more epochs for extract specifically; nothing about this session
tested whether that helps."

**More epochs was not the answer, and the evidence that it was not is a 2×2 that
took eleven minutes.** Two policies, two module cross-sections, one seed, one
curriculum stage, everything else fixed:

| stage 0, seed 1070, 512 episodes | 450 × 130 × 20 mm | 450 × 160 × 35 mm |
| --- | ---: | ---: |
| extract v16w65 (trained on 160 × 35) | 76.95% | **99.02%** |
| extract v17m130 (retrained on 130 × 20) | 84.57% | 61.52% |

The unchanged policy scores 99.02% on the module it was built for and 76.95% on
the current one. **The cross-section costs 22 points.** Fine-tuning recovered
eight of them. There was never a training-length problem to solve.

## Why a thinner module is a harder pull

`BLADE_SIZE` went from 450 × 160 × 35 mm to 450 × 130 × 20 mm to buy clearance
for the *destination* seating, and it was measured there. What nobody measured is
what it did to the **source** bay, where the extraction happens.

`scripts/check_workcell_geometry.py` now derives it, in a second, with no
simulator:

| The module, inside its own channel | 160 × 35 mm | 130 × 20 mm |
| --- | ---: | ---: |
| Lateral half-gap | 0.750 mm | 15.750 mm |
| Vertical half-gap | 0.500 mm | 8.000 mm |
| Pitch at full engagement | 2.22 mrad | 35.56 mrad |
| **Roll** | **6.25 mrad** | **124.59 mrad** |

The extract task's own docstring says the rails constrain five of six motions.
For this section they no longer do, and the axis they gave up most of is roll —
which is also the axis a pair of flat pad normals cannot resist, because the
normals lie along it. While the channel held roll to six milliradians the grip
never had to; the moment it stopped, the grip inherited a job it has no geometry
for.

## Three things were wrong, and none of them was the policy

### 1. The rack could move the module further than the gripper could follow

`GUIDE_CENTER_OFFSET_Y` was documented as "1.5 mm total clearance around the
160 mm blade" and did not move when the blade stopped being 160 mm wide. The
pads are bolted to the arm. Anywhere the channel lets the module go, the grip has
to be able to follow, or the rack generates lost grips on its own.

A pair of flat pads 27 mm wide on a 30 mm pin keeps half its face on the pin
while the offset stays inside the **pin's own half-width**, 15.0 mm — the pad
widths cancel exactly. A module in the *corner* of a rectangular channel is
offset by `hypot(lateral, vertical)`, and the vertical gap is not available to
trade: it sets the 35.56 mrad hand-off requirement, which already binds. So

    lateral ≤ sqrt(15.0² − 8.0²) = 12.689 mm

and the rack was at **15.750**, with a channel corner of 17.66 mm against
0.90 mm on the module the skill was certified on. `GUIDE_CENTER_OFFSET_Y` is now
derived from that inequality. It costs nothing at the destination: the hand-off
requirement is `min(pitch, yaw)` and pitch binds, so yaw moving from 70.00 to
56.39 mrad changes nothing.

### 2. The grip criterion was a ball around a pose the load path sits 12 mm from

A tapered pin holds by **feeding**: the module lags the tool under load, thicker
material is drawn between the pads, and they are forced apart against the drive.
That is the entire reason this interface exists — a parallel jaw on a passive
feature holds about 6 N along the pull axis and the wedge holds 77 N from
geometry alone.

So the feed is the load path working, and it moves the tool along the pin.
Measured over 433 successful extractions, the pads come to rest **12.0 mm** from
the pin's drawing pose, in a band 0.8 mm wide.

Both grip criteria in this project measured the *distance* from that drawing
pose — 20 mm to count as captured, 30 mm to count as dropped. So 12 mm of each
was spent before the policy acted, isotropically, in whichever direction happened
to consume it. Measured on v17m130 at stage 0, **50 of 79 failures ended on that
ball with 79% of the error along the pin and the module 14.7 mm into a 525 mm
pull** — the pin seating at the first load transfer, reported as a dropped
module. On the section the policy was certified on, that happens once in 512.

The criterion is now three questions on the pin's own axes, and two of the three
are *tighter* than the ball by a factor of two:

| Axis | Bound | Where it comes from |
| --- | ---: | --- |
| Feeding along the pin | −42.0 mm | half the wedge stays under the pads |
| Backing out along the pin | +5.0 mm | the collar is a hard stop at zero |
| Across the pin | 15.0 mm | half the pad face stays on the pin |

The retention reward was charging the same fiction. Its position term has a 4 mm
free band about the drawing pose and the smallest reachable grip error under load
is 11.4 mm, so it was **saturated on every step of every episode**: a constant
charge for the interface holding, and a gradient pointing at a pose the collar is
in the way of. Lateral drift — which is what the failures are made of — had no
gradient at all. It does now.

### 3. The reset was generating episodes no policy could win

A joint-space noise box does not map to a bounded grip error, and extraction's is
0.020 rad at its widest stage. Measured on v17m130 at stage 2:

| | episodes | success |
| --- | ---: | ---: |
| Dead inside three control steps | 202 of 513 | **0.00%** |
| Survived the reset | 275 of 513 | **83.64%** |

The dead ones sit a median of 17.4 mm across the pin, 47.5 mm at the 95th
percentile, with the pin never fed: the pads closed on nothing and the policy
never acted. **Thirty-nine per cent of that stage's certification was a
measurement of the reset.**

The fix is not a narrower box. The chain refuses to hand a captured module to
this skill until the grip error is inside `WORKFLOW_HANDOVER_GRIP_M` — 10 mm,
the same tolerance the capture skill's own success predicate is written against
— so an episode starting outside it is not an extraction this workflow can ever
ask for. The reset now scales each draw's noise **vector** so the tool
displacement it induces stays inside that gate. The direction is untouched, so
the joint-space diversity the stages exist to provide is preserved; what changes
is that every drawn state is one a capture could have produced.

## What each of those was worth, on an unchanged policy

Every step was measured on the **same checkpoint** before anything was retrained,
so the policy cannot be credited with a better ruler or a better rack. One seed,
512 episodes a point. `evidence/extract_attribution.json`.

| Step | stage 0 | stage 2 |
| --- | ---: | ---: |
| As certified: 30 mm ball, 15.750 mm rack, unbounded reset | 84.57% | 60.55% |
| + the grip judged on the pin's axes | 82.03% | 39.30% |
| + the rack's lateral clearance derived | 83.82% | 44.83% |
| + the reset bounded by the hand-over gate | **91.21%** | **84.80%** |
| + 2,000 further epochs of PPO | 90.80% | 85.23% |

The criterion **costs** points, which is what a stricter ruler does, and it is
reported that way rather than netted off. The rack and the reset are worth the
rest.

## A fourth thing, which this session broke and then found

Moving `GUIDE_CENTER_OFFSET_Y` moved the rails and **nothing else**, and two
other pieces of the rack were positioned from it:

- `_FLARE_CENTER_Y`, the lateral lead-in, was an authored literal placed so its
  inner face met the rail face exactly at the mouth. The rails moved inboard
  3.061 mm and the flares stayed, so the lead-in became a step.
- `_RAMP_SURFACE_OFFSET`, the vertical lead-in, is the *difference* between
  those two — written that way on the stated reasoning that "the two lead-ins
  cannot drift apart" — so the ramps moved 3.061 mm the other way at the same
  moment.

Measured: the chain scored **0.00% over 32 episodes** with the module arriving
1.0 mm from the seated plane and 47.1 mrad square. Not a jam and not a miss —
**4.04 mm of lateral against a 2.5 mm tolerance**, against 1.85 mm on the same
chain the day before. Every episode ran out its clock without the seating
predicate ever holding through the settle.

`_FLARE_CENTER_Y` is now derived from the rail face it continues, so the ramp
offset falls out unchanged, and `tests/test_workcell_geometry.py` holds both.
With that one line the same seed goes to **93.75%**.

It is written down at this length because the mistake is the same class as the
one the session set out to fix: a constant positioned relative to something that
later moved, with nothing checking the relationship. The check is the fix.

## What the certifications say

Three curriculum stages on each of three held-out seeds, 500 episodes a point,
under the criterion, rack and reset above. `evidence/grapple_*_certification.json`.

| | pooled | stage 0 | stage 1 | stage 2 |
| --- | ---: | ---: | ---: | ---: |
| extract v17m130, the checkpoint being replaced | 87.78% | 91.53% | 89.01% | 82.80% |
| extract v18pin, 2000 further epochs | **87.75%** | 91.08% | 88.09% | 84.08% |
| grasp v7m130, unchanged, on the derived rack | 85.69% | 99.14% | 79.23% | 78.68% |

**Two thousand epochs bought nothing pooled and 1.3 points on the worst stage.**
That is the second time this session that retraining extract has come back empty
and the first time there was a reason to expect otherwise, so it is now a
measurement rather than an open question: the ceiling on this skill is not
training budget.

Where it *is*, the task's own curriculum says plainly. Extract's three stages
differ in how far the module starts out of the slot, which is to say in how much
of it the rails still hold:

| Stage | Engaged length | Pitch the rails admit | Success |
| --- | ---: | ---: | ---: |
| 0 | 450 mm | 35.6 mrad | 91.08% |
| 1 | 435 mm | 36.8 mrad | 88.09% |
| 2 | 358 mm | 44.7 mrad | 84.08% |

The rate falls monotonically with the freedom the channel leaves, on the same
policy, in the same episode budget, over a *shorter* pull. That is the same
mechanism the cross-section change caused, measured a second way.

Grasp is unchanged by the rack, which is the control that says the 3 mm move did
not simply make everything harder: 85.69% against a published 86.64%, inside the
Wilson interval on 4,507 episodes.

## What is not being claimed

- Nothing here says the retrained policy is better than v17m130 by the amount
  the headline moved. The control above is what separates them, and it is run
  under exactly the same criterion, rack and reset.
- The certifications this replaces were not wrong under their own criterion.
  `play.py --legacy_grip_ball_m 0.030` and `--legacy_unbounded_reset` reproduce
  it exactly, which is how the table above was taken.
- `GUIDE_CENTER_OFFSET_Y` is derived as the *largest* clearance the pads can
  follow, so the shipped module sits exactly on that bound with no margin. The
  window runs down to 5.738 mm and a value in the middle of it would leave a few
  millimetres on both sides. Not measured; named.
