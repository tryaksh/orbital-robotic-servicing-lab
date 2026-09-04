# Why the seating phase is a guarded advance and not a policy

A reviewer will ask why a paper about learned servicing hands the final,
hardest phase to a scripted controller. The answer is not that the policy was
not trained enough. It is that the quantity the seating phase is limited by does
not respond to the objective, and the scripted controller's advantage over the
policy is a single structural property that can be named, measured, and — the
point of this note — **derived before either controller is written**.

## What is measured

The learned insert skill has been certified fourteen times across ten
checkpoints. Two numbers matter.

| | on its own reset bank | on 96 recorded predecessor hand-offs |
| --- | ---: | ---: |
| `v24rack`, epoch 2100 | **36.77%** (1,103 / 3,000) | **0.00%** (0 / 96) |
| the guarded advance it must beat | — | 97.92% under the legacy criterion; 22/24 strict |

`evidence/grapple_insert_v24rack_certification.json`,
`evidence/workflow_robot_carried_insert_v24rack_chain_policy_certification.json`,
decision in `evidence/seating_controller_head_to_head.json`. The chain keeps the
scripted advance, unanimously on all three held-out seeds.

**0.00% here is not "a bit worse".** Terminal axial error is 1.35 m at the
median and terminal orientation 2.75 rad. The module is not seated short; it is
lost.

## Three things that were ruled out, each with its losing arm kept

1. **It is not the training budget.** 900 epochs moved extraction 1.4 points and
   2,000 more moved it 0.0, while three task corrections moved it 13 on an
   unchanged checkpoint (`evidence/extract_attribution.json`). The same ordering
   holds here: `v24rack` resumed `v23lock` for 700 epochs and plateaued from
   about epoch 1,800.
2. **It is not the objective.** The module ends at about 84.5 mrad against a
   52.4 mrad tolerance under three different objectives — a baseline time cost, a
   4x time cost trained to convergence, and a 7x orientation penalty — which land
   84.26, 84.61 and 84.58 mrad apart. **An angle that does not move when the
   reward is changed three ways is not the reward's to give.**
   `evidence/insert_attitude_diagnosis.json`.
3. **It is not the reset distribution, or the load path, or the action scaling.**
   All were corrected and measured: the reset bank was matched to the chain's
   hand-off, the form lock was put in the skill's load path, the action scaling
   was fixed, the controller was projected onto module-relative assembly state,
   and the hand-off station was curriculumed backwards from where the policy
   already succeeded. `v24` is 0/768 at reset stations 0–3 and succeeds at 4–8;
   the chain always hands over at station 0.

## What the limit actually is

A module held at attitude `theta` can engage at most `2c/theta` before it wedges,
where `c` is the channel's clearance per side. This is Whitney's classical
quasi-static peg-in-hole geometry applied to a long flat module in a rectangular
channel; it is not new and the paper cites it as prior art rather than claiming
it.

`evidence/insert_depth_is_attitude.json` shows the population split exactly along
that law:

```
stalled  (167)   96.8 mrad attitude   5.10 mm lateral   174.5 mm short
seated   ( 93)   46.9 mrad attitude   2.46 mm lateral     0.8 mm short
grip is 11.5 mm along the pin on both, to a tenth of a millimetre
```

`2c/theta` at 96.8 mrad is **260.6 mm** against this bay's vertical relieved
clearance and **323.9 mm** against its lateral one. The stalled episodes stop
174.5 mm short of a 436 mm stroke, so they travel **261.5 mm** -- the tight end
of that bracket.

The channel has two clearances and the law gives two answers.
Per side the relieved bay is 12.613 mm vertically and 15.678 mm laterally, so
`2c/theta` at 96.8 mrad is **260.6 mm** against the vertical gap and **323.9 mm**
against the lateral one. The quoted agreement uses the vertical figure. That is
the right rule -- a wedge forms at the first constraint reached, so the bound is
the minimum over the constraining directions -- but it is a rule that has to be
stated, because choosing the tighter of two numbers after seeing the answer is
not a prediction. It is also contingent: it holds if the stalled attitude is
about the axis that closes the vertical gap, which is consistent with the
geometry and is not separately shown. `insert_depth_is_attitude.json` says the
same in its own limitations -- the arithmetic brackets the observed travel
rather than predicting it to a millimetre -- and that sentence, not this
paragraph, is the one to defend.

Stated that way it is still the strongest number in this note: it is a
mechanistic agreement between a bound and a measured population, and unlike the
three-objective comparison it does not depend on how many training seeds produced
that population. They are not stopping short of a depth they could reach. They
are as deep as their own attitude permits.

So the requirement on the seating phase is a bound on attitude *maintained
through the stroke*: to travel the chain's 529 mm the module must be held inside
`2c/L`, about 42 mrad. The guarded advance delivers roughly 20 mrad. The policy
delivers 96.8.

## The one structural difference

The guarded advance and the policy differ in exactly one thing that is not a
hyperparameter: **the advance refuses to push a cocked module.** It steps the
axial target only while the deployed estimate is inside the entry envelope, and
holds otherwise. The policy has no such interlock. It can drive the module past
the depth its own attitude admits, and once it has, the remaining control steps
are spent in a state no action recovers.

That is a statement about the *problem*, not about reinforcement learning. A
seating controller for this interface must contain a geometric interlock, because
the interface's admissible depth is a function of the attitude the controller is
currently holding. A controller without one can wedge, and a wedged module ends
the changeout.

**The corrected-clearance sweep is independent evidence for this.** At 6 mm of
lateral clearance per side, `2c/theta` at the 46 mrad hand-over attitude says a
module should wedge at 261 mm of a 529 mm stroke. The guarded advance seats 36 of
64 anyway, with zero jams and 0.5 mm of terminal axial error, because it does not
carry the hand-over attitude through the stroke — it refuses to advance until the
module is square enough for the depth it is about to reach.

## What is still open, and what it would change

`Isaac-ZeroG-Blade-GrapplePin-InsertWedgeGated-v0` puts that interlock into the
*task* rather than into a controller: `mdp.wedged` ends the episode where
`2c/theta` says the module can go no further, so the credit for wedging lands on
the steps that cocked the module instead of being spread over a thousand steps of
pushing a part that cannot move. The observation width is unchanged, so it
resumes the frozen `v24rack` weights and the comparison is one policy under two
rules rather than two policies.

- **If it seats**, the paper reports a learned seating phase and the interlock as
  the task property that made it learnable — a stronger result than either the
  scripted controller or another failed retrain.
- **If it does not**, the paper reports that the interface bound survives even
  when the policy is told exactly where it is, and the scripted advance is
  retained on the argument above. That is also a result, and it is the one this
  note is written to support.

Either way the number is published beside the guarded advance's on the same three
held-out seeds, and the losing arm is kept.
