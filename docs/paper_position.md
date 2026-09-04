# What this paper may claim, after the 2026-09-03 literature check

`PAPER_PLAN.md` is frozen and carries the 2026-09-02 review. This file is the
second pass, run after the boundary work of 2026-09-03 changed the result set,
and it exists because that work produced a framing that **loses to prior art if
stated the obvious way**. Recording the separation before drafting is cheaper
than discovering it in review.

## The framing that must not be used

**"The closed-form bounds are worst-case and therefore conservative relative to
what the simulator measures."** That is worst-case versus statistical tolerance
analysis, it is in every mechanical design textbook, and it has been since the
1960s. Worst-case stackup guarantees every assembly works and is conservative
because the chance of every dimension hitting its limit at once is vanishingly
small; RSS is the statistical alternative. A reviewer with a mechanical
background will read the sentence above as a rediscovery, and they will be right.

Three more things must be cited rather than claimed:

* **`2c/theta`, wedging and jamming** are Whitney's quasi-static peg-in-hole
  analysis. The project already knew this and `docs/seating_controller.md` says
  so; it is repeated here because the temptation grows as the result gets
  stronger.
* **Sizing a lead-in from initial misalignment** is chamfer-crossing analysis,
  also classical. `servicing_design.requires_a_correcting_lead_in` is that
  analysis used as a gate, not a new law.
* **Design-for-serviceability guidance exists and is quantitative.** NASA's
  *Design Guidelines for Remotely Maintainable Equipment* (N89-19885) and
  NASA-STD-3001's maintainability requirements already prescribe alignment
  guides, soft-dock features and capture misalignment tolerances for ORUs. The
  contribution cannot be "somebody should derive serviceability requirements".

## The separation from the closest learning work

**Niu et al., *Tolerance-Guided Policy Learning for Adaptable and Transferrable
Delicate Industrial Insertion* (CoRL 2021, arXiv 2108.02303)** is the nearest
neighbour and it runs in the opposite direction. They compute a tolerance model
from CAD and *feed it to the policy* as a task embedding, so that one policy
transfers across workpieces. Here the policy is measured and the *tolerance is
the output*. Stated as one sentence in the introduction, that contrast does more
work than a page of positioning:

> Tolerance-guided policy learning conditions a controller on a known interface.
> This paper measures a controller and emits the interface.

The other three neighbours are unchanged from the 2026-09-02 review and are
cited, not competed with: NVIDIA SRL's assembly line (Factory, IndustReal,
FORGE) for insertion rates, *Residual RL for Precise Assembly* (arXiv 2407.16677)
for what a properly distilled vision transfer costs, and Lee et al. on skill
chaining and terminal-state regularization for the hand-off failure.

## The thesis that survives, and it is sharper than the one it replaces

**A closed-form design bound transfers to a policy-driven assembly process
exactly when the quantity it bounds is one the process does not correct.**

That is the pattern in the measurements, and it is not a statement about
worst-case versus statistical anything:

| bound | quantity it bounds | corrected by the process? | outcome |
| --- | --- | --- | --- |
| `2c/L` on a seated module | terminal attitude | no -- the module is already home | **confirmed**, 0.87 to 1.02 of the law over eight points |
| pad half-bearing offset | how far a pad has slid off the pin | no -- the pull happens where the module sits | **confirmed by mechanism**: the only two points the criterion flags are the only two whose failing episodes sit further off the pin |
| upper clearance bound | attitude a *resting* module may reach | no | **confirmed**: 16 mm/side loses 0.203 before delivery against nominal's 0.031, Wilson-separated |
| lower clearance bound | attitude an *entering* module carries | **yes** -- the flare and the guarded advance square it during the stroke | **contradicted**: 6 mm/side scores 56.25% against nominal's 54.69% |
| rail stop error, via the pad bearing bound | how far a pad has *slid off* the pin -- a static quantity | not corrected | **right axis, wrong criterion.** The bound is 1.624 mm, the cliff is between 4 and 6 mm, and the failing episodes are extracted and still gripped: they finish 2 mm from the extracted plane at a normal 12.5 mm tool-to-pin offset and fail the *settling* condition, carrying 16 to 30 mm/s against a 14.29 mm/s limit. The failure is dynamic; the bound is static |

**The last row is the one worth writing the paper for**, and it took three
passes to state correctly, which is itself the argument for the mechanism check.

Pass one, from the ladder alone: *the bound is conservative by a factor of three*
-- the textbook framing this document opens by forbidding. Pass two, from the
grip signature: *the bound predicts a failure that does not occur*, because the
lost modules are held at a normal offset. Pass three, from the phase counts and
the settling limits: **the bound is static and the failure is dynamic.** The
module leaves the bay carrying residual motion that zero gravity never removes,
and the extraction predicate, which requires it to be *settled*, never fires.
Why it carries that motion is a hypothesis and must be written as one -- a pull
off the bay's centre line applies a moment to two flat pads on a pin, the one
thing this interface cannot resist -- because the episode rows record a speed and
not a direction. `--handoff_trace` records the vector and one traced rung settles
it.

That is the paper's most useful negative result, because the correction is
nameable: an axis whose failure is a settling condition needs a bound on the
residual velocity an off-axis pull imparts, and that is a dynamic quantity a
static geometric tool does not compute. **A design tool has to say where it stops
applying, and what would have to be added.** This is the measurement that says
both.

Recording the near miss, because it is the reason the mechanism check exists: the
ladder alone reads as "the bound is conservative by a factor of three", which is
the textbook framing this document opens by forbidding. Only the grip signature
distinguishes a conservative bound from an inapplicable one.

The lower clearance bound was not wrong arithmetic. It was the right law applied
to the wrong state. That is a usable rule for a designer -- *check whether the
assembly process corrects the quantity your bound constrains, and if it does,
your bound sizes the corrector, not the clearance* -- and it is what turned the
same law into `requires_a_correcting_lead_in`.

## Why the bound has to be computed before training, which is the novel coupling

In classical assembly design the inserting device's delivered pose is a
specification: you buy a machine with a stated repeatability and you stack it. A
learned policy's delivered pose is **an outcome you can only measure**, and this
repository has the measurement that makes it binding: the terminal attitude sits
at 84.26, 84.61 and 84.58 mrad under three different objectives -- a baseline
time cost, a 4x time cost trained to convergence, and a 7x orientation penalty.
Three objectives, 0.4 mrad apart.

So the usual iteration -- tighten the controller until it meets the interface --
is not available. The angle does not respond to the reward. The interface has to
be sized for what the policy will deliver, and that has to happen before the GPU
is spent, because afterwards there is no knob. **That is the coupling the paper
is about**, and no search run today returned it.

## A correction to this document, made the same day it was written

**The recommendation to stop spending GPU on the learned seating phase was
wrong, and the owner was right to push back on it.** It rested on the claim that
the terminal attitude is set by the interface rather than by the learner, which
three objectives 0.4 mrad apart appear to support. What that argument never
checked is whether the policy could *perceive* the quantity it was being scored
on.

It could not. The seating policy's observation vector is joint positions and
velocities, tool pose, grip error, gripper state, module velocity, previous
action and goal error. **There is no contact force in it, and the grapple-pin
scene has no contact sensor at all.** Ten checkpoints of contact-rich assembly,
done blind.

`BladeContactWrenchObservation` has existed in this repository since the
force-limited insertion work. Its own docstring says "the missing half of force
control", explains that "two force-penalty strengths failed to change measured
contact load because the policy was asked to regulate a quantity absent from its
observations", and cites FORGE. It was never wired to the skill the chain runs.

**The literature says this is the ingredient class, not a detail.** FORGE
(arXiv 2408.04587) feeds the policy a noisy end-effector force estimate, adds a
force threshold it is conditioned on, and randomizes controller gains, action
scale, friction and a joint-friction dead-zone. It transfers contact-rich
insertion zero-shot **while tolerating up to 5 mm of fixed-part pose error and
2.5 mm of position noise.** This project's estimator is accurate to about 2 mm
and its chain collapses on camera-derived state. A system that tolerates more
pose error than ours has, using ingredients ours lacks, is the strongest possible
evidence that the ceiling here is the recipe rather than the physics.

Three of those ingredients are missing here, and they are separable:

1. **Force observation.** Being tested now as
   `Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0`, one change, from scratch.
2. **A compliant action space.** The seating policy commands pose deltas through
   differential IK -- a stiff position controller. FORGE commands a target into a
   spring-damper with randomized stiffness, so a jammed part pushes back instead
   of being pushed harder. This is the second arm if the first is not enough.
3. **Dynamics randomization during training.** Friction, controller gains, action
   scale. FORGE's own ablation is instructive and matches this project's [T5]
   prediction: removing it *raises* nominal success to 0.91 and destroys
   robustness to controller gains at deployment. A flatter curve at a lower peak
   is the trade, and it should be reported as one.

**What this does to the claims.** Claim 1 needs rewording and survives in a
better form. "The delivered attitude is not the reward's to give" is supported;
"no policy can deliver it" was never measured and must not be implied. The honest
statement is that the attitude does not respond to reward shaping *for a policy
that cannot sense contact*, which is a statement about observability rather than
about the interface -- and `2c/theta` predicting the achieved depth to within a
millimetre still stands regardless of which controller is driving.

If the force-feedback policy seats, the paper reports a learned seating phase and
the reason ten previous attempts failed, which is a considerably better result
than the negative one. If it does not, `docs/seating_controller.md` is a far
stronger argument for having been tested against the recipe the field agrees on.

## The theme that ties the results together, found twice independently

**The same class of error appears in both halves of this project: a correct
quantity used to answer the wrong question.** It is worth stating explicitly,
because it is what makes a collection of measurements into a paper.

* **In the design model.** `2c/L` correctly bounds the attitude a *seated*
  module can hold, and it was being used to size the clearance an *entering*
  module needs. The entering module does not carry its hand-over attitude --
  something straightens it on the way in -- so the bound was right about a state
  the module is never in at that moment. Correcting it turned a clearance floor
  into a lead-in requirement, and removing the lead-in produced the predicted
  failure.

* **In the deployed controller.** `FIDUCIAL_GUARDED_ORIENTATION_TOLERANCE_RAD`
  correctly bounds *whether the pose estimate is trustworthy* -- its own comment
  derives it from the certified estimator errors -- and it was being used to
  decide *whether the module may enter the bay*. Those are different questions
  and the bay's own funnel catches five times the angle. Correcting it was worth
  33 percentage points with nothing retrained.

Neither was a tolerance that had been tuned loose, and neither was a bug in the
arithmetic. Both were quantities that were individually right, attached to
decisions they were not about. **That is a failure mode a reviewer can carry away
and apply to their own system**, and it is more transferable than either
individual number.

It also gives the validation protocol its justification. Scoring a criterion
against the *failure mode it predicts* rather than against a pooled success rate
is precisely how you catch a bound that is attached to the wrong decision: the
rate moves for a dozen reasons, the predicted mechanism moves for one.

## Claim structure, in the order the paper should make them

1. **A learned manipulator's delivered pose is not a design variable.** Led by
   the mechanism: the stalled population holds 96.8 mrad and travels 261.5 mm,
   against `2c/theta` = 260.6 mm -- the closed-form bound predicts the achieved
   depth to within a millimetre. Corroborated by three objectives landing
   0.4 mrad apart, with every losing arm kept and the single-seed limitation
   stated. Motivates everything.
2. **Therefore the interface must be derived from measured performance, before
   training.** `servicing_design.py` is that derivation, asserted against the
   certified geometry check over a 36-cell grid, and the direction of derivation
   is the separation from Niu et al.
3. **A simulator can say which of those derived bounds actually transfer, and the
   rule is whether the process corrects the bounded quantity.** The table above.
   Includes the two the simulator contradicts and why.
4. **The cost of perception, measured as a substitution rather than a
   comparison, then attributed, then removed.** Three measurements, and the
   middle one is the contribution.

   *Substituted:* 20/24 with the module pose from the simulator, 4/24 from the
   cameras, one term changed in one code path -- and the estimator is no worse on
   the episodes it loses (1.89/2.00/2.11 mm on winners against 2.29/5.99/1.98 on
   losers), which is what makes it a training-distribution result rather than a
   sensor one.

   *Attributed:* on an unchanged checkpoint, noising the pose channels alone
   costs 8.33 points and the velocity channel alone 10.21, while both together
   cost 41.15. **The interaction is 22.61 points, larger than the sum of the
   parts.** Neither channel is the problem. A policy absorbs a noisy pose while
   its velocity is true and a noisy velocity while its pose is true; with both it
   has no reliable channel and nothing local repairs that. This is the result
   that makes the mitigation non-obvious -- it says why a filter, a longer
   differencing window or a better camera would each have failed -- and it is the
   part of claim 4 that is ours rather than the literature's.

   *Removed:* the fine-tune on both channels at once takes seed 4070 from 2/8 to
   5/8 at eight hundred of two thousand epochs, with the module-loss failure mode
   gone. Pooled over three seeds, queued.

   Cite *Residual RL for Precise Assembly* (arXiv 2407.16677) as the properly
   distilled baseline -- 98% teacher to 73% student -- and do not compete with it.
   The claim here is the attribution, not the rate.
5. **Method notes, honestly small:** score each criterion against the failure it
   predicts rather than against the pooled rate, because at the design point 27
   of 29 failures are the controller's own terminal precision; and keep the
   instrument's own defects in the record.

## Venue

Unchanged in order, and the date is now confirmed rather than assumed.

**The matching collection is open and its manuscript deadline is 28 February
2027**: *Intelligent Manipulation of Space Robots: Environmental Perception,
Autonomous Decision-Making, and Dexterous Operations*, hosted by Frontiers in
Robotics and AI's Space Robotics section with Frontiers in Space Technologies
participating. Checked 2026-09-03; the page states it is currently accepting
articles.

Two consequences, and both change how the remaining work should be planned.

**The binding date is the owner's, not the venue's: a project ready to submit by
early November 2026**, set 2026-09-03. The collection stays open to 2027-02-28,
so the venue is not the constraint and a slip costs nothing externally -- but the
work is planned against November, which is about two months.

That is comfortable for what is queued and tight for anything new. It rules out
a second training campaign of the size of the seed spreads, and it means any
further experiment has to earn its place against the four claims already
standing rather than opening a fifth.

**The collection's own framing names perception**, which moves claim 4 from
liability to fit. A reviewer for a topic about "environmental perception,
autonomous decision-making and dexterous operations" is the right reader for a
measured 67-point perception cost with its mitigation, rather than one who reads
4/24 as a broken system.

| Venue | Fit for the claim structure above | Blocker |
| --- | --- | --- |
| **Frontiers in Robotics and AI, Space Robotics** | best, and confirmed open to 2027-02-28: the section's scope is the paper's domain, the collection names perception, and a simulation-only requirements study is in scope | none beyond finishing the measurements |
| **Acta Astronautica** | strong for the journal version; a design/requirements contribution reads naturally there | wants the sim-to-real account written out, which `docs/sim_to_real.md` now is |
| **IEEE T-ASE** | possible only if claim 3 is presented as the methods contribution and evaluated as a method | would want the protocol applied to a second system, which this project does not have |
| i-SAIRAS / ASTRA / IEEE Aerospace | good conference route, and the fastest | length caps force choosing between claims 3 and 4 |

**Recommendation: Frontiers first, Acta Astronautica as the fallback.** Do not
submit to T-ASE on the current evidence -- the methods contribution is one
system, and a methods venue will ask for two.

## What would most improve the paper from here, in order

1. **The flare-removal run.** It decides whether the corrector in claim 3's fifth
   row is the geometry or the controller, and the rule is much stronger if it is
   the geometry, because then the closed form sizes a *part*.
2. **The seed spreads for the published *skill rates*** -- grasp and extract,
   both running. Not for claim 1, and that is a decision rather than an omission.

   Claim 1's obvious weakness is that three objectives at one training seed is
   three samples of one seed, and the obvious fix is to retrain each objective at
   three seeds. That fix is not available: the three arms are resume lineages
   whose exact provenance cannot be reproduced -- `v20chain` ends at epoch 1,400
   on a lineage that is not the one `run_insert_chain.sh` documents -- so nine
   new runs would measure the seed spread of a *different* procedure and answer
   nothing.

   **The claim has better evidence anyway, and it is one number rather than a
   coincidence of three.** The stalled population holds 96.8 mrad and stops
   174.5 mm short of a 436 mm stroke, so it travels 261.5 mm; `2c/theta` at that
   attitude in that channel is 260.6 mm. **The law predicts the achieved depth to
   within a millimetre.** That is a mechanistic agreement between a closed-form
   bound and a measured population, and it does not depend on how many seeds
   produced the population.

   So the paper leads claim 1 with the mechanism, uses the three objectives as
   corroboration that the angle does not respond to reward shaping, and states
   the single-seed limitation in the same paragraph rather than hoping nobody
   asks. The GPU that nine runs would have cost goes to claim 4 instead.
3. **The RGB cohort**, because claim 4 is the one a space-robotics reviewer will
   press hardest and 4/24 with no mitigation reads as a broken system rather than
   as a measured cost.
4. Not more insertion checkpoints. `docs/seating_controller.md` is the argument,
   and the wedge-gated run is the last one worth spending.
