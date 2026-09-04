# What this paper may claim, after the 2026-09-03 literature check

> ## Corrections made overnight on 2026-09-03/04 -- read before writing
>
> An audit found six claims stated more strongly than their evidence supports.
> If a draft already contains any of these, it is wrong and this file is right.
>
> 1. **`2c/theta` does not predict the stall depth to within a millimetre.** It
>    gives a bracket, 260.6 to 323.9 mm, because the bay has two clearances.
>    **Lead claim 1 with the eight-point sweep instead**: fitting measured
>    attitude against relief gives `3.609 * relief + 6.217 mrad`, R^2 = 0.9998 --
>    the law's *form* confirmed, its coefficient 0.812 of the ideal, and the
>    bound exceeded by about 2.4% at the two tightest points.
> 2. **The requirement is computable before the *seating* policy is trained, not
>    "before any policy is trained".** The delivered attitude is measured from a
>    policy that already exists. The claim as first written is circular.
> 3. **"The estimator is no worse on the episodes it loses" is too strong.**
>    Those figures were medians; two losing episodes carry 154 and 355 mm. Say
>    that thirteen of twenty failures carry 2.38 mm or less, so most failures are
>    not explained by estimator error.
> 4. **The sight-line derivation computes line of sight, not readability.**
> 5. **`criterion_retention_v1.json` is retracted.** Do not quote a retention
>    statistic; the archives carry no hand-over value and the velocity version
>    was circular with the success predicate.
> 6. **Do not quote the grip signature as mechanism.** It comes from episodes
>    that never captured. The criterion is carried by the *rate of capture
>    failure*, 33 of 192 against nominal's 3.
>
> Two things also got stronger. Every A/B here is a paired design and reads
> better paired -- the lead-in guard and rack retention are both significant
> under McNemar and inconclusive under unpaired Wilson intervals -- and the
> perception interaction survives restatement as an odds ratio, 0.495
> [0.378, 0.650]. And open the results with the failure-mode decomposition of
> the nominal point: the chain delivers on 187 of 192 episodes, 97.4%, and the
> residual is terminal seating precision. A reader who meets 57% first has
> already formed an objection.

> ## 2026-09-04: the thesis, sharpened, and claim 2 rewritten
>
> **One thesis, two subsystems.** The paper reads as two papers sharing a
> workcell. It is one result twice: *the component is inside its specification
> and the system fails anyway, because the binding quantity belongs to the closed
> loop rather than to the part.* The arm is precise and its **delivered
> attitude** is what sizes the rack. The estimator is accurate to 2 mm and what
> binds is **whether any channel remains trustworthy**. Say this once, before
> either half, and claim 4 stops being a guest.
>
> **Claim 2 is much stronger than a clearance window.** No passive channel
> satisfies this interface: admitting the delivered attitude needs 10.350 mm per
> side, the seating gate accepts 2.500 mm, and in zero gravity nothing recentres
> a released module. Two thresholds follow -- `2*t_lat/L` = 11.11 mrad where
> passive alignment dies, and `2c/stroke` ~ 40 mrad where passive entry dies --
> and three regimes. This arm is at 46 mrad, in the third, and the cell
> implements exactly what the third demands. **Lead claim 2 with the regimes.**
>
> **Claim 3 gains a prescription.** The shipped rack is 3.897 mm past the tool's
> own upper bound, its source bay sits on the design point, and the dominant
> failure is lateral. A zero-relief arm is running to test whether correcting the
> rack as the tool prescribes removes the failure.
>
> **Claim 4's gate is closed:** 17/24 against 4/24, with neither single change
> distinguishable from the baseline. And the most transferable result in the
> paper is next to it -- certified skills compose under exact state and
> over-predict the camera chain by at least 72.8 points.

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
-- the textbook framing this document opens by forbidding. Pass two, from where
the losses stopped: *the bound predicts a failure that does not occur*. All forty
losses at `base_y_+6mm` time out in extraction, not in capture, holding the
module at 12.74 mm against the successes' 12.96 mm. They were gripped properly
and lost afterwards, so whatever ends them is not the grip.

**Use the offset this way and not the other way.** A near-identical offset on the
losses is evidence that the grip is *not* the mechanism, and that reading is
sound. The reverse -- a larger offset on the losses proving the grip *is* the
mechanism -- is not available, because an episode that never captures records the
distance the tool stopped at, and that is large by definition. The section axis
shows exactly this: its signature of +0.71 mm comes entirely from 33 episodes
that never gripped, while the seven that gripped and then lost it sit at 13.11 mm
against the successes' 13.13. There the criterion is carried by the *rate of
capture failure* instead -- 33 of 192 against nominal's 3 -- which is the phase
the criterion is about and is not circular. Pass three, from the phase counts and
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
the textbook framing this document opens by forbidding. Only knowing *where the
losses stopped* distinguishes a conservative bound from an inapplicable one.

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

**Say "before the seating policy is trained", never "before any policy is
trained", and the difference is not pedantic.** `ManipulatorPerformance` takes
three numbers. Two are design inputs: the seating tolerance is a property of the
interface, and the pad half-bearing offset is read off the gripper's collision
geometry. The third, the *delivered attitude*, is measured from robot-carried
transit runs -- which means it comes from a policy that already exists. A claim
that the requirement is computable before any policy is trained is circular, and
it is the kind of sentence a reviewer reads twice.

The true ordering is still the contribution, and it is worth stating as an
ordering rather than as a speed:

1. Characterise the arm. Whatever moves the module has to be measured, because
   its delivered pose is an outcome and not a specification. In this repository
   that is the robot-carried transit; in a programme it could be a datasheet, a
   scripted controller, or a policy from a previous build.
2. Derive the interface from that measurement, in closed form, on a CPU. The
   rack is a long-lead item and this is where its requirement is fixed.
3. Train the seating policy against the interface the derivation produced.

The novelty is that step 2 cannot be moved after step 3 -- the usual fix of
tightening the controller is unavailable -- and that step 1's output is a
distribution nobody specified. What the CPU buys is that step 2 costs seconds
instead of a training run, so the rack can be sized while the arm is still being
characterised.

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
   the mechanism, and **lead it with the sweep, not with the single point**.
   `workcell_geometry_check.json` varies the channel relief across eight points
   and records the attitude the module settles at; the fit is
   `attitude = 3.609 * relief + 6.217 mrad` with **R^2 = 0.9998**. That confirms
   the *form* `2c/L` asserts -- attitude linear in clearance -- while the
   coefficient is 0.812 of the law's and there is a 6.2 mrad intercept the law
   does not have. Say both. Read as an upper bound the law is respected at six of
   eight points and exceeded by about 2.4% at the two tightest, which is the end
   the design point sits at, and that is worth stating rather than averaging
   away.

   The single point stays as corroboration: the stalled population holds
   96.8 mrad and travels 261.5 mm, against `2c/theta` = 260.6 mm taken against
   the *tighter* of the channel's two clearances. Write it as the minimum over the constraining directions, state
   that the looser clearance gives 323.9 mm, and say the observed travel falls at
   the tight end of that bracket -- not that the law predicts it to a millimetre.
   Corroborated by three objectives landing
   0.4 mrad apart, with every losing arm kept and the single-seed limitation
   stated. Motivates everything.
2. **Therefore the interface must be derived from measured performance, before
   training.** `servicing_design.py` is that derivation, asserted against the
   certified geometry check over a 36-cell grid, and the direction of derivation
   is the separation from Niu et al.
3. **A simulator can say which of those derived bounds actually transfer, and
   the rule is whether the process corrects the bounded quantity.** The table
   above. Includes the two the simulator contradicts and why.

   **State the rule as an observation, not as a predictor, and do not try to
   dress it up.** An attempt to make it a statistic was made and retracted on
   2026-09-03: the plan was to rank episodes by the quantity a criterion bounds
   as the transit hands it over. The episode archives do not contain a hand-over
   value -- `_freeze` stores the row at the moment of judgement -- so the
   statistic measured a concurrent association, and for velocity it was circular
   with the success predicate, which includes linear and angular velocity and
   duly produced an AUC of 1.000. See `evidence/RETRACTED.md`.

   If a reviewer asks why the rule is not quantified, the honest answer is that
   quantifying it needs the hand-over state recorded per episode and the boundary
   arms re-run, and that we chose to report the rule as what it is rather than
   publish a number that reads back the outcome. That answer is stronger than a
   statistic nobody checked.

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
5. **Method notes, honestly small:**

   0. **Read the A/Bs as the paired designs they are.** Every comparison here is
      the same seeds, the same checkpoints and the same environments with one
      flag changed, and every one was first reported as two independent Wilson
      intervals. McNemar's exact test on the discordant pairs changes the
      conclusion for two of the three main arms -- the lead-in guard bound
      (10 gained, 2 lost, one-sided p = 0.019) and rack retention (5 gained,
      0 lost, p = 0.031) both have overlapping unpaired intervals and are
      significant paired. Report both, and state the fixed cohort with the
      number, because the pairing is an assumption about how the runs were made
      and not a property of the data. `scripts/compare_paired_arms.py`.
 score each criterion against the failure it
   predicts rather than against the pooled rate, because at the design point 27
   of 29 failures are the controller's own terminal precision; and keep the
   instrument's own defects in the record.

## The threat to validity a reviewer will raise first, and the answer

**"Your nominal configuration succeeds 57% of the time. What can a boundary
study on top of that possibly mean?"** It is the fair question and the paper has
to meet it in the results, not in the limitations.

The answer is that 57% is the wrong decomposition of the number. At the design
point, over 192 episodes on three held-out seeds:

| what happened | episodes | rate |
| --- | ---: | ---: |
| lost before delivery | 5 | 2.6% |
| lost in transit | 0 | 0.0% |
| jammed in the bay | 0 | 0.0% |
| arrived, seated, and held | 110 | 57.3% |
| arrived and missed the terminal gate | 77 | 40.1% |

**The serviceability chain -- releasing the module, carrying it across and
delivering it into the destination bay -- completes on 187 of 192 episodes,
97.4%.** Nothing jams. What fails is the last few millimetres: of the 187 that
arrive, 110 seat, 58.8%. That residual is the seating controller's terminal
precision, which is a control problem in a bay the module has already entered,
and it is the phase this paper reports separately for exactly that reason.

So the boundary study is not measuring perturbations against a coin flip. It is
measuring them against a delivery process that works 97 times in 100, using a
partition that scores each criterion on the failure it predicts rather than on a
pooled rate dominated by a phase no criterion claims to govern. **Lead the
results section with this decomposition.** A reader who meets 57% before they
meet its parts has already formed the objection, and every later number is read
through it.

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
   attitude is 260.6 mm against the vertical clearance and 323.9 mm against the
   lateral one. **The observed travel sits at the tight end of the bracket the
   law gives.** That is a mechanistic agreement between a closed-form bound and a
   measured population, and it does not depend on how many seeds produced the
   population -- but it is a bracket, and the paper must not narrow it to a
   millimetre by quoting only the clearance that happens to match.

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

   So the paper leads claim 1 with the mechanism, uses the three objectives as
   corroboration that the angle does not respond to reward shaping, and states
   the single-seed limitation in the same paragraph rather than hoping nobody
   asks. The GPU that nine runs would have cost goes to claim 4 instead.
3. **The RGB cohort**, because claim 4 is the one a space-robotics reviewer will
   press hardest and 4/24 with no mitigation reads as a broken system rather than
   as a measured cost.
4. Not more insertion checkpoints. `docs/seating_controller.md` is the argument,
   and the wedge-gated run is the last one worth spending.
