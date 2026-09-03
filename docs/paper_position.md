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
| parked-base kinematic gate | reachability | not applicable -- the loss is in the learned phases | **not a geometric result at all** |

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
   comparison.** 20/24 with the module pose from the simulator, 4/24 from the
   cameras, one term changed in one code path -- and the estimator is no worse on
   the episodes it loses (1.89/2.00/2.11 mm on winners against 2.29/5.99/1.98 on
   losers), which is what makes it a training-distribution result.
5. **Method notes, honestly small:** score each criterion against the failure it
   predicts rather than against the pooled rate, because at the design point 27
   of 29 failures are the controller's own terminal precision; and keep the
   instrument's own defects in the record.

## Venue

Unchanged in order, with one correction. The 2026-11-09 Frontiers deadline in
`PAPER_PLAN.md` could not be confirmed today; the live Frontiers in Robotics and
AI collection that matches is *Intelligent Manipulation of Space Robots:
Environmental Perception, Autonomous Decision-Making, and Dexterous Operations*,
and the Space Robotics section's own scope explicitly invites in-orbit servicing
under zero-g and changing illumination. **Check the collection's date before
planning around it.**

| Venue | Fit for the claim structure above | Blocker |
| --- | --- | --- |
| **Frontiers in Robotics and AI, Space Robotics** | best: the section's scope is the paper's domain, and a simulation-only requirements study is in scope | none beyond finishing the measurements |
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
