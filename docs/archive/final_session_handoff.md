# Where this project actually stands

> **Superseded on 2026-08-24 by
> [`grip_criterion_handoff.md`](grip_criterion_handoff.md).** One conclusion
> below is refuted there and should not be carried forward: this document ends
> by suggesting that "the obvious next move is more epochs for extract
> specifically". It is not. Extract's problem was that the module's
> cross-section had changed underneath it, that the criterion judging it charged
> the pin's own load path as a dropped module, and that its reset was starting
> 39% of its worst stage with the pads closed on nothing. Two thousand further
> epochs, run against the corrected task, moved the pooled rate by 0.0 points.
> Everything else here stands.

Written at the end of the working session before that. It supersedes
`next_session_handoff.md`, which is kept because the reasoning in it is what
produced this state.

Read this before quoting any number from this repository.

## The one-sentence version

A six-axis robot grasps a compute module out of one bay, carries it to another,
drives it home and lets go after a settled re-check — the whole job, end to end,
with nothing but the robot holding the module — and it does that **96.88% of the
time** over 96 episodes and three held-out seeds, Wilson 95% interval
**[91.2%, 98.9%]**.

It scored **31.25%** at the start of the session. Both numbers are in
`evidence/`, because a before is what makes an after mean anything.

## The chain

Five phases, in the plain words for them:

| | What runs it | Why that side of the line |
| --- | --- | --- |
| **Grasp** | RL policy | The pin is a wedge; a closing pad on a wedge in zero gravity thrusts the free module along the pull axis before it grips. Whether the grip takes depends on contact transients across 0.1 s. Nothing models that. |
| **Extract** | RL policy | Same contact, under load, with the rails still constraining five of six motions. |
| **Transit** | Solved inverse kinematics, commanded as actuator targets | Five pose-to-pose moves through a workcell whose kinematics are solved in closed form and agree with the simulator to 0.006 mm. There is no uncertainty here for a policy to be robust to. |
| **Insert** | Guarded advance on the deployed RGB-D estimate | The bounded axial target moves only while perception says the module is inside the bay's envelope, so a lost fiducial stops the insertion instead of continuing blind. |
| **Back off** | Scripted, after the settled re-check | The hand opens only once every seating condition has been re-checked over 0.70 s. |

`../service_interface_spec.md` §10 states that split as a requirement and lists
the gates it is held to, so it is checkable rather than described.

## What was wrong, and is not any more

Six defects, each of which had been reported as a result at some point.

**A per-environment attitude authority was clamped without its row axis.** Any
robot-carried transit with more than one — or exactly three — environments raised
mid-leg. That is why every number about this chain was n = 1 and why
`run_robot_carried.sh certify` had never produced a pooled report. One
`unsqueeze`.

**The first transit leg was stamped with entry step 0.** A leg times out when
`step - entered` passes its budget, and the grasp and extraction take about 400
control steps between them, which is exactly the retreat's budget — so the
retreat was ended by its own timeout on its first step in every run this branch
ever recorded. The squaring leg that follows has the same axial target, so it
silently did both jobs at once: it squared the module while its nose was still
between the lead-in flares, which is the one thing the retreat exists to prevent.

**The hand-off was gated on the seated success check.** 52.36 mrad is the test
for whether an installed module counts as installed. It is not an entry
requirement. The gate is now `2c/l` taken at the **lead-in** gap — 35.56 mrad,
derived by `scripts/check_workcell_geometry.py` in a second with no simulator —
which is deliberately conservative: it is the tightest surface on the path at the
longest engagement, so a module that meets it clears every surface at every
depth. It is not the largest attitude that can seat, and §9.8 of the spec is
careful about the difference. Handed over at 52.4 the module wedges 53 mm short,
and the guarded advance's stall detector correctly refuses to push.

**The seating stroke was performed by the transit and reported as the
insertion.** The last transit leg drove the module the full 446 mm into the bay
and the phase labelled "insert" advanced it 0.7 mm in six control steps, while a
loaded insert policy was never asked for an action. The insertion phase now
receives the module at the mouth and drives the whole stroke, so the deployed
estimate is in the loop for 446 mm rather than for the last millimetre.

**The seating controller was using the insert policy's action scale.** Given the
whole stroke at 8 mm and 20 mrad per control step, the follower walks the module
out of its own envelope faster than its 1 mm axial step walks it in: measured, 17
mm of advance and then 4,557 control steps held. The scale belongs to the
controller, not to the phase, and the guarded advance now carries the transit's
2 mm and 8 mrad.

**The last transit leg aimed at the wrong pose for whichever controller was
receiving.** Where it should aim depends on the receiver: the guarded advance has
no reset pose and takes the module at the mouth, the learned insert policy has
exactly one, 128 mm inside the slot. The flag that chose between them was
computed at import time from the *default* `MATING_MODE`, which the command line
overwrites much later — so it was true whatever was asked for. Handed the module
at the mouth, the policy dragged it to x = −0.486 m and 1.52 rad off square,
which says nothing about the policy and everything about where it was started.
Given its own pose it reaches 48 mm short at 54.9 mrad, which is a number worth
comparing.

## One idea that was built, checked and thrown away

A depth-dependent attitude envelope for the guarded advance. The reasoning is
sound — `2c/l` shrinks as the module goes in, so a fixed bound keeps pushing a
module that stopped fitting — and it was written, linted and tested before a
check against data already in hand refuted it.

`c` is not one number along the stroke. It is 8.00 mm at the nominal lead-in
surfaces and 12.61 mm in the relieved channel behind them, and the bound built
from the lead-in gap gives 35.6 mrad at full engagement against modules that
measurably seat at **46.7**. It would have held exactly the runs that succeed.

It is recorded here, and in the guarded advance's own report under
`why_not_depth_dependent`, because the reasoning is attractive enough to be
reinvented.

## What was claimed and was not true

**`evidence/robot_carried_full_chain_seated.json` never left the transit phase.**
Its own `reached_phase` is `"transit"`. The "six of seven seating conditions
pass" it was quoted for is the last transit leg overshooting its waypoint by
98 mm and happening to stop at the right depth. The insertion phase contributed
nothing to it. Logged in `evidence/RETRACTED.md`.

**The satellite base compliance is authored and carries nothing.** A D6 joint is
declared at 12 kN/m and 600 N·m/rad between the mount anchor and the robot's base
link, and `robot_mount_unstable` is written to end an episode when they disagree
by 16.5 mm — but the robot spawns with `fix_root_link`, so PhysX welds the root
to the world and the spring has nothing to deflect. `mount_rotation_rad` is
0.000000 on every step of every run. The report now says so in
`base_mount_compliance`, because a zero deflection is also what a *working*
compliant mount looks like and the two must not be confusable. It is not simply
deleted because its spawner also relocates the UR10e articulation root, so
removing it changes the robot as well as the claim.

**The live service preset ran a configuration that cannot work.** It omitted the
rail and set an episode shorter than one workflow, so the one run a viewer
actually sees could only ever end by running out of clock. Both fixed, and
`tests/test_service_core.py` pins them.

## What moved it, and by how much

Measured on one seed with one thing changed at a time, so the credit is
attributable rather than shared:

| Configuration | Seed 4070 |
| --- | ---: |
| The chain as it stood at the start of the session | 37.50% |
| + the six fixes above | **78.12%** |
| + grasp and extract retrained on the module that exists | **90.62%** |

Roughly 41 points from the fixes and 12 from the retraining. Pooled over three
seeds the finished chain is 90.62%, 100%, 100%.

### The skill certifications described a module that no longer exists

`BLADE_SIZE` went from 450 × 160 × 35 mm to 450 × 130 × 20 mm — a third of the
mass and a different inertia tensor, in tasks whose entire difficulty is a
contact transient on a free-floating payload. The published figures for grasp and
extract, 94.46% and 94.89%, were taken on the old one. Evaluated on the current
module, at curriculum stage 0 which is where the chain runs:

| | Checkpoint the chain used | Retrained | Pooled over all three stages |
| --- | ---: | ---: | --- |
| Grasp | 89.06% | **99.00%** | 63.43% → **86.64%** |
| Extract | 78.65% | **85.37%** | 72.92% → **74.27%** |

Grasp gained 23 points pooled and 37 on its worst stage. **Extract gained
nothing** — 900 epochs moved it 1.4 points pooled and cost it 2.7 on its worst
stage — and it is now the binding skill in the chain. The obvious next move is
more epochs for extract specifically; nothing about this session tested whether
that helps.

The controls that make those comparisons legitimate are
`evidence/grapple_grasp_v6w65_on_current_module_control.json` and
`evidence/grapple_extract_v16w65_on_current_module_control.json`. Without them
"the retrained grasp scores 86.64%" reads against a 94.46% taken on different
hardware, and looks like a regression instead of a 23-point gain.

### Why the chain beats both of its skills

96.88% against 86.64% and 74.27% is not an inconsistency. The chain is not the
product of the skill certifications: those pool three curriculum stages and the
chain runs stage 0; each phase hands over on the **next** phase's precondition
rather than on its own success criterion; and the guarded seating recovers
deliveries a skill certification would have scored as failures. The two numbers
answer different questions and both are reported.

### The insert policy

Fine-tuned 900 epochs from the promoted two-bay checkpoint onto the current
geometry it certifies at **0.00%** pooled over 3,002 episodes, against 10.50%
before. Its reward was still climbing steeply when the clock stopped it
(−81.7 → −42.0 and rising), so that number describes an unconverged policy rather
than a refutation.

A longer fine-tune was started and then **abandoned deliberately**. It had run
600 more epochs, with the reward slightly worse (−44.9), when the decision was
taken to retrain the insert policy from scratch next session rather than keep
fine-tuning a checkpoint shaped for the previous module. Its partly trained
weights sit in
`logs/rl_games/zero_g_blade_insertion_contact/grapple_insert_l0_seed70_v14m130long/`.
**It was never certified and no number may be quoted from it.**

The chain does not use it. The seating is the scripted guarded advance, the
report labels it so, and `--insert_controller policy` exists so the comparison
can be made on this chain rather than inferred.

## What is still simplified, and named

- The form lock's load path is a break-rated PhysX fixed joint while rigid and a
  bounded spring-damper while compliant. Its jaws carry no collider, so contact
  between them and the pin is not simulated; the clearances are derived
  analytically by `scripts/check_service_latch_clearance.py`.
- The lateral rail indexes a robot base already fixed to the world. Motor, screw,
  bearings and brake are unmodelled. What is *not* simplified is the claim it
  exists to support: the module is never written, never constrained to the world,
  and never held by anything but the robot, and the transit trace records the
  base's own position so a carriage that was commanded and did not move cannot be
  reported as one that crossed.
- The base mount compliance is not in the load path, as above.
- Simulation throughout. No result in this repository was produced on hardware.

## What was deliberately not done

**The world-mounted payload shuttle behind `--base_rail_on_relocation` is still
in the driver** — about 200 references, plus `configure_base_rail` and the
`payload_stage` scene entry. It is forbidden by this project's own rules, it is
why the driver carries two transit controllers, and it is why the honesty label
was once keyed on the wrong flag. It is kept out of the live preset and out of
the certified path by `tests/test_robot_carried_contract.py`, and its *results*
are preserved in `evidence/`.

Deleting it is the right thing to do and it was not done in the last session,
because it is a 200-reference change to the file that produces every number
above and there was no measurement budget left to re-validate the chain
afterwards. Do it first, next time, with the pooled certification re-run
immediately after.

## How to reproduce anything here

```bash
scripts/run_robot_carried.sh rail
```

One environment, one seed, the whole job. About eight minutes.

```bash
scripts/run_robot_carried.sh certify
```

32 environments, three held-out seeds, pooled with a Wilson interval.

```bash
python scripts/check_workcell_geometry.py
```

Where the arm can stand, how much authority it has there, what the channel and
the lead-ins admit, and the hand-off attitude requirement — in about a second,
with no simulator, validated against the simulator's own recorded configurations
before it reports anything.
