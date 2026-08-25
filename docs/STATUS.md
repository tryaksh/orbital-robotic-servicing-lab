# Status

The verified state of this repository, and everything an agent needs to resume
from it. Read this before quoting a number or changing a constant.

Last verified: 2026-08-24, branch `industrial-relocation`.

---

## 1. What runs

The whole servicing job runs end to end in one continuous episode. A UR10e
locates a compute module in a rack bay, grips it, pulls it out, carries it to the
neighbouring bay, drives it home, and releases only after every seating condition
has been re-checked over a 0.70 s settle. The robot holds the module throughout,
with 0.9 mm and 2.5 mrad of tool-to-module drift at the median across the
certification and 2.3 mm and 6.3 mrad at worst.

RGB-D perception, occupancy planning, the local compute service, telemetry,
provenance and artifact export all work.

| Phase | Controller | Certified separately |
| --- | --- | --- |
| Grasp | RL policy | yes — `grapple_grasp_v7m130_on_derived_rack_certification.json` |
| Seat (0.03 s dwell) | scripted | — |
| Extract | RL policy | yes — `grapple_extract_v18pin_certification.json` |
| Transit, 5 legs | solved inverse kinematics, actuator targets | — |
| Insert | guarded advance on the deployed RGB-D estimate | the learned alternative is measured against it, §5 |
| Back off | scripted, after the settled re-check | — |

`docs/service_interface_spec.md` §10 states that split as a requirement and lists
the gates it is held to.

## 2. The numbers

### Chain

**97.92% pooled** — 94 of 96 episodes, 32 environments on each of three held-out
seeds, Wilson 95% interval **[92.7%, 99.4%]**. Per seed: 93.75%, 100%, 100%. The
project's gate is 95% pooled and 95% worst-case; both pass.

`evidence/workflow_robot_carried_m130pin_guarded_certification.json`

History, all preserved because a before is what makes an after mean anything:

| Session | Pooled | Report |
| --- | ---: | --- |
| Two sessions ago | 31.25% | `workflow_robot_carried_relocate_certification.json` |
| One session ago | 96.88% | `workflow_robot_carried_m130_guarded_certification.json` |
| Now | **97.92%** | `workflow_robot_carried_m130pin_guarded_certification.json` |

### Skills

Three curriculum stages on each of three held-out seeds, 500 episodes a point.

| | Pooled | Stage 0 (what the chain runs) | Worst stage |
| --- | ---: | ---: | ---: |
| Grasp v7m130, unchanged, on the derived rack | 85.69% | 99.14% | 78.68% |
| Extract v17m130, the checkpoint replaced | 87.78% | 91.53% | 82.80% |
| Extract v18pin, 2,000 further epochs | 87.75% | 91.08% | 84.08% |

Neither passes the 95% gate. **The chain exceeds both, and that is not a
contradiction:** the chain is not the product of the skill certifications. Those
pool three curriculum stages while the chain runs stage 0, each phase hands over
on the *next* phase's precondition rather than on its own success criterion, and
the guarded seating recovers deliveries a skill certification would score as
failures. Report both; quote neither alone.

### Checkpoints the chain uses

```
logs/rl_games/zero_g_blade_insertion_contact/
  grapple_grasp_l0_seed70_v7m130/nn/last_..._ep_3100_rew_30.262873.pth
  grapple_extract_l0_seed70_v18pin/nn/last_..._ep_12600_rew_172.70488.pth
```

Two, not three. `--insert_checkpoint` is optional: the seating is the scripted
guarded advance, so a chain that does not use a policy does not load one, and no
report can list a policy that never ran.

## 3. What was found this session, and why it matters

### Extract's ceiling is the task, not the training budget

900 epochs moved extract 1.4 points in the previous session; 2,000 more moved it
0.0 pooled in this one. What moved it was fixing three things about the task,
each measured on an **unchanged** checkpoint so the policy cannot be credited
with them (`evidence/extract_attribution.json`):

| Step | Stage 0 | Stage 2 |
| --- | ---: | ---: |
| As certified: 30 mm grip ball, 15.750 mm rack, unbounded reset | 84.57% | 60.55% |
| + grip judged on the pin's own axes | 82.03% | 39.30% |
| + rack lateral clearance derived (15.750 → 12.689 mm) | 83.82% | 44.83% |
| + reset bounded by the hand-over gate | **91.21%** | **84.80%** |
| + 2,000 further epochs of PPO | 90.80% | 85.23% |

A stricter criterion costs points, and that is reported rather than netted off.

### The module's cross-section is what made extraction hard

`BLADE_SIZE` went to 450 × 130 × 20 mm to buy clearance at the *destination*. In
the **source** bay it took the rails out of the load path:

| Module inside its own channel | 160 × 35 mm | 130 × 20 mm |
| --- | ---: | ---: |
| Lateral half-gap | 0.750 mm | 15.750 mm |
| Vertical half-gap | 0.500 mm | 8.000 mm |
| Pitch at full engagement | 2.22 mrad | 35.56 mrad |
| Roll | 6.25 mrad | 124.59 mrad |

Roll is also the axis a pair of flat pad normals cannot resist. The same
unchanged policy scores 99.02% on the section it was built for and 76.95% on the
current one, on one seed with nothing else different.

The task's own curriculum says the same thing a second way: its three stages
differ in how much of the module the rails still hold, and the rate falls
monotonically with the freedom left — 450 mm engaged → 91.08%, 435 mm → 88.09%,
358 mm → 84.08%, on one policy over a *shorter* pull.

### A tapered pin holds by feeding, and both grip criteria charged that as a drop

The pads come to rest **12.0 mm** along the pin from its drawing pose every time
a pull takes load — measured over 433 successful extractions, in a band 0.8 mm
wide. Both criteria were isotropic balls about that pose, 20 mm to count as
captured and 30 mm to count as dropped, so 12 mm of each was spent before the
policy acted. Of extract's stage-0 failures, 50 of 79 ended on that ball with 79%
of the error along the pin and the module 14.7 mm into a 525 mm pull.

The criterion is now three bounds on the pin's own axes — feed −42.0 mm, back-out
+5.0 mm, across 15.0 mm — two of which are *tighter* than the ball by a factor of
two. `docs/service_interface_spec.md` §3.2.

### The reset was generating episodes no policy could win

A joint-space noise box does not bound a grip error. At extract's widest stage,
202 of 513 episodes were dead inside three control steps with the pads closed on
nothing; conditional on surviving the reset the same policy scored 83.64%. The
reset now scales each draw's noise vector so the tool displacement stays inside
`WORKFLOW_HANDOVER_GRIP_M`, the gate the chain itself enforces before handing a
module to this skill. The noise *direction* is untouched.

### The rack could move the module further than the gripper could follow

A pair of flat pads 27 mm wide on a 30 mm pin keeps half its face on the pin only
while the offset stays inside the pin's own half-width, 15.0 mm. A module in the
*corner* of a rectangular channel is offset by `hypot(lateral, vertical)`, and
the vertical gap is spent on the hand-off attitude requirement, so lateral is
bounded at `sqrt(15.0² − 8.0²) = 12.689 mm`. The rack was at 15.750 mm.
`GUIDE_CENTER_OFFSET_Y` is derived from that inequality now.
`docs/service_interface_spec.md` §6.3.

### Lead-ins have to move with the rails they continue

Deriving the guide offset moved the rails inboard 3.061 mm. `_FLARE_CENTER_Y` was
an authored literal and stayed; `_RAMP_SURFACE_OFFSET` is the *difference*
between the two, so the vertical ramp moved 3.061 mm the other way. **The chain
scored 0.00% over 32 episodes on that rack**, with the module still arriving
1.0 mm from the seated plane — the failing condition was 4.04 mm of lateral
against a 2.5 mm tolerance. Both lead-ins are derived from the rail face now, and
`tests/test_workcell_geometry.py` holds the relationship. See
`evidence/RETRACTED.md`.

### A skill has to be trained on the problem the chain hands it

Three policies are trained in isolation and then run inside one continuous
episode. That only means anything if each skill's task *is* the chain's phase,
and nothing checked it. Audited on 2026-08-24, capture and extraction agreed with
the chain on every dimension and the insert skill agreed on **none**:

| | Insert skill task | Chain's insert phase |
| --- | --- | --- |
| Bay | drew both, one a 505 mrad stretch | the second, robot parked opposite it |
| Vertical entry lead-in | absent | fitted |
| Channel relief | none | 4.6125 mm per side |
| Destination surfaces | production friction 0.8 / 0.65 | low-friction pairing |
| Load path | pads on the pin, lock off | bounded spring-damper on the lock |
| Module at reset | one pose, 362 mm from the hand-off | at the mouth |
| Seated goal | 0.75 m in bay 1 | 0.676 m, the depth release permits |
| Axial action scale | 45 mm/s, sized for a 167 mm stroke | the same field, sized for 529 mm |

The first four came from one function. `configure_service_destination()`
installs the vertical lead-in, applies the relief and puts the bay's running
surfaces on the low-friction pairing, and only `run_workflow_demo.py` ever called
it. Its own docstring records what a bay without it does to a module delivered
from outside: it "cocked to 36 mrad, exactly the `2c/L` the channel admits, and
then did not move for six thousand control steps of pushing, under every mating
variant tried."

So the insert skill was being asked for something the geometry forbids, which is
why it certified at 0.00% while holding the grip perfectly. Extraction agrees
with the chain — its lock is off in the task and off in the chain too, because
the chain arms the lock only once the module is clear of the rails, which is the
moment extraction ends — and that agreement is why extraction transfers.

Seven of the eight are now equal on both sides and
`tests/test_skill_chain_agreement.py` holds each one. It is a source-level test,
so it runs in CI without a GPU — this project's rule for anything that has to
keep being checked.

**The load path is the one still open, and it is measured rather than
overlooked.** Switching the lock on in the skill does not work: its joint is
authored between the wrist and the module at their *spawn* poses, and this task's
reset writes the module anywhere along 436 mm of stroke, so PhysX resolves the
disagreement by snapping them together. With the lock as the only change, same
checkpoint and seed:

| | Dead inside ten control steps | Roll about the pin |
| --- | ---: | ---: |
| Lock on | 125 of 128 | 247.6 mrad |
| Lock off | 0 of 128 | 9.4 mrad |

Closing it needs `mdp.GrappleLatch` to re-anchor after a reset, which is code
rather than a configuration value. The chain's mating numbers are declared on
the task next to the measurement, so it is a one-line change when someone takes
it on.

**This is the finding of the session**: the certifications were honest and the
chain was honest, and the two described different problems with nothing standing
between them.

### What breaks the chain first

One variable at a time around the certified configuration, 16 environments and 16
episodes a point, one seed. Coarse on purpose: the Wilson interval on each point
is about twenty points wide, so this ranks variables rather than measuring them.
`evidence/chain_robustness_sweep.json`.

| Point | Success | Below nominal |
| --- | ---: | ---: |
| nominal | 93.75% | — |
| module 120 × 16 mm | **0.00%** | 93.75 |
| robot base +10 mm across the bay | **6.25%** | 87.50 |
| rack lateral clearance 16 mm | 75.00% | 18.75 |
| module mass 40 kg | 81.25% | 12.50 |
| module 140 × 26 mm | 87.50% | 6.25 |
| module mass 20 kg | 93.75% | 0.00 |
| rack lateral clearance 6 mm | 93.75% | 0.00 |
| destination relief 0 mm | 93.75% | 0.00 |
| robot base 50 mm further back | 100.00% | −6.25 |

Three things to take from it.

**Module cross-section dominates, and it was predicted.** The closed-form
envelope in `check_workcell_geometry.py` calls 120 × 16 mm rejected on grip and
140 × 26 mm and 130 × 20 mm accepted, and the simulator agrees on all three. The
rack-clearance points confirm the window from both ends: 6 mm is inside it and
costs nothing, 16 mm is outside it and costs 18.75 points.

**Where the robot parks across the bay is the second most brittle thing.** A
10 mm error takes the chain from 93.75% to 6.25%. That is the rail's indexing
accuracy, and nothing in this project had ever put a number on how good it has to
be. It has to be good.

**Mass is nearly free and depth is not the trade it looks like.** Doubling the
module to 20 kg costs nothing and quadrupling it to 40 kg costs 12.5 points,
which is a much wider payload band than the interface specification claims.
Moving the base 50 mm further from the rack scores 100%, consistent with the
authority argument in `service_interface_spec.md` §6a — but note that an earlier
session measured the *capture policy* failing to close at that base position with
a different checkpoint, so this is one seed and one grasp policy, not a
recommendation to move the base.

## 4. Settled, and not to be re-litigated

- **The mating stroke must be compliant.** Re-measured on the current module:
  rigid reaches 0.2275 m and 269 mrad against compliant's 0.6753 m and 7 of 7
  conditions. `evidence/robot_carried_rigid_mating_refuted.json`.
- **The rail carries the robot, never the module** (`--robot_rail_on_relocation`).
  Parked opposite a bay the arm's configuration is the one it has at bay 1, so no
  bay needs a skill another bay does not already have.
- **Do not widen the channel and do not shorten the module.** Both measured, both
  dead ends. Narrowing the channel is a different question and is now derived.
- **"The robot at the end of its own reach into the rack" was wrong.** The
  destination seated plane is set by the latch's release interlock, not by reach:
  the closed-form kinematics solve every station from 0.147 to 0.75 m to 0.0001 mm.
  The plane itself is unaffected.
- **The world-mounted payload shuttle** behind `--base_rail_on_relocation` is a
  labelled historical baseline only. It is unreachable from the live preset and
  `tests/test_robot_carried_contract.py` keeps it out.
- **A depth-dependent attitude envelope for the guarded advance** was built,
  checked against data already in hand, and refuted before it ran. The reasoning
  is attractive enough to be reinvented; it is recorded in the guarded advance's
  own report under `why_not_depth_dependent`.

## 5. Open

- **Extract and grasp both miss the 95% gate** (87.75% and 85.69% pooled). Extract
  is no longer the binding skill; grasp's worst stage is now the lower of the two.
  Neither responds to more epochs on the evidence available.
- **`GUIDE_CENTER_OFFSET_Y` sits exactly on its upper bound**, because it is
  derived as the largest clearance the pads can follow. The window runs down to
  5.738 mm and a value in the middle would leave margin on both sides. Not
  measured.
- **The insert policy does not seat, and the reason has moved.** Four things
  about the task were wrong and are fixed: the reset was one pose 362 mm from
  where the chain hands over and now spans the whole stroke from a bank solved in
  closed form; the axial action scale was sized for a 167 mm stroke and is now
  sized for 529 mm; bay 1's goal was 74 mm past the depth the release interlock
  permits and is now the deliverable plane; and the retention reward's position
  half was charging the pin's own load path, about 150 an episode against the ~71
  that finishing pays.

  That last one was the dominant term, and removing it is visible immediately:
  the best mean reward jumped from −80 to −18 on the step the corrected reward
  took effect. It also fixed what it was supposed to — on 128 held-out episodes
  the policy loses the grip in **0** of them, against a checkpoint history where
  grip loss was the failure mode.

  Then the audit in section 3 found seven more ways this task and the chain
  described different problems, and fixing them changed the behaviour
  completely. Trained from scratch on the corrected task, the mean reward goes
  **positive for the first time in this project** — −80 before, +13.7 after —
  and the policy aligns:

  | | Before the audit | After |
  | --- | ---: | ---: |
  | Lateral error at the end | 20.7 mm | **7.9 mm** |
  | Orientation error | 128 mrad | **86 mrad** |
  | Best mean reward | −74 | **+13.7** |
  | Success | 0.00% | 0.00% |

  It still does not seat: 0.00% over 1,536 episodes
  (`evidence/grapple_insert_v20chain_certification.json`), stopping a median of
  204 mm short with the whole clock spent, against tolerances of 2.5 mm and
  52.4 mrad. So the chain keeps the scripted advance and this is published as a
  negative result, as section 10.2 of the specification requires — but it is now
  a *training* result rather than a task that could not be solved.

  **The next thing to try is already in the tree and is untested.** The module is
  still moving at 3.65 mm/s when the clock stops, against the 120 mm/s its action
  scale allows and the 60 mm/s the scripted advance uses to cover the same stroke
  in nine seconds. It is not jamming, it is creeping — and creeping is what the
  objective paid for, because progress is potential-based and dawdling cost 3
  over a whole episode against a success worth 30. `elapsed_time_penalty` is now
  weighted so a full clock costs 12, chosen to sit *below* the 15 that failing
  costs so the policy cannot learn to give up early instead of hurrying. Trained
  300 epochs it had not yet moved the creep, which is too early to read: the run
  before it took 800 before its behaviour settled. Retraining that to
  convergence is the open task. **This is now a training problem rather than a
  task-specification problem**, and it is the first time in this project that has
  been true of the insert skill. The chain seats with the scripted guarded
  advance, `--insert_checkpoint` is optional so a chain that does not use a
  policy need not load one, and the head-to-head numbers are published beside
  each other.
- **The chain's number and the perception number are measured on different
  inputs and have never been combined at scale.** The pooled 97.92% runs on
  `Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0`, the *state* task: the module
  pose comes from the simulator and the guarded advance's "deployed estimate" is
  the deployed code path reading ground truth. Perception is certified separately
  on 1,024 rendered frames. The RGB-D chain has been run end to end, at one seed,
  and not since this session's changes. Putting the two numbers side by side
  without saying this is the easiest way to overstate what is built, so it is
  said here. Certifying the chain on the vision task is the highest-value
  measurement missing.
- **Every policy is one PPO training seed.** The evaluation seeds are held out,
  so the rates are honest, but training repeatability is untested and no number
  carries a spread. Three training seeds a skill would turn "this policy scores
  X" into "this method scores X plus or minus something".
- **Every certification is at robustness level 0.** Levels 1 to 4 exist -- pose
  noise, module mass, friction, mount wobble -- and none of them is exercised by
  a published number.
- **Training randomizes none of the variables the sweep says the chain is
  sensitive to.** A 10 mm error in where the robot parks across the bay takes it
  from 93.75% to 6.25%. Randomizing base position, module section and rack
  clearance *during training* is what turns a point certification into a
  tolerance band.
- **Delivered angle has about 10 mrad of margin** — modules seat at 46 mrad
  against a 56 mrad channel. It is the only quantity in the certification
  operating against a limit.

## 6. Reproducing

```bash
# The whole job, one environment, end to end. About eight minutes.
scripts/run_robot_carried.sh rail

# 32 environments on each of three held-out seeds, pooled with a Wilson interval.
scripts/run_robot_carried.sh certify

# One skill, three stages, three held-out seeds.
SKILL=Extract CKPT=<path> TAG=<name> scripts/certify_grapple_skills.sh

# One variable at a time around the certified point.
scripts/sweep_chain_robustness.sh && python scripts/report_chain_robustness.py

# Geometry, no simulator, about a second.
python scripts/check_workcell_geometry.py
```

Two flags exist only so an archived checkpoint can be re-run under the criterion
it was certified against, which is what keeps a criterion change and a policy
change from being quoted as one number:
`play.py --legacy_grip_ball_m 0.030` and `--legacy_unbounded_reset`.

## 7. Rules that produced this state

1. A phase may not be labelled learned unless a policy produced the actions that
   ran. Key the label on the controller that stepped, never on a flag.
2. Every geometric requirement is derived by a check that runs without a
   simulator, and that check validates itself against the simulator first.
3. Change one thing at a time, and keep the losing arm. A criterion change and a
   policy change are never quoted as one number.
4. Never widen a tolerance to pass a gate. If a criterion is wrong, replace it
   with one derived from the parts and re-run the old checkpoint under both.
5. Check the geometry before spending the GPU. A policy cannot make a 3 mm swing
   fit through a 0.5 mm gap.
6. Failed and superseded results stay in `evidence/`, labelled. Claims that turn
   out to be wrong go in `evidence/RETRACTED.md`.

## 8. Where the rest of the reasoning is

`docs/archive/` holds the session handoffs in the order they were written. They
are kept for the reasoning and the negative results in them, not for their status
claims — every one has been superseded by this file. `docs/archive/README.md`
says what each contains.
