# Handover prompt

Paste the block below to a fresh agent picking this repository up.

---

You own `orbital-robotic-servicing-lab`, on branch `industrial-relocation`. Act as
the senior robotics simulation engineer who does. You have **full authority** to
change the design, the code, the geometry, the tasks, the rewards, the workcell,
or the approach — including reversing decisions made so far. Nothing below is a
constraint on *what* you do; it is a record of what has been measured, so you do
not pay for it twice.

## What the project is

An Isaac Sim 5.1 / Isaac Lab study of **design for serviceability**: what must a
modular compute unit present, physically, for a 6-axis arm to swap it in
microgravity, and what loads does that impose. The deliverable is a design
specification — `docs/service_interface_spec.md` — not a policy. Zero gravity,
30 Hz policy / 120 Hz physics, one RTX 5070 Ti Laptop GPU, RL-Games PPO.

## Where to read, in order

1. `CLAUDE.md` — the operating rules and the *Do not retry* list. Each was paid
   for with a session.
2. `docs/status.md` — every result including the negative ones. Long. Its last
   several sections are this branch.
3. `evidence/RETRACTED.md` — what has been withdrawn, plus the note that every
   number on `main` describes a **different workcell** than this branch builds.
4. `python scripts/compare_workcells.py` — the current before/after table,
   generated from the evidence files rather than typed.

## State at handover

`main` is certified and published on the old workcell and is untouched. This
branch moved the robot base from x = −0.45 to **−0.65** and re-earned everything
against it.

| | `main` | this branch | |
| --- | ---: | ---: | --- |
| Removal chain | 98.78% | **99.48%** | gate passed |
| Capture skill | 88.78% | **94.46%** | fails its own 95% gate, as `main`'s does |
| Extract skill | 99.02% | 94.89% | regressed; the gap is timeouts at the widest reset |
| Insert skill, two bays | 98.60% | **10.50%** | bay 1 = 0.00%, bay 2 = 21.45% |
| Installation chain | 96.35% | **0.17%** | follows insertion down |
| Relocation chain | never ran | **0/64** | the arm now flies the whole path; the grip loses the module |
| Vision arms | 88.72 / 84.90 / 34.03 | not run | would be three zeros while insertion is broken |

## What is still to do

- **Insertion.** The single blocker: the installation chain and the relocation
  both depend on it, the removal chain does not.
- **The relocation chain**, which needs insertion *and* a transit that keeps hold
  of the module.
- **Perception.** Untouched this session. The pose head needs re-collecting and
  retraining on the changed geometry (`scripts/rebuild_perception.sh`), and the
  three vision arms re-certifying once a manipulation chain completes again.
- **A latch-on-release A/B** was set up and never produced a result. The code is
  in place (`run_workflow_demo.py --latch_on_release`) and untested.

## What has been measured, so you need not re-measure it

**The workcell.** Driving position and the head-on capture attitude together, the
arm holds that attitude only *outside* a region around its own base axis: about
**0.4242 m deep**, moving one millimetre per millimetre with the base, and
**155–167 mm wide**, a cone widening slightly with depth. Eight base positions,
64 classified cells: `evidence/workcell_reach_solution.json`,
`evidence/attitude_wall_lateral_profile.json`. Moving the base back 200 mm put
every pose the task needs outside it — that is what took the removal chain up and
the relocation's transit from stalling to flying.

**The relocation now fails differently.** On `main` the transit died on the cross
leg, the arm unable to hold attitude. Now `legs_remaining=[59, 0, 0]` — nothing
remains on the retreat or the cross, the tool ends 1.2 mm from its final waypoint
— and the *module* is lost, tool-to-module 1.216 m at the median. The vise behind
that is recorded: **hold** through the final leg turns the wedge into a thruster
on an unconstrained module, **retain** leaves it undriven (95 mm of 436).

**Why insertion broke, precisely.** Its success predicate refuses above **0.20 rad**
of grip attitude. The certified policy on the old cell was passing that at
p95 = 0.19447 — 2.8% of margin. The workcell change moved that quantity about 6%
and it crossed. In bay 2 every geometric condition passes at the median (5.0 mm
axial of a 12 mm limit, 0.9 mm lateral of 2.5) and only grip attitude refuses; in
bay 1 the module is aligned to 1 mm and never pushed in at all.

**Two hypotheses were tested and closed.** Recovering on the single-bay task
(one arm configuration instead of two) did not work. Arm extension is not the
cause: extraction starts at the same seated pose, breaks the module free of the
rails there, and reaches 98.5%.

**One change was made and never evaluated.** Insertion's grip-attitude penalty
was raised to extraction's tuned values — it previously charged 0.08/step at the
attitude its own criterion refuses at, and now charges 4.06. No policy has been
trained under it. Treat it as an untested hypothesis, and revert it freely.

## Two traps this session paid for

- **Training termination counters lie about resumed skills.** Insertion read
  0.0000 for 1,500 epochs while the same checkpoint certified at 10.50%. The
  counter carries exploration noise; certification does not. Judge a resumed
  skill with two minutes of deterministic `play.py`.
- **Moving the robot is not one constant.** Base, mount anchor,
  `GRAPPLE_HEAD_ON_ARM_JOINT_POS`, `SECOND_SLOT_STAGING_ARM_JOINT_POS`. An anchor
  left behind fires `robot_mount_unstable` every step and the arm never acts —
  which a reach sweep reports as a clean boundary rather than an error. Two
  contract tests defend it now.

## House rules that are not negotiable

- Every number in any document names its file in `evidence/`.
  `scripts/check_evidence_links.py` enforces it and a test enforces that.
- Never weaken a success threshold to pass a gate. Correcting a reset that
  produces unwinnable states is different, allowed, and must be stated with its
  measurement.
- A skill certification is not evidence about a chain. Certify both.
- Keep negative and retracted results in the record.
- One Isaac process at a time; check `Get-Process kit` before every launch and
  kill the parent shell too, or a loop will start another.
- Commit directly to the branch, author `tryaksh`, no `Co-Authored-By` trailers.
