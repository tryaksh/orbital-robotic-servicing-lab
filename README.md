# Orbital Robotic Servicing Lab

A simulated zero-gravity workcell for one question: **can a robot pull a failed
compute module out of a spacecraft rack and slide a fresh one in, and if it
cannot, which dimension is to blame?** A UR10e arm does the whole swap in one
continuous episode — find the module, grip it, pull it out, carry it to the next
bay, line it up, push it in, let go — and the useful output is not the controller
but the list of things about the rack and the module that decide whether any
robot could have managed it.

Everything here is simulation, built on NVIDIA Isaac Lab. Nothing has run on
hardware and nothing here is a claim that it would fly.

---

## Why anyone would want this

Hardware that fails in orbit is usually written off. Servicing it means either
sending an astronaut outside the spacecraft — rare, expensive and dangerous — or
sending a robot, and robots are not yet trusted to do work that involves pushing
parts into other parts. As satellites and orbital computing platforms get more
modular, being able to swap a dead module for a live one, by robot, with nobody
up there, is the difference between an asset you can repair and one you throw
away.

The part that is genuinely hard is not the arm. It is that nobody designs a rack
or a module with a robot in mind, and by the time anyone tries to automate the
swap the metal has been cut. **The point of this repository is to find those
decisions early and cheaply, in simulation, with the evidence kept — including
the evidence that went against the person who ran it.**

---

## The short version of what was found

**The mechanical interface decides the outcome, not the controller.** Three
separate measurements say so, and all three were surprises.

| What was measured | The number | What it means |
| --- | --- | --- |
| What a plain two-finger gripper can hold on a smooth post, against what pulling the module out actually demands | about **6 N** held against **66.4 N** demanded | A factor of eleven. Gripping *harder* made it worse. That is a structural gap, not a tuning problem, and it is the measured argument for building purpose-made grab fittings into serviceable hardware. |
| How square a rack can hold a part that is merely resting in it | a rigid part of length `L` in a channel with `c` of clearance per side wedges at **2c/L** and cannot be squarer than that | The rack, not the robot, sets the angle of a part sitting loose in it. This rack held a module at **56.40 mrad** while demanding **52.36 mrad** before it would call it seated. It was asking for something its own geometry forbids. |
| Which design dimensions actually move the success rate | a **120 × 16 mm** module cross-section takes the chain from 93.75% to **0.00%**; a **10 mm** error in where the robot parks takes it to **6.25%**; doubling the module's mass costs nothing | These are the dimensions worth arguing about in a design review. Mass is not one of them. A closed-form geometry check called every cross-section result before the simulator was started. |

`N` is newtons, a unit of force: 6 N is roughly what a full soft-drink can weighs
on Earth. `mrad` is milliradians, a small-angle measure — 52.36 mrad is about
3 degrees.

> If you take one thing away: **a rack has requirements too, and this one had a
> requirement nobody had written down.** A channel may not hold a module outside
> its own acceptance criterion.

---

## Where the work actually stands

Being blunt about this is the point of the repository, so it goes near the top
rather than in a footnote.

**The whole swap runs end to end and scores 91.67%** — 22 of 24 episodes across
three held-out random seeds, with a 95% confidence range of **[74.2%, 97.7%]**.
A held-out seed is one the policy never saw while training. A range that wide is
what 24 episodes buys you, and it is reported rather than hidden.

**It fails its own gate, and the gate was not moved.** The gate is 95% and has
been 95% since it was written. 91.67% is not 95%, so the honest statement is that
the chain does not pass.

The rule being scored against is strict: the robot lets go of *both* of its own
supports and the rack alone has to hold the module for 0.70 seconds afterwards.
An older **97.92%** figure (94 of 96) is still in the records and is still true,
but it was scored while the robot was still helping. That is a different and
easier question, so it is kept as a **legacy** supported-settle comparator and
must not be quoted as the current rate.

| Arm | Result | Read it as |
| --- | ---: | --- |
| Full chain, rack alone holds the module | **22/24** | the current, strict number |
| The same chain with the rack's retention removed | 17/24 | the control: rack-side retention is worth the difference |
| Episodes that reached measured seating and then passed the rack-only recheck | 22 of 22 | with no measured drift; the two failures happen earlier and never reach the rack |

**Finding the module by camera is where it collapses.** With the module's
position read straight out of the simulator, the chain scores 20/24. With the
same chain driven by a depth camera — **RGB-D**, a camera that returns colour
plus a distance for every pixel — it scores **4/24**. The best camera-driven
configuration found so far gets back to 17/24, and no single change accounts for
that; the improvement is entirely in the combination.

**The learned seating skill does not survive being handed real work.** Scored on
its own it seats the module 36.77% of the time. Started from the states the rest
of the chain actually delivers, it scores **0 out of 96**. The hand-written
guarded controller scores 94 of 96 on those same handoffs, so that is what the
chain runs.

An earlier seating policy scored **0.00% over 1,536 episodes**, and this is the
sharpest single result in the repository: its three reward variants ended at
**84.26, 84.61 and 84.58 mrad** against a **52.4 mrad** tolerance. Three quite
different objectives landed within half a milliradian of each other. The reward
could not move the angle, because the angle was never the reward's to give — it
belongs to the interface, through the `2c/L` wedge above.

**Two skills sit just under their gate.** Capture scores 86.90% and extraction
87.64%, both against 95%. Both overlap the earlier certificates they were meant
to improve on, which scored **85.69%** and **87.75%**, so neither retrain can be
called an improvement.

**The serviceability envelope is not qualified.** The check that compares the
closed-form geometry against what the simulator actually does returns **not
qualified**: the rack-clearance and module-section arms disagree with each other,
a +10 mm rail-stop error is geometrically feasible but fails in simulation, and
capture clearance has only ever been calculated, never run. That verdict is the
current state of the headline claim, and it is in the records rather than
smoothed over.

---

## How it is put together

The split between learned and hand-written control follows the physics rather
than fashion, and each phase is labelled by the controller that actually stepped
it, never by a configuration flag.

| Phase | What runs it | Why that one |
| --- | --- | --- |
| Find and grip the module | a learned policy (**PPO**) | contact decides the outcome, and that contact is hard to write down |
| Pull it out of the bay | a learned policy (PPO) | the same |
| Carry it to the next bay | solved kinematics, collision-checked | free space; geometry already answers it, so learning would only add noise |
| Push it in and seat it | a hand-written guarded advance | it keeps going while the estimate stays inside a derived envelope, and stops when it does not |
| Let go | scripted, then rechecked | the rack's catches engage, both robot supports release, and the rack alone must hold for 0.70 s |

**PPO** — Proximal Policy Optimisation — is a standard reinforcement-learning
method: the robot attempts the task millions of times in simulation and its
behaviour is nudged towards whatever earns reward. **Solved kinematics**, often
written **IK** for inverse kinematics, means calculating the joint angles that
put the hand where you want it instead of learning them.

---

## Trusting the numbers

- **Every report is classified, mechanically.**
  [`evidence/MANIFEST.json`](evidence/MANIFEST.json) is generated from the files
  themselves and currently holds **64 canonical, 12 retracted and 167 historical**
  reports. Quote canonical. Never quote retracted.
  [`evidence/RETRACTED.md`](evidence/RETRACTED.md) says why each retraction
  happened.
- **A retracted certificate is a real retraction.** The old RGB-D perception
  certificate was withdrawn because the marker it was detecting floated 90 mm
  above the module it was supposed to be stuck to. Moving and aiming the fixed
  camera — and changing nothing about the accuracy gates — took held-out
  detection of the critical rack from **43.27% to 99.85%**, and overall detection
  to 92.87% over 1,024 frames.
- **Ten older reports cannot be reproduced, and say so.** They were produced from
  **uncommitted** code: the runs happened, but the exact code behind them does
  not exist any more. Thirteen reports carry a binding to the source that
  produced them — two match the working tree, one was mechanically **recovered**,
  and **ten** are marked lost. Nothing resting on a lost report is offered as a
  final claim, and closing this is task **T0** in
  [`docs/NEXT_WORK.md`](docs/NEXT_WORK.md): any result a final claim needs has to
  be re-run from a clean commit.
- **Losing arms are kept.** The refuted rigid-mating result, the 0.00% insert
  baseline, the retracted perception certificate and the inert probes are all
  still here, each with its scope attached.
- **There is no cheating in the chain.** No world constraint, no teleporting, no
  writing the module's position directly, no invisible carrier holding it. The
  robot holds the module the whole way.
- **A clone does not carry the weights.** Trained policies live under `logs/` and
  `checkpoints/`, which are not in git. Every report records the hash of the
  checkpoint behind it, so a report can be read without them but not re-run.

---

## Running it

Install steps and the exact Isaac Lab version are in
[`docs/INSTALL.md`](docs/INSTALL.md).

The checks that need no simulator, no GPU and no graphics — **1,049 tests in
about twenty seconds**:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not isaac and not camera and not benchmark"
```

Then the two that check the records still describe the code:

```powershell
.\.venv\Scripts\python.exe scripts\build_evidence_manifest.py --check
.\.venv\Scripts\python.exe scripts\check_source_provenance.py --depth 200
```

One end-to-end run in the simulator, which does need Isaac Sim and a GPU:

```powershell
scripts\run_robot_carried.sh rail
```

[`scripts/README.md`](scripts/README.md) is a generated index of every script in
that directory and what it is for. It is written from each script's own first
documentation line, so it cannot describe a script differently from how the
script describes itself.

---

## What this is not

Simulation only. No hardware, no real camera, no real connector or cable, no
thermal path, no orbital dynamics, no flexible spacecraft base, no
flight-readiness claim, and no safety certification of any load.

Some of the mechanism is drawn rather than modelled. The robot-side latch is
visible geometry with an idealized load path — a rigid joint while locked and a
spring-damper while compliant — and the robot's own base is fixed to the world,
so a base spring that exists in the configuration never actually deflects. The
rack's catches are the same: a disclosed 600 N / 30 N·m joint carries their
simulated load, and its reaction is not exposed. The 6 N against 66.4 N
comparison is a diagnostic from a simulation probe, not a hardware load rating.

One robot, one rack, one module family, one gravity setting — zero.

---

## Where everything is

| Need | Read |
| --- | --- |
| What is finished, what is open, and what each open item would cost | [ROADMAP.md](ROADMAP.md) |
| Which branches exist and why, and where the deleted ones went | [docs/REPO_MAP.md](docs/REPO_MAP.md) |
| The rules an agent works under here | [AGENTS.md](AGENTS.md) |
| Verified current state, in detail | [docs/NOW.md](docs/NOW.md) |
| The open task list with costs, as the sessions wrote it | [docs/NEXT_WORK.md](docs/NEXT_WORK.md) |
| Which report answers which question | [evidence/MANIFEST.json](evidence/MANIFEST.json) |
| What was withdrawn, and why | [evidence/RETRACTED.md](evidence/RETRACTED.md) |
| Past session handovers, kept for the reasoning behind decisions | [docs/handover/](docs/handover/) |

Every report carries its own declared scope. Read it before quoting a number out
of it.

### One name that will confuse you

The Python package is called `zero_g_blade_swap`, and tasks are registered under
names like `Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0`. "Blade" means a server
blade: this repository began as an experiment in swapping one in zero gravity,
and the modules are still called blades throughout the code. The names were left
alone on purpose — 35 evidence reports record the hash of a file at
`src/zero_g_blade_swap/…`, and renaming the package would break every one of
those provenance links to make a directory listing read better.
