# Next session: make the whole chain work by fixing where the robot stands

## The goal

Get the six-axis robot to do the complete job, end to end, for real:

**grasp the module → pull it out of slot 1 → carry it to slot 2 → push it all the
way into slot 2 and let go.**

Right now the first three work and the fourth does not. This session is about
fixing that by **changing where the robot is**, not by tuning the controller.

## What already works, so don't rebuild it

- The robot grasps the module and pulls it out of slot 1. Trained policies,
  already certified, unchanged.
- The robot **carries** the module across to slot 2 while holding it steady to
  about 3 mm. This uses a visible robot-side latch that clamps the module's
  shaft. It works and it should stay.
- Camera-based perception (RGB-D) drives the whole chain. It works.
- One full run with cameras on: `evidence/robot_carried_rgbd_seed6070.json`.

## What does not work, and exactly why

The robot pushes the module into slot 2 and stops **16 cm short** of fully home.

The reason is geometry, not software. At the spot where slot 2 is, the arm
cannot hold the module straight — it arrives about **3.6 degrees crooked**. The
module is 45 cm long. A long part held crooked wedges in a slot, the same way a
long plank jams in a doorway if you carry it at an angle.

Widening the slot was swept properly and **does not fix it**. Every extra
millimetre of slot clearance lets the module go about 1.2 mm further, but also
lets it end up about 0.2 degrees more crooked, because the slot walls were what
was straightening it. The two effects cross before the module ever seats. The
numbers are in `evidence/robot_carried_seating_sweep.json`.

Pushing harder does nothing either: four times the force moved the module
0.1 mm.

**So the fix is upstream. The arm has to be somewhere it can hold the module
straight.** Section 4 of `docs/robot_carried_handoff.md` has the full
measurement trail if you want it.

## What to do, in order

### 1. Do the geometry first, before any training

Work out where the robot has to stand so that **every** step of the job is
comfortable — not just the insertion. Check all of these before spending any
GPU time:

- Can it reach the grasp pose in slot 1 and hold the module square there?
- Can it pull the module straight out without the arm folding up?
- Can it reach slot 2 and hold the module square there too?
- Does the arm or the module hit anything along the way?

The existing measurement of where the arm goes bad is section 6a of
`docs/service_interface_spec.md` — there is a region around the arm's own base
axis where it cannot hold both position and angle at once, and slot 2 currently
sits in it. Use that. `scripts/check_service_latch_clearance.py` is the pattern
for a geometry check that runs with no simulator.

Only move on once the geometry says every pose is achievable with margin.

### 2. If no single fixed position works, try the rail

Fallback, and possibly the better answer anyway: **put the robot on a rail** so
it can drive along like a train and park in front of whichever slot it is
working on.

This is worth taking seriously on its own merits, not just as a backup. In space
a servicing robot faces a rack with many slots spread over a long structure —
far more than one arm can reach from one spot. A rail turns "can the arm reach
slot 2" into "can the arm reach *any* slot", which is the question that actually
matters for a real constellation.

If you go this way, the rail carries the **robot**, not the module. The module
must stay in the robot's grip the whole time. Do not move the module with
anything but the arm.

### 3. Then train properly

Once the geometry is settled, run a proper training pass to confirm every skill
still works from the new position. The grasp and extract policies were trained
at the old position, so moving the robot means they need re-certifying.

### 4. Produce three videos

1. Grasping the module and pulling it out of slot 1.
2. Pushing the module into slot 2 and letting go.
3. **The most important one:** the whole job in one continuous run —
   grasp, pull out, carry across, push in, release.

`scripts/run_robot_carried.sh rgbd` records video. Note that video recording and
lighting randomisation are separate switches; you need one run of each if you
want both a clean video and evidence with randomisation on.

## Rules that do not change

- The **robot** must carry the module. No invisible carrier, no teleporting the
  module, no writing the module's position directly, no attaching it to the
  world. `tests/test_robot_carried_contract.py` enforces this — keep it passing.
- The old world-mounted "payload shuttle" behind `--base_rail_on_relocation` is
  kept only as a labelled historical record. Do not use it and do not relabel
  its old results as robot-carried.
- Keep failed results, labelled honestly. If something cannot be done, report
  the measurement that says so rather than working around it.

## How to work

- **Plain English.** Short sentences, no jargon, no buzzwords. If a number
  matters, give the number.
- **Work quietly.** Do the work and report once at the end, not step by step.
- **Do not leave background tasks running.** This is a real problem from the
  last session: every wait-for-a-run timer that gets started must be stopped
  when the run finishes. Check with `tasklist` for `kit.exe` *and* check for
  leftover shell processes before saying you are done.

## Where things are

| What | Where |
| --- | --- |
| Read this first | `docs/robot_carried_handoff.md` |
| Why the insertion fails, with numbers | `evidence/robot_carried_seating_sweep.json` |
| The one full camera-driven run | `evidence/robot_carried_rgbd_seed6070.json` |
| Where the arm goes bad | Section 6a of `docs/service_interface_spec.md` |
| The workflow driver | `scripts/run_workflow_demo.py` |
| Running each stage | `scripts/run_robot_carried.sh` |
| Branch | `industrial-relocation` |
