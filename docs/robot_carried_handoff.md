# Robot-carried relocation: what is proven, what is scripted, what is open

`docs/claude_opus_5_handoff.md` is the task this branch was given. This is the
answer to it, and it is the file to read first.

## The question, and the answer

**Can the six-axis robot itself carry the compute module from bay 0 to bay 1?**

Not on the finger pads. Measured on 16 environments with nothing changed but the
interface, the passive parallel-jaw grip loses the module entirely: **0 of 16**
retained the tool-to-module transform the flight was planned from, at a median
drift of **808 mm** and **3.14 rad** — π, a module that has turned end-for-end.
The tool travels 168 mm while the module travels 913 mm. Retention is lost at
the median at control step 303, about ten seconds in, and every one of the 16
workflows times out inside the transit.

Yes on a **robot-side form lock**: the same flight holds the transform to
**2.3 mm** and **6.2 mrad**, and the module travels with the tool rather than
away from it.

The seating in the destination bay is **not closed**. Section 4 below is the
measured reason, and it is a conflict between two requirements rather than a
tuning gap.

## What is learned, scripted, perceived, and constrained

| Phase | What drives it | State |
| --- | --- | --- |
| capture | trained policy, certified checkpoint, unchanged | works |
| seat | scripted pause | works |
| extract | trained policy, certified checkpoint, unchanged | works |
| transit | scripted: five module-space waypoints on the form lock | works, 2.3 mm / 6.2 mrad |
| insert | scripted guarded advance on the deployed estimator | **not closed** |
| done | settled re-check, then the hand opens | not reached |

Perception is the calibrated RGB-D fiducial estimator and the occupancy planner,
both unchanged and both fail-closed. Simulator truth is used for scoring and for
one geometric interlock that protects the rack from the mechanism; it is never a
policy observation.

## 1. The lock is on the robot, not the module

Section 8.4 of `docs/service_interface_spec.md` already contained the rule, from
a sweep of the gripped section against the measured hand: a serviceable module
cannot carry a positive axial stop forward of the pads, so the axial lock has to
come from the end-effector. Two module-side attempts had been built and refuted
before it was written. This is the first design on the correct side of it, and
it costs the module nothing — the pin is unchanged, so every certification taken
against it still describes the part that is built.

Section 9 of the specification is the design.
`evidence/service_latch_clearance.json` derives every clearance from the measured
gripper envelope with no simulator; the tightest are +3.0 mm and +5.9 mm. The
derivation rejected two of its own dimensions before it passed.

The carriage **seeks** the collar rather than assuming it, because a tapered
wedge does not seat a module at one depth. Measured travel used: 12.0 mm, which
is the self-seating distance section 4 of the specification predicts, arrived at
from the other direction.

## 2. The transit is planned in module space

A pad-held module moves in the grip, so the old follower servos the tool and
corrects the module afterwards. A form-locked module is a rigid extension of the
wrist, and that structure does not merely waste effort on it — it diverges,
because the alignment sub-phase computes a tool target from the module's
position and rotating the tool moves the module. Measured: 0.66 rad walked to
1.61 rad while the module was dragged 380 mm backwards.

The replacement is five module poses and one servo, with the position command
inverted through the attitude the tool *has* rather than the one it is being
asked for. Without that inversion a standing attitude error becomes a standing
position error of the same size times the 340 mm offset — measured twice, as a
lateral leg parking 61 mm and then 64 mm short.

Five legs rather than three because of the workcell, not the payload: section 6a
measures a region around the arm's own axis where position and the head-on
attitude cannot both be held, and bay 0 sits on that axis. A leg that asks the
arm to cross *and* square there gets a compromise — 0.164 rad, which on a rigid
payload is also 56 mm of position error it cannot correct. Squaring is therefore
a leg of its own, done where the arm can afford it.

## 3. The rack needed a lead-in on an axis nobody had asked about

Section 6 of the specification measures the lateral flares as load-bearing:
removed, two fully trained policies insert nothing. That result is about the
lateral axis. Nothing had ever asked about the vertical one, because **both
insertion skills reset with the module already inside the channel** and so never
entered the mouth from outside.

A relocation enters from outside. Section 6.1 is the requirement that follows,
and section 6.2 is the one after it: a module delivered *rigidly* cannot be
straightened by the channel it is entering, so the channel has to admit the
attitude the manipulator delivers — clearance ≥ L·θ/2.

## 4. What is open, and why it is hard

Carrying and mating want opposite things from the same mechanism.

| Lock state during mating | What the seating did |
| --- | --- |
| rigid throughout | module advanced **0.3 mm** in a 30-second budget: an arm pushing a rigid link against a channel |
| released at the phase boundary | advanced **15.6 mm**, then wedged — already inside the channel, already crooked |
| released before the mouth | slid **laterally out of the bay**: the pads do not resist lateral load, which section 8 measures |
| compliant at 2.5 kN/m | the lead-in pushed it 19 mm off centre, outside its own 16.6 mm catch |
| compliant at 10 kN/m | 7.7 mm off centre, entered 46 mm, jammed |
| compliant at 40 kN/m | reached 0.5804 m, 0.07 mm outside the hand-off gate |
| compliant, 1 kN of push | wedged at a third of its travel, with and without channel relief |
| compliant, soft in rotation | module rotated 0.309 rad against the spring and jammed crooked |
| compliant, stiff in rotation, as a wrench | unstable at 30 Hz: module left the cell at 1.5 m |
| compliant, stiff in rotation, as a joint | stable, and jams at the mouth: 14 mrad will not enter a 1 mm channel |

The current design gives the lock three states — rigid to carry, a bounded
compliance with a finite stroke to mate, released once seated — which is what
assembly compliance devices have done since the 1970s and what the SSRMS
latching end effector does when it berths. Two things about that middle state
are requirements rather than tuning. It is **a joint, not an applied wrench**,
because an explicit spring at the 30 Hz command rate is stable only while it is
soft. And it is **soft in translation and stiff in rotation**, because the rack
aligns a module by pushing it and cannot straighten it.

**The underlying number is the one to argue with.** A 450 mm module, a channel
with 0.5 mm per side, and an arm that delivers 14 mrad of attitude — which is
3 mm of tip swing. That is a jam-prone assembly by construction, and it is the
finding this project exists to produce. The ways out, in order of preference:
stand the arm outside the reach boundary of section 6a so it delivers better
than 14 mrad; chamfer the module's leading edges; widen the channel to L·θ/2.

## Where the work is

| Purpose | Path |
| --- | --- |
| The latch's dimensional contract | `src/zero_g_blade_swap/service_latch.py` |
| Its clearance derivation | `scripts/check_service_latch_clearance.py`, `evidence/service_latch_clearance.json` |
| The workflow driver | `scripts/run_workflow_demo.py` |
| Reproducing every stage | `scripts/run_robot_carried.sh` |
| The live compute-service preset | `src/zero_g_blade_swap/service/presets.py` |
| Contracts that keep the shuttle out | `tests/test_robot_carried_contract.py` |
| The latch geometry's contract | `tests/test_service_latch_geometry.py` |

## What was retained, not superseded

`evidence/full_chain_state_16_report.json` and
`evidence/full_chain_rgbd_service_seed4070.json` describe the payload-stage
baseline: a chain in which a hidden world-mounted D6 stage took the module off
the arm after extraction and moved it independently. They are good measurements
of that mechanism and they are **not** robot-carried results. The mechanism is
still reachable behind `--base_rail_on_relocation` so those numbers can be
reproduced, and it is kept out of the live preset by a test rather than by
anyone's memory.
