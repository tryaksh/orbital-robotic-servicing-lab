> **Superseded.** Read [`final_session_handoff.md`](final_session_handoff.md)
> first. Most of the work below was done and several of its premises turned out
> to be wrong — in particular the reproducibility problem in "Where the chain
> stands" was a two-sided depth window plus a first-leg timeout, and the seated
> evidence it cites never left the transit phase. This file is kept for the
> reasoning, not for its status claims.

# Next session: make the hybrid claim true, then make it robust

## What this project is claiming

An orbital servicing robot removes a module from one bay and installs it in
another, using a **hybrid** strategy: reinforcement learning where contact is
rich and the model is bad, deterministic control where the geometry is known and
the requirement is repeatability.

That is the right thesis. **The implementation does not currently support it**,
and the gap is not subtle. Read this section before touching anything.

| Phase | What runs | Certified |
| --- | --- | ---: |
| Capture | RL policy | 99.48% (576 eps, 3 seeds) |
| Seat | scripted | — |
| Extract | RL policy | 99.48% (pooled with capture) |
| Transit | scripted, 5 legs | — |
| **Insertion** | **scripted guarded advance** | — |
| *(insert policy)* | *loaded, hashed, never asked for an action* | **10.50% pooled, 0.00% near-stage** |

So the hybrid is really "RL for the two phases where it works, scripted for
everything else, plus a dead policy carried along for appearance." That is
defensible as an engineering outcome but only if it is *stated*, and until this
session the report said the opposite — `learned_phases` listed `insert` and
`loaded_but_not_executed_policies` was empty, because the label branched on the
old payload-shuttle flag instead of on which controller actually runs. Fixed and
pinned by `tests/test_robot_carried_contract.py`.

**The job this session is to make the split principled instead of accidental,
and to certify both halves of it.**

## Where the chain stands

Rack as built, no channel relief, seed 4070, one environment:

| Seating condition | |
| --- | --- |
| Axial depth | pass — 0.3 mm short |
| Orientation | pass — 13.6 mrad |
| Linear / angular velocity | pass |
| Grasp position / orientation | pass |
| **Lateral alignment** | **fail — 4.2 mm against 2.5 mm** |

```bash
scripts/run_robot_carried.sh rail
```

`evidence/robot_carried_full_chain_seated.json`. Every phase runs. It is one
seed and one environment and must not be quoted as a rate.

**And it is not reproducible run to run, which is the first thing to fix.**
Re-running the identical configuration to regenerate that evidence with the
corrected honesty label produced a *different* trajectory: the module reached
0.6763 — the seated pose, at 13.5 mrad, correctly — but the last transit leg
overshot its own staging waypoint by 91 mm of tool travel, so the
transit-to-insert handoff contract never fired and the run sat there until the
step budget ran out. The two runs that did hand off went to insert at step 2183
with the module at 0.5935.

The only changes between them were the report label and moving the fiducial
plate; neither touches control. So either the chain has a genuine run-to-run
sensitivity, or the staging and seated poses are now close enough
(0.578 against 0.676) that the last leg can pass through one on its way to the
other. **Find out which before trusting any pooled number.** The handoff
contract lives in `_step_rigid_transit`'s arrival test and keys on distance to
`relocation_staging_pos`; a leg that overshoots it has no way back.

## Order of work

Do these in order. Each one either removes a weakness that invalidates a claim
or produces a number the next step depends on. Do not start training before
step 2 — a policy trained against a limit-cycling controller learns the limit
cycle.

### 1. Decide and document where the RL/deterministic boundary goes

This is a design decision, not a discovery, and it should be made explicitly and
written into `docs/service_interface_spec.md` as a section.

The defensible split, given everything measured:

- **RL owns contact under uncertainty**: capture on the pin (wedge geometry,
  pad slip, a grip whose success depends on contact transients no model
  predicts), and the final millimetres of seating where the module is inside the
  channel and the lead-ins are doing work.
- **Deterministic owns known geometry**: the seat dwell, the retreat, the rail
  crossing, the squaring legs, and the long free-space approach — all of which
  are pose-to-pose moves through a workcell whose kinematics
  `scripts/check_workcell_geometry.py` solves in closed form to 0.006 mm.

Under that split the current code has it backwards in one place: the guarded
advance is deterministic through the whole contact-rich seating stroke, and the
insert *policy* — the thing that should own that stroke — is not used at all.

### 2. Replace the scripted legs' controller with solved IK

**Highest-value single change, and a precondition for training anything.**

The scripted legs use IsaacLab's differential IK in relative mode: it re-anchors
on the tool's current pose each control step and drives to current + delta across
the decimation, so while the joints lag the deltas accumulate ahead of the arm.
The squaring leg therefore does not converge, it limit-cycles at about one action
scale — 4.5, 11.4, 13.9, 15.1 mrad on successive samples. Both a smaller rotation
gain (0.25) and a smaller uniform six-channel gain (0.3) make it **diverge** to
1.4–2.3 rad, because the arm is holding the module against the pads' closure
rather than merely aiming it.

Command joint targets from a solved IK instead.
`GraspSettlingDifferentialInverseKinematicsAction.set_joint_target_override`
already exists and is used by the joint-path replay. `check_workcell_geometry.solve_ik`
is validated against the simulator's own recorded configurations to 0.006 mm and
0.000 mrad — a torch port of it, seeded from the current joint positions, is the
whole change. It commands actuator targets like every other leg; nothing about it
is a teleport.

This also closes the one failing seating condition. The arithmetic: entry needs
channel clearance ≥ 0.225·θ per side, the seated check needs ≲ 1.8 mm per axis,
and both hold only if **θ < 8 mrad**. Get the delivery there, then tighten the
module back toward 450 × 157 × 32 mm, which is what the 2.5 mm seated tolerance
wants.

### 3. Make the simulation honest before training on it

Each of these is a place where the simulator is easier than reality in a way
that would make a trained policy wrong.

**The compliant mount is inert.** `CompliantD6JointCfg` is authored between
`MountAnchor` and `Robot/base_link` at 12 kN/m and 600 N·m/rad, and
`robot_mount_unstable` terminates on it — but `make_grapple_pin_robot_cfg` uses
`floating=False`, so `fix_root_link` is `True` and the joint carries nothing.
Measured: `mount_rotation_rad` is **0.000000 on every step of every run**. Every
claim about satellite base compliance on this branch is about a joint that is
not in the load path. Make the root floating, re-certify, and expect the numbers
to get worse — that is the honest version. If you decide against it, delete the
joint from the grapple scene rather than leave it implying something.

**The latch jaws have no collider.** The form lock's load path is a break-rated
PhysX fixed joint; its jaws are visual only, so contact between the jaws and the
pin is never simulated. `scripts/check_service_latch_clearance.py` asserts the
clearances analytically. Give the jaws colliders and let PhysX disagree. It is a
cheap experiment that could invalidate a specification section, which is exactly
why it is worth running.

**The estimator has a silent oracle.** `_payload_feedback` falls back to
`attached_blade_pose_world` — exact simulator pose — whenever no estimator is
attached. Correct for the state task; a hidden oracle if a vision task ever
fails to attach one. Make it raise when the task name says vision.

**The rail carriage is kinematic.** It indexes a base already fixed to the
world; motor, screw, bearings and brake are unmodelled. `robot_base_y_m` in the
transit trace proves the base moved when commanded, which is a correctness check
and not dynamics. Disclosed in `README.md`; keep it disclosed.

### 4. Retrain, on the geometry that now exists

**Everything certified is certified on a module that no longer exists.**
`BLADE_SIZE` went 450 × 160 × 35 mm → 450 × 130 × 20 mm, the destination bay's
seated plane 0.75 m → a derived 0.676 m, and the fiducial plate moved from a
tilted stalk to flush on the top face. The 99.48% removal figure is a
measurement of the old blade's mass and inertia.

Train and certify, in this order:

1. **Capture + extract**, retrained on the new module. They transfer well enough
   to run — that is how the current chain works — but "runs" is not "certified".
   Gate: pooled ≥ 95%, worst stage ≥ 95%, zero instability terminations, three
   held-out seeds, which is what `workflow_remove_w65` already holds itself to.
2. **A seating policy that owns the contact-rich stroke**, per the step-1 split:
   reset with the module at the mouth, square to whatever step 2's controller
   delivers, and reward seated depth with the lateral and orientation conditions
   as terminations. This is the policy that replaces the guarded advance, and
   it is the one that makes "hybrid" mean something. If after a fair attempt it
   does not beat the deterministic advance, **say so and delete the insert
   checkpoint from the chain** — a measured negative result is worth more than a
   carried checkpoint.
3. **Re-certify the fiducial** on the new plate
   (`evidence/fiducial_rgbd_service_plate.json` is superseded, logged in
   `evidence/RETRACTED.md`), then run the chain on the vision task on three
   seeds and report the state-vs-vision delta rather than assuming it is zero.
   Seed 4070 has never got through capture on the vision task.

Budget, from previous sessions: 20–40 minutes for a few hundred PPO epochs at
512 environments; about 15 minutes for a 3-stage × 3-seed certification; 5–10
minutes for one end-to-end relocation at one environment.

### 5. Certify the chain the way every other claim here is certified

`scripts/run_robot_carried.sh certify` — 32 environments × 3 held-out seeds,
pooled, with a Wilson interval. Everything currently quoted about the chain is
n = 1. Expect the pooled number to be worse than the single run; that is the
point of running it.

### 6. Produce one video that does not need an apology

The `workcell` inspection view renders through the viewport at control rate with
showcase key/fill lights at 4,500 and 1,500 intensity, and the output is heavily
speckled — visible on the arm's links in every clip. It is a render setting, not
a simulation problem.

Raise the RTX sample count on the recording path or record through a dedicated
high-sample camera rather than the viewport, and give the scene a floor and a
horizon so the rack is not floating in black. Target one clean 1080p clip of the
whole job. Current clips are in `artifacts/robotcarried/video/`.

## Deletions

- The **world-mounted payload shuttle** behind `--base_rail_on_relocation`: 227
  references in `scripts/run_workflow_demo.py`, plus `configure_base_rail` and
  the `payload_stage` scene entry. It is forbidden by the project's own rules,
  kept only as a labelled historical record, it is why the driver has two
  transit controllers, and it is why the honesty label was keyed on the wrong
  flag. Its *results* are preserved in `evidence/`; the code is not.
- The dead `elif` branch beside `_step_rigid_transit`'s caller, which only runs
  for the passive control. Three changes were made there before anyone noticed
  it never executes.

Together, roughly a quarter of the driver.

## What "industrial" should mean here, as gates

Write these into the spec and hold the run to them, rather than describing the
system qualitatively:

- Every phase that claims to be learned is certified ≥ 95% pooled and ≥ 95%
  worst-stage on three held-out seeds, with zero instability terminations.
- Every phase that is scripted is labelled scripted in the report, and the label
  is keyed on the controller that runs, not on a flag.
- The chain has a pooled success rate with a Wilson interval, not an anecdote.
- Every geometric requirement in the spec is derived by a check that runs in CI
  without a simulator, as `check_workcell_geometry.py` and
  `check_service_latch_clearance.py` already do.
- Every simplification in the load path is named in `README.md` under Honest
  limits and in the report that depends on it.

## What not to repeat

- **Do not widen the channel.** 10 mm of relief against 4.61 gives a
  byte-identical result; relief widens the tilt the channel permits by the same
  ratio it widens the entry.
- **Do not shorten the module.** Length is derived into every axial target;
  250 mm broke extraction outright. Take clearance from the cross-section.
- **Do not read `final.orientation_error_rad` as the delivered attitude** once
  the module is inside a channel — that number is the channel's permission, 2c/L.
- **Do not read a leg as passing because the report has no residual for it.**
  Check `environments_forced_by_timeout` first.
- **Do not edit the `elif` branch beside `_step_rigid_transit`'s caller.** It is
  dead for every run this branch makes.
- Geometry first, on the CPU. `check_workcell_geometry.py` answers in a second
  what a simulator sweep answers in an hour, and it is validated against the
  simulator so its answers are usable.

## How to work

- Plain English, short sentences, numbers where they matter.
- Work quietly; report once at the end, in four or five sentences.
- Batch runs into one background command. Never poll a log. Before saying you
  are done, check `tasklist` for `kit.exe` **and** for orphaned `sleep.exe` from
  wait loops whose condition never came.

## Rules that do not change

- The **robot** carries the module. No world constraint, no teleport, no direct
  module pose write, no hidden carrier.
  `tests/test_robot_carried_contract.py` enforces it — keep it passing.
- `--base_rail_on_relocation` is the old world-mounted payload shuttle;
  `--robot_rail_on_relocation` moves the robot.
- Keep failed results, labelled honestly, and log supersessions in
  `evidence/RETRACTED.md`.
