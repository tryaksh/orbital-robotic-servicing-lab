# Verified state

Everything here is simulation evidence. No result on this page was produced on
real hardware.

## Stack

Native Windows 11, Isaac Sim 5.1, Isaac Lab v2.3.2 at
`37ddf626871758333d6ed89cf64ad702aef127d0`, bundled Python 3.11, RL-Games PPO,
RTX 5070 Ti Laptop GPU 12 GB. Zero gravity, 120 Hz PhysX, 30 Hz control.

Isaac Sim 5.1's published VRAM minimum exceeds this laptop's 12 GB; use
benchmark-driven environment counts.

## Promoted policies

Checkpoints are local and intentionally ignored by Git. Publish them through a
GitHub Release, not Git history.

### Level 0, collision-free insertion, promoted 2026-08-07

```text
logs/rl_games/zero_g_blade_insertion_rigid_grasp/rigid_grasp_l0_fresh_seed60/nn/
last_zero_g_blade_insertion_rigid_grasp_ep_700_rew_74.81321.pth
```

SHA-256 `1635C0DA6464A34DE5D5D423D45D272AD0E19D808EA35B8502068F68B5043332`.
Trained from scratch at seed 60.

Deterministic evaluation on unseen seeds 1060, 2060, 3060; 128 parallel
environments; nine runs:

| Reset distance | Result | Terminal axial p95 / max | Cycle time p50 |
| --- | ---: | ---: | ---: |
| Near / stage 0 | 3,052 / 3,052 (100%) | 4.37 / 4.81 mm | 1.17 s |
| Medium / stage 1 | 3,032 / 3,032 (100%) | 9.31 / 9.34 mm | 3.67 s |
| Full / stage 2 | 3,002 / 3,002 (100%) | 11.95 / 11.96 mm | 7.90 s |
| Total | 9,086 / 9,086, Wilson 95% lower bound 0.9996 | — | — |

Report: `evidence/rigid_grasp_l0_ep700_certification.json`.

### Level 1, physical wide side rails, promoted 2026-08-08

`rigid_grasp_l1_wide_rails_seed61` fine-tuned the epoch-700 checkpoint for 500
more PPO epochs at 512 environments, seed 61, robustness level 1: physical wide
side-rail collision plus doubled reset joint noise. Reward went 56.6 at epoch
800, through 78.2 at epoch 1000, to 74.4 at epoch 1200. The dip is the contact
shock; the recovery is re-adaptation.

Deterministic evaluation of epoch 1200 on unseen seeds 1061, 2061, 3061:

| Reset distance | Result | Terminal axial p95 / max | Cycle time p50 |
| --- | ---: | ---: | ---: |
| Near / stage 0 | 3,006 / 3,006 (100%) | 2.32 / 3.07 mm | 1.23 s |
| Medium / stage 1 | 3,007 / 3,007 (100%) | 8.38 / 8.46 mm | 3.67 s |
| Full / stage 2 | 3,001 / 3,001 (100%) | 10.02 / 11.56 mm | 7.57 s |
| Total | 9,014 / 9,014, Wilson 95% lower bound 0.9996 | — | — |

Report: `evidence/rigid_grasp_l1_ep1200_certification.json`.

Contact fine-tuning improved terminal precision rather than merely preserving
success: stage-0 axial error fell from 4.15 mm mean at Level 0 to 1.65 mm at
Level 1. Terminal angular velocity roughly doubled, from 0.013 to 0.025 rad/s,
which is the expected signature of real rail contact and stays well inside the
0.080 rad/s limit.

### Level 2, tight rails plus randomized 5–15 kg mass, promoted 2026-08-08

`rigid_grasp_l2_tight_mass_seed62` fine-tuned the Level-1 epoch-1200 checkpoint
for 600 more PPO epochs at 512 environments, seed 62, robustness level 2: tight
1.5 mm side-rail clearance plus blade mass randomized over 5–15 kg with
recomputed inertia. Reward went 56.9 at epoch 1300 to 75.8 at 1400 and held
between 74.7 and 75.8 through epoch 1800.

Deterministic evaluation of epoch 1800 on unseen seeds 1062, 2062, 3062:

| Reset distance | Result | Terminal axial p95 / max | Cycle time p50 |
| --- | ---: | ---: | ---: |
| Near / stage 0 | 3,009 / 3,009 (100%) | 4.49 / 5.84 mm | 1.20 s |
| Medium / stage 1 | 3,012 / 3,012 (100%) | 9.41 / 10.12 mm | 3.30 s |
| Full / stage 2 | 3,000 / 3,000 (100%) | 9.70 / 10.59 mm | 7.20 s |
| Total | 9,021 / 9,021, Wilson 95% lower bound 0.9996 | — | — |

Success by blade mass band, over an observed 5.00–14.97 kg range:

| Mass band | Result |
| --- | ---: |
| Low | 3,114 / 3,114 (100%) |
| Mid | 2,889 / 2,889 (100%) |
| High | 3,018 / 3,018 (100%) |

Report: `evidence/rigid_grasp_l2_ep1800_certification.json`. Full-distance cycle
time improved to 7.20 s median from 7.90 s at Level 0, despite tighter clearance
and a threefold payload mass range.

**Margin warning.** Terminal orientation error at Level 2 reaches 0.0512 rad
against the 0.0524 rad success limit, which is 97.8% of the budget, at both
stage 1 and stage 2. Orientation is the axis with the least headroom and is the
most likely first failure mode when Level-3 rail stiction is enabled.

### What the three promotions do and do not show

All three runs had zero timeout, insertion-failure, mount-instability,
non-finite, and uncategorized terminations across 27,121 held-out episodes.
Terminal metrics are captured in `_reset_idx` before Isaac Lab's automatic
reset, so auto-reset cannot corrupt them.

They do not prove learned grasping, perception, cross-seed *training*
repeatability, or real transfer. Because every episode succeeded, the terminal
error distribution is bounded by the success criterion itself: it shows where
inside the tolerance box the policy lands, not accuracy independent of that box.
Margin is thin on axial depth at Level 0 (11.96 of 12 mm) and on orientation at
Level 2 (0.0512 of 0.0524 rad).

## Static validation

Ruff passes. 80/80 non-Sim tests pass. Sustained environment-only benchmarks
passed at 1024 state and 256 camera environments; full PPO memory differs.

`train.py --smoke` was run once per registered task after the 2026-08-10 prune,
at 32 environments (8 for vision). Nine of eleven pass; the two failures are
pre-existing and were confirmed to reproduce with identical numbers on the
unmodified pre-prune tree:

| Task | Level | Result |
| --- | ---: | --- |
| `Isaac-ZeroG-Blade-Insertion-v0` | — | pass |
| `Isaac-ZeroG-Blade-Insertion-Robust-v0` | 2 | pass |
| `Isaac-ZeroG-Blade-Insertion-Contact-v0` | 2 | **fail**, pre-existing |
| `Isaac-ZeroG-Blade-Insertion-RigidGrasp-v0` | 2 | pass |
| `Isaac-ZeroG-Blade-Insertion-ForceLimited-v0` | 2 | pass |
| `Isaac-ZeroG-Blade-Insertion-StrictForceLimited-v0` | 2 | pass |
| `Isaac-ZeroG-Blade-Insertion-ForceFeedback-v0` | 2 | pass |
| `Isaac-ZeroG-Blade-Insertion-GuidedSlot-v0` | 2 | pass |
| `Isaac-ZeroG-Blade-CaptureInSlot-v0` | 0 | **fail**, pre-existing |
| `Isaac-ZeroG-Blade-GrapplePin-Capture-v0` | 0 | pass |
| `Isaac-ZeroG-Blade-Insertion-Vision-v0` | 2 | pass |

**The contact task's scripted pull test fails**, and it is the inverted finger
command this page already records. The task commands pregrasp 0.80 rad and
closed 0.68 rad, but the measured convention is that zero is fully *open* and
the opening closes at 106.2 mm/rad, so 0.80 leaves a 2 mm gap and 0.68 leaves
15 mm: the "closed" command opens the fingers by 13 mm. Pulling then moves the
blade 13.7 mm the wrong way and opens a 24.3 mm grip error. The commands have
deliberately not been corrected, because changing them changes the physics three
promoted certifications were produced under.

**The capture-in-slot task fails its own smoke contract**, on a config
inconsistency rather than on physics: it declares `contact_grasp = True`, so the
smoke looks for a live `Handle` collider, while its rigid-grasp parent sets
`handle_collision_enabled = False`. The task never reaches a step, so its
recorded 6 N holding result stands unchanged and unaffected.

Both are recorded rather than fixed. Neither blocks the pose-uncertainty work,
and fixing the first would silently move the ground under three published
certifications.

**One smoke defect was fixed.** The scripted axial feasibility probe is now
scoped to the contact-grasp family it was written for. This page previously
recorded it exhausting its 300-step budget with 23.5 mm of residual axial error
on the rigid-grasp task while the learned policy inserts in 35 control steps. On
a rigid grasp the blade is welded to the tool, so axial feasibility holds by
construction and the probe tested nothing while failing; the rigid-grasp, force,
and vision tasks now smoke cleanly.

## Capability envelope

Certification says the policy works inside the distribution it trained on. It
says nothing about where that ends. These sweeps push the promoted Level-2
policy past its training range, one axis at a time, at full reset distance with
500 episodes per point on evaluation seed 7060. They are **measurements, not
certification**: the reports carry `evidence_type:
simulation_capability_envelope` and their promotion gate is marked
non-applicable.

### Initial pose error is the binding axis

Scale multiplies the trained reset joint-noise envelope of 0.001/0.002/0.004 rad
by stage. Blade mass is held at the trained range.

| Pose noise | 1× | 2× | 3× | 4× | 6× | 8× | 12× |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Success | 100% | 100% | 97.0% | 87.8% | 62.4% | 42.6% | 21.2% |
| Failures | 0 | 0 | 15 | 58 | 166 | 268 | 394 |
| Timeouts | 0 | 0 | 0 | 3 | 22 | 19 | 0 |

Report: `evidence/rigid_grasp_l2_envelope_pose_error.json`.

Degradation is monotonic and graceful, with roughly double the trained noise
absorbed for free and the half-success point near 7×. **Every point, including
12×, had zero instability and zero non-finite terminations**: the policy fails
by not completing the insertion, never by diverging numerically.

The failure mechanism is **lateral divergence, not orientation**. Terminal
lateral error p95 jumps from 0.0 mm at 3× to 60.6 mm at 4× and 92.9 mm at 12×,
tripping the 60 mm lateral failure predicate; timeouts stay a small minority.
This *contradicts* the earlier prediction from the Level-2 margin analysis, which
expected orientation to break first because it consumed 97.8% of its tolerance.
Orientation error does rise in failing episodes, but it is a symptom; the
termination is lateral.

### Blade mass is not a meaningful axis in this regime

Pose error held at the trained level; only the mass range varies.

| Mass range (kg) | 5–15 (trained) | 3–20 | 2–25 | 1–35 | 1–50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Success | 100% | 100% | 100% | 100% | 100% |

Report: `evidence/rigid_grasp_l2_envelope_blade_mass.json`. Terminal metrics are
nearly identical at every range (axial p95 about 9.7 mm, orientation p95 about
0.050 rad).

**This weakens the Level-2 claim and should be stated plainly.** "Robust to
5–15 kg payload mass" sounds substantial but is close to vacuous here: in zero
gravity, with the blade held by a fixed joint and the tool moving in
millimetre-scale quasi-static increments at 30 Hz, inertial forces are tiny, so
mass barely enters the dynamics. The Level-2 mass-bucket gate passed because the
task is insensitive to mass, not because the policy learned mass robustness. A
mass axis only becomes meaningful with faster motion, real grasp friction where
weight sets slip margin, or gravity.

## Contact force, the damage proxy

PhysX solved these contacts all along; until now nothing observed or reported
them, so the project could not answer the first question a servicing reviewer
asks: would this insertion damage the connector. `play.py --contact_metrics`
attaches a contact sensor to the blade and records peak contact force and
accumulated impulse per episode. It is off by default, so training throughput
and the trained policy are unaffected.

Promoted Level-2 policy, three stages by three held-out seeds, 4,513 episodes,
all successful:

| Metric | mean | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| Peak contact force (N) | 6.73 | 4.71 | 16.56 | 66.36 |
| Contact impulse (N·s) | 6.96 | 4.67 | 16.29 | 19.91 |

Peak force by reset distance:

| Stage | Episodes | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| Near / stage 0 | 1,503 | 4.06 | 6.70 | 9.75 |
| Medium / stage 1 | 1,510 | 5.10 | 9.02 | 17.49 |
| Full / stage 2 | 1,500 | 11.03 | 31.60 | 66.36 |

Report: `evidence/rigid_grasp_l2_contact_forces.json`.

**Contact load scales strongly with approach length.** Worst-case peak force
rises about sevenfold from the near start to the full start, and the single
worst full-distance episode reaches 66 N against a 9.75 N worst case at the near
start. Success rate hides this completely: every one of these episodes
succeeded. Nothing in the reward, the termination set, or the action space
currently bounds contact force, so the policy has no reason to prefer a gentle
insertion over a hard one.

These are simulated contacts against primitive rail geometry with no connector,
latch, chamfer, or measured force-displacement curve. Treat them as a relative
damage proxy for comparing policies, not an absolute force budget for hardware.

## Force-limited insertion: a negative result

Contact load was measured but not constrained, so two force-aware policies were
fine-tuned from the Level-2 checkpoint and all three were judged on the same
force-limited task, 4,500+ episodes each, three stages by three held-out seeds.
Observations and actions are unchanged throughout, so every policy is the same
network shape.

| Policy | Success | Aborts | Peak force mean | p95 | max | Impulse p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no force objective | 99.98% | 1 | 6.64 N | 16.73 N | 67.97 N | 16.10 N·s |
| Mild penalty (5 N free, 60 N abort) | 100% | 0 | 6.62 N | 16.83 N | 57.57 N | 16.11 N·s |
| Strict penalty (1.5 N free, 30 N abort) | 100% | 0 | 6.47 N | 16.06 N | 58.22 N | 16.18 N·s |

Reports: `evidence/force_limit_baseline_l2.json`,
`evidence/force_limited_ep2500_certification.json`,
`evidence/force_strict_ep2500_certification.json`.

**Reward shaping on contact force did not work here.** The mild profile was too
weak by construction: at the measured median contact of 4.7 N its penalty is
exactly zero, and at p95 it is 0.17 per step against a success reward of 35. The
strict profile fixed that arithmetic, charging about 19.5 per step at p95, the
same order as the success term. It still moved almost nothing: mean fell 2.6%,
p95 fell 4%, impulse and cycle time did not move at all. The only real change is
the worst case, and that is the abort termination clipping the tail rather than
the policy learning to be gentle.

Two explanations fit the data, and they compound:

1. **The policy cannot perceive what it is being asked to regulate.** Contact
   force is not in the observation space, so PPO can only reduce it by changing
   state-conditioned motion. If contact is close to a deterministic function of
   geometry and approach, there is no gradient to exploit. This is the ordinary
   robotics lesson that force control needs force feedback.
2. **Much of the contact looks irreducible for this action space.** Near-start
   episodes sit at about 4 N mean and never go lower across all three policies,
   which is consistent with a geometric floor: a blade crossing a 1.5 mm
   clearance slot under position-based differential IK has to touch the rails.

The next experiment follows directly: put contact force into the observation
space, or replace position-based IK with an admittance or impedance action
space, and retrain rather than fine-tune, because either change alters the
policy interface. Do not simply raise the penalty weight again; the strict
profile already shows that is not the binding constraint.

## Force feedback: the diagnosis was half right

The force-shaping negative result above left two hypotheses: the policy cannot
regulate a force it cannot sense, and some contact is geometrically
irreducible. Both were tested at once by adding contact force to the
observation vector and retraining.

Seven values were added to the 50-value observation, taking it to 57: the
contact force in tool axes, the same force through a 100 ms first-order filter
standing in for a real force/torque signal chain, and the exact scalar the
penalty and the abort key on. Nothing else changed. Because the observation
width changed, no earlier checkpoint could be resumed.

A matched **control** was trained as well: the identical strict force task with
the observation left alone. Both arms were trained from scratch on the same
L0 → L1 → L2 schedule that produced the promoted policy (700 / +500 / +600 PPO
epochs, 512 environments, training seed 65, one shared PPO configuration), so
the only difference between them is whether the policy can sense contact. The
control exists because the earlier force policies were *fine-tuned*; without it,
any difference could be attributed to training from scratch instead.

Both arms were judged under the 60 N abort limit every earlier force policy was
measured under, on held-out seeds 1065/2065/3065:

| Policy | Episodes | Success | Aborts | Peak force mean / p95 / max | Impulse mean / p50 / p95 | Cycle p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control, no force feedback | 4,524 | 100% | 0 | 6.31 / 15.78 / 59.71 N | 7.55 / 6.26 / 16.55 N·s | 3.30 s |
| Force feedback | 4,518 | 99.93% | 3 | 6.24 / 14.70 / 66.10 N | 3.06 / 0.70 / 9.94 N·s | 3.47 s |

Reports: `evidence/force_feedback_control_certification.json`,
`evidence/force_feedback_certification.json`.

**Force feedback moved contact impulse and did not move peak force.** Impulse
fell 59% at the mean, 89% at the median, and 40% at p95, while peak contact
force changed by roughly 1% at the mean and stayed identical per stage (full
distance: 10.55 N mean without feedback, 10.48 N with). Mean cycle time was
unchanged, 3.87 s against 3.89 s, so this is not speed traded for gentleness.
No earlier intervention moved impulse at all: it sat at 16.1, 16.1, and 16.2 N·s
p95 across the baseline and both penalty strengths.

That splits the two hypotheses cleanly, and **both turn out to be correct about
different quantities**:

- *Sensing was the binding constraint on sustained contact.* Impulse is the
  accumulated force-time the blade spends rubbing the rails. Once the policy
  could see contact, it learned to stop rubbing, and the median episode now
  delivers 0.70 N·s where the control delivers 6.26.
- *Peak force is geometrically irreducible in this action space.* The first
  strike as the blade crosses a 1.5 mm clearance slot under position-based
  differential IK did not change. Testing that further needs the other half of
  roadmap item 6, an admittance or impedance action space, not more sensing.

The cost is honest and small: three force-limit aborts in 4,518 episodes, all at
full reset distance, so the force-feedback policy is 99.93% rather than 100% and
its worst case is slightly worse (66.10 N against 59.71 N). It clears the 0.95
promotion gate with zero instability and zero non-finite terminations, but it no
longer holds the perfect record the earlier policies hold. A policy that
regulates sustained load and occasionally trips a safety limit is the more
useful servicing behaviour, but that is a judgement, not a measurement.

## Learned grasping is blocked: the handle sits past the fingertips

`docs/status.md` previously recorded only that the real Robotiq pad/handle
contact task "failed its axial pull gate". `scripts/grasp_diagnostics.py`
replaces that pass/fail with a measurement, and the measurement says the gate
was never close to passing for a reason nobody had looked for.

The script holds the arm still, closes the fingers, then applies a constant
axial pull to the blade. Environment `i` takes one point of a 4 closure x 32
force grid from 0 to 120 N, so 128 environments sweep the surface in one run.

| Quantity | Measured |
| --- | ---: |
| Peak gripper drive torque at the configured grasp pose, against a 10 N·m limit | 0.39 N·m |
| Pad-to-blade contact force reported by PhysX, over the full 0–0.8203 rad finger range | 0.0 N |
| Handle distance from the flange at which the fingers *do* obstruct on the blade | 0.06–0.15 m |
| Handle distance the task actually configures (`tool_offset_pos`) | 0.179 m |
| Axial force held before slip | 0 N |
| Axial force the promoted insertion policy's contact reaction demands | 66.4 N |

Report: `evidence/grasp_axial_pull_gate.json`.

**There is no grasp to characterise.** The solver says so directly: with a
contact sensor filtered to the two inner-finger bodies, PhysX reports exactly
0.0 N between pad and blade at every commanded closure across the full
`finger_joint` range of 0 to 0.8203 rad, and the joint reaches every commanded
value exactly with zero drive torque, so nothing obstructs the pads anywhere in
their travel. That measurement does not depend on where the pad surfaces are
assumed to be. The fingers never touch the handle, and the blade is a
free-floating body in zero gravity. Its motion under load confirms this arithmetically rather than by
inspection: 120 N on a 10 kg blade for 1.5 s predicts 13.5 m of free travel, and
the measurement is 12.5 m. The earlier "failed pull gate" was not a weak grip.

**The cause is a reach error: the handle is configured beyond the fingertips.**
Sweeping the handle along the tool axis while the fingers close on it saturates
the 10 N·m drive limit for handle distances between roughly 0.06 and 0.15 m from
the flange, so the fingers genuinely obstruct on the blade in that band. The
task configures 0.179 m, where drive torque sits at 1e-5 N·m: the fingers close
past the handle and never touch it.

An earlier revision of this page attributed the failure to a "165.6 mm
tool-frame calibration error", derived from the distance between the tool frame
and the finger *body origins*. That was wrong and has been retracted. Every
2F-85 body in this asset — base, both knuckle pairs, both fingers — is collapsed
to within 18 mm of the flange, with `base_link_0` exactly coincident with
`wrist_3_link`, so those origins say nothing about where the pad surfaces are.
The zero-contact result was never in doubt; only the explanation for it was.
Three consequences:

- **`tool_to_handle_error_m` cannot detect a grasp problem.** In the rigid-grasp
  task it measures exactly 0.0000 m, because the fixed joint welds the blade to
  the tool frame that the metric compares against. It is a self-consistent
  tautology, not an audit of a grip. `docs/claim_vs_evidence.md` already said
  the near-zero value is "a property of that joint, not of a grip"; that is now
  confirmed to the last decimal place.
- **The blade is welded where the fingers are not.** The fixed joint holds the
  blade at the tool frame, 0.179 m from the flange, while the fingers only reach
  to about 0.15 m. Insertion is unaffected, because it depends on
  blade-versus-slot geometry, but the render shows a gripper that is not
  actually holding the blade it appears to carry.
- **The fix cannot be a one-line edit to the shared constant.** Because the
  promoted insertion tasks read the same `tool_offset_pos`, and that frame
  appears in the policy observation through `end_effector_pose_local`, changing
  it would move every promoted checkpoint's observation values and invalidate
  three certifications. The grasp task needs its own corrected offset, leaving
  the insertion tasks on the value they were trained against.

A second defect sits behind the first and will bite the moment it is fixed: the
finger command runs backwards. Pad separation grows monotonically with the
commanded value, from 0 mm at 0.00 rad through 42.7 mm at 0.45 to 58.9 mm at
0.60, and the joint's own limits are 0 to 0.8203 rad. Zero is therefore the
closed end. The task calls 0.45 "pregrasp" and 0.60 "closed", so what it treats
as closing the gripper opens it by 16 mm.

**What this does and does not invalidate.** It does not touch the promoted
Level-0/1/2 insertion results. Those depend on blade-versus-slot geometry, rail
contact, and blade dynamics, all of which are real, and the fixed joint is
documented throughout as an abstraction rather than a grasp. What it invalidates
is the assumption that the contact-grasp task was a nearly working grasp needing
tuning. It is not: the gripper and the handle are not in the same place, so
Phase-3 grasping starts with a geometry fix, not with PPO.

## Grasp fix: a grip now forms, and it throws the blade away

Three defects were found and corrected, and the correction moved the failure
from "nothing ever touches" to a different and more interesting one.

**What was wrong.** The handle was a 30 mm tab lying on the blade's 160 mm-wide
deck, while the gripper's pads open to 82 mm. The pads foul the deck long before
they can straddle a tab that short, so a grip was geometrically impossible. The
configured tool offset of 179 mm hid this by placing the control frame in open
space *below* the blade, where nothing could collide. Separately, the finger
command ran backwards: `finger_joint` limits are 0 to 0.8203 rad with 0 fully
closed, so the task's "pregrasp 0.45 / closed 0.60" pair opened the fingers by
16 mm when it meant to close them. And the action term still steered the old
179 mm frame while the observations and grasp metrics had been moved, so the
policy would have controlled a point 74 mm from the one it was scored on.

**What changed.** The grasp task now carries a raised grapple post, 75 mm wide
by 60 mm deep by 150 mm tall, rooted 1.7 mm inside the deck and reaching to
150 mm above the blade centre. Its mid-height coincides with the finger pads at
the unchanged arm reset pose, so no joint angles had to be re-derived. The
finger commands were inverted to match the measured convention, and the action
term's control point was moved onto the same frame the metrics report. The
promoted insertion task is pinned to its original tool frame, finger commands,
and short tab, so its three certifications still describe the task in the file.

**Result.** Drive torque rose from 0.39 N·m, the noise floor, to the actuator's
full 10 N·m limit in 124 of 128 environments. The fingers are loaded against the
post for the first time in this project's history; a grip forms.

**And the grip then ejects the blade.** With no pull applied at all, the blade
travels 32 mm and the tool-to-handle error grows by up to 180 mm, after which
drive torque falls back to zero because there is nothing left between the pads.
The closed command asks for 67 mm across a 75 mm post, and PhysX resolves 8 mm
of interference on an unconstrained 10 kg body by launching it.

That is a real physical result, not a bug: **you cannot grasp a free-floating
mass in zero gravity by squeezing it.** Any asymmetry in contact timing becomes
net momentum, and there is no gravity or fixture to absorb it. Terrestrial
grasping quietly relies on the object resting on something.

The next attempt therefore has to change the *task*, not the tuning: capture the
blade while it is still constrained by its slot or caddy, so the rails absorb
the closing asymmetry, and only then break it free. Reducing the interference
and softening the drive are worth trying alongside, but they treat the symptom.
Holding capacity remains unmeasured, because a grasp that ejects its payload has
no capacity to report.

Report: `evidence/grasp_axial_pull_gate.json` (pre-fix baseline) and
`scripts/calibrate_grasp_pose.py` for the kinematic check.

## Guided slot and capture-in-slot: built, gated, not yet passing

Two new tasks, both separate registrations so the certified Level-0/1/2 geometry
is untouched: `Isaac-ZeroG-Blade-Insertion-GuidedSlot-v0` and
`Isaac-ZeroG-Blade-CaptureInSlot-v0`.

**The guided slot turns two walls into a channel.** Overhanging upper lips sit
1.0 mm above the blade deck, spanning 62.5 to 82.5 mm either side of centre so
they overlap the blade edge by 17.5 mm while leaving the middle clear for the
grapple post. Two 80 mm plates at the mouth, each rotated 12 degrees, widen the
lateral catch from 0.75 mm per side to 16.6 mm per side, and carry the lowest
friction in the slot so a lead-in guides rather than grabs. With about 1 mm of
lift available at each end of a 450 mm blade, pitch is mechanically limited to
roughly 0.0044 rad against the 0.052 rad the policy currently has to control by
itself. The channel is meant to do the alignment the reward has been doing.

**Capture-in-slot closes the grasp while the rails still hold the blade.** This
needed no new arm pose: the stage-0 reset already parks the blade 31 mm short of
fully inserted, entirely inside the rails, with the grapple post under the pads.
Only the rails had to be made solid at level 0, where the parent profile turns
them off.

It measurably helped. Against the free-floating grasp, peak drive torque became
a controlled 1.4 to 2.2 N·m instead of saturating the 10 N·m limit, and slip
under near-zero load fell from 118 mm mean to 45 mm.

**It does not yet pass the gate**: axial holding capacity is still about 6 N
against the 66.4 N required, with 124 of 128 environments slipping.

The reason is structural and worth stating plainly. The rails constrain the
blade sideways and the new lips constrain it vertically, but **the extraction
axis is the one direction a slot must leave free**, and it is exactly the
direction the pull test loads. Nothing but friction can resist it. Grip force is
currently far too low to supply that friction: closure targets of 0.70 to 0.77
rad sit within a millimetre or so of the 75 mm post width, so the fingers barely
squeeze, which is why torque is 2 N·m out of an available 10.

The next attempt is a closure sweep between roughly 0.62 and 0.70 rad with the
rails solid. Free-floating, 0.68 rad ejected the blade at 10 N·m; constrained,
that interference may be exactly what supplies the missing friction. This is a
tuning question with a measured bracket at both ends, not an open one.

## The 2F-85's own geometry, measured instead of assumed

Twice now this project has designed against guessed gripper geometry and had to
retract the result. `scripts/measure_gripper_envelope.py` replaces the guessing:
it collects every prim carrying `UsdPhysics.CollisionAPI` under each gripper
body, expresses those collision meshes in the body's own frame, and carries them
out through the pose PhysX reports into the `wrist_3_link` frame the tool offset
is measured in. It never reads a body origin as a pad location.

Report: `evidence/gripper_collision_envelope.json`.

| Quantity | Measured |
| --- | --- |
| Closing axis | wrist x |
| Approach axis | wrist z, matching every tool offset in this project |
| Clear opening, `finger_joint` 0 | 87.08 mm |
| Clear opening, `finger_joint` 0.8203 (the joint limit) | 0.0 mm |
| Closing rate | 106.2 mm/rad |
| Finger pads, distance from the flange | 105 to 162 mm |
| Palm face, distance from the flange | 90 mm |
| Gripper envelope about the tool axis | 155 mm closing, 75 mm across |

**Zero is fully open.** The opening falls monotonically with the command over
the joint's whole range, and 87 mm at the open end matches the hardware's 85 mm
stroke. This contradicts what the code recorded until now, which had zero as the
closed end, and it means the "pregrasp 0.80 / closed 0.68" pair the contact task
carried was still inverted after the earlier correction: it opened the fingers
by 14 mm where it meant to close them.

**The throat behind the pads is full.** Slicing the same point cloud by depth
shows the inner knuckles sweeping through the 90-to-105 mm gap between the palm
face and the pad trailing faces, reaching within 8 to 24 mm of the tool axis
depending on closure. That kills the obvious capture interface before it is
built: a mushroom head with a flat shoulder has to sit in exactly that gap to
bear on the pads, has to be wider than the pad gap to catch them, and has to be
narrower than the knuckles to fit. No closure satisfies both.

## Head-on grapple pin: the gate passes at 69 N

A parallel-jaw grip cannot hold this blade against extraction, and that is
structural. The gripper closes along one axis while the blade leaves along
another, the rails must leave the extraction axis free, and flat pads on a
smooth post can then oppose the pull only by friction: about 6 N against the
66.4 N the promoted Level-2 policy's own contact reaction demands.

Putting geometry on the pull axis means approaching along it, which moves the
interface onto the module, as the ISS ORU standard does. The blade now carries a
tapered grapple pin on its `-x` face and the gripper takes it head-on:

| Section | Extent along the pin | Size |
| --- | --- | --- |
| Shaft | 80 mm from the blade face | 30 x 30 mm |
| Collar | 6 mm | 90 mm tall, 30 mm wide |
| Wedge | 60 mm | 70 mm tapering to 16 mm, 24.2 degrees |

The 80 mm shaft is not padding. The pads are 57 mm long and the blade's front
face sits 75 mm inside the rack mouth when fully inserted, so anything shorter
would put the gripper inside the slot, where it fouls the floor plate. The
collar is taller than the 87.08 mm the pads can ever open to, so it is an
absolute depth stop rather than something a wide-open gripper slides past, and
it gives the insert direction a face to push on.

Arm pose: `scripts/calibrate_grasp_pose.py`, extended here to servo orientation
as well as position, converged all three curriculum stages to under 0.01 mm and
0.00003 rad with the tool's approach axis along world +x and its closing axis
vertical. Vertical closure is what keeps the gripper's narrow 75 mm dimension
between the rails. Report: `evidence/grapple_pin_head_on_pose.json`.

**A real grip forms, and it holds an order of magnitude more.**

| Quantity | Flat pads on a post | Head-on grapple pin |
| --- | ---: | ---: |
| Environments where a finger was blocked | 124 / 128 | 363 / 363 |
| Peak drive torque against the 10 N-m limit | 10.0 N-m, then ejection | 10.0 N-m, seated |
| Axial force held within 2 mm of slip | about 6 N | **69 N** |

**The gate passes**, 69 N against the 66.4 N required, on a 3 closure by 121
force grid at 1 N resolution. Report:
`evidence/grapple_pin_axial_pull_gate.json`.

### Capture and hold are two different commands

Getting there needed one idea, and it is the interesting result of this
section. A single closure command, however chosen, caps out at 59 N. Capacity
*falls* as the fingers close harder, from 59 N at 0.56 rad to 43 N at 0.60 and
24 N at 0.64, with drive torque saturated at 10 N-m throughout. That is not the
old failure of fingers closing on air: the pads stop at about 0.22 rad in every
case, blocked by the wedge, so the grip is real.

The mechanism is the wedge itself. It converts closing force into thrust along
the pull axis, which is the one axis a slot has to leave free, so a firm capture
drives the payload away before it has been taken. Holding, once the pin is
seated against the collar, wants the opposite: everything the drive can produce.

Splitting the two settles it. Capturing at 0.48 rad and firming to 0.68 once the
grip is established holds 69 N, and median axial slip falls from 1.1 mm to
0.7 mm:

| Capture | Hold | Force held |
| ---: | ---: | ---: |
| 0.44 | 0.64 | 63 N |
| 0.48 | 0.68 | **69 N** |
| 0.52 | 0.72 | 68 N |
| 0.56 | 0.76 | 26 N |

Report: `evidence/grapple_pin_capture_plateau.json`. The window is narrow and
asymmetric, so the capture command is biased low, and `TwoStageRobotiqAction`
implements the two stages inside the action term rather than leaving them to a
script.

### Why the collar was the weak link

A new measurement explains it. Under load the fingers are forced back open by
0.055 rad at p95, so the payload **cams the pads apart** rather than sliding
between them, and the drive's stiffness is what resists. That is why more
holding force helps and why a violent capture, which leaves the pin poorly
seated, does not.

The remaining give is rotational rather than axial. Slip decomposes into 0.7 mm
of median axial movement against 0.054 rad of angular movement, and the blade
only starts to lever once the pull has dragged it clear of the rails that were
constraining it. That is a property of the pull test, which loads a body the
rails do not constrain along x, more than of the interface.

## Grip force is not the binding constraint, and that was worth testing

The obvious reading of the 59 N result is that the gripper is too weak. Axial
capacity through a wedge is proportional to pad normal force, the drive's
effort limit caps that force, and this project inherited a 10 N-m limit that
yields about 100 N per pad by virtual work against the measured 106.23 mm/rad
closing rate. Robotiq specifies 20 to 235 N for the 2F-85, so the simulation was
modelling the gripper at well under half its rated strength.

That is a real fidelity gap, so it was closed and measured: the drive limit was
raised to 235 N x 0.10623 m/rad = 24.96 N-m, scoped to the grapple-pin task
alone so the contact and rigid-grasp tasks keep the actuator their published
results were produced under. Judged on a matched 2 N grid:

| Drive limit | Pad force | Best force held | Capture above 0.65 rad |
| --- | ---: | ---: | --- |
| 10 N-m, inherited | about 100 N | 66 N | holds |
| 24.96 N-m, rated | about 235 N | 62 N | lost entirely |

Report: `evidence/grapple_pin_rated_grip_force.json`.

**Two and a half times the grip force made the capture worse.** Capacity fell,
and above 0.65 rad of closure every environment exceeded the slip tolerance
before any pull was applied at all. The mechanism is the wedge itself: it
converts closing force into thrust along the pull axis, which is exactly the
axis a slot has to leave free, so in zero gravity a harder squeeze drives the
payload rather than gripping it. Axial slip at the median rose from 1.1 mm to
5.7 mm and angular slip from 0.043 to 0.133 rad, all of it spent before the
measurement began.

One methodological note, because it nearly cost the result: the first attempt
raised drive stiffness from 40 to 80 N-m/rad alongside the torque limit, on the
argument that a position-controlled proxy needs gain to reach rated force inside
the joint's range. That measured 0 N held at every closure. Separating the two
recovered 62 N, which is what the table reports. Changing one thing per
experiment is a rule in this repository for a reason.

**What is left.** The gate threshold does not move. Pad force is refuted, and
steepening the taper is nearly exhausted, since capacity goes as the sine of the
taper angle and 24.2 degrees already puts the wedge's free end at 70 mm inside
an 87.08 mm aperture with 8.5 mm of approach clearance a side. The measurement
points instead at rotation: the collar holds along the pull axis at 1.1 mm of
median axial slip while the blade levers at 0.166 rad p95, and it only levers
once the pull has dragged it clear of the rails that were constraining it. An
interface feature that opposes yaw, or an extract skill that keeps the blade
railed for longer, attacks the failure that is actually being measured.

Grasp, extract, and insert are therefore not trained. The order of work puts the
pull gate before PPO precisely so that a policy is never trained against an
interface that cannot hold the load.

## Three skills: built, measured, and deleted

Grasp, extract, and insert were implemented as three separately gated tasks on
the head-on capture scene, trained, and **deleted on 2026-08-10** with the
eight-phase swap task. Their measurements are kept below because they are what
justified the deletion.

| Task | Starts | Succeeded when | Measured |
| --- | --- | --- | --- |
| `GrapplePin-Grasp-v0` | Head-on, 10 to 50 mm of pose error | Drive torque loaded, tool on the grip point, held 0.3 s | 99.3% at stage 1, 35% at stage 2 — and the first number was hollow |
| `GrapplePin-Extract-v0` | Captured, blade in the slot | Blade centre at x 0.225, so its rear face clears the mouth | 0/1024, blade travelled 71 mm of 494.5 mm |
| `GrapplePin-Insert-v0` | Captured, at the certified staging pose | The insertion predicate, with a physical grip | 0/1024 at stages 1 and 2 |

Three design points were deliberate and are worth carrying forward. Only the
grasp skill commanded the gripper, because a policy that cannot choose when to
close is not learning to grasp. The observation carried finger angle *and* drive
torque together, because the angle alone cannot distinguish fingers closed on a
pin from fingers closed on nothing, which is exactly the failure this project
did not see for three sessions. And extraction carried its own workspace
predicate, because `insertion_failure` treats a blade below x 0.45 as an escape
and that is where a successful extraction ends.

Getting those first runs to take a step surfaced four faults worth recording,
because three of them would have silently degraded training rather than
crashing:

- The potential rewards zeroed their baseline on reset, charging the first step
  of every episode the whole distance from the goal.
- The approach reward scored the *live* blade, so a policy could be paid for the
  blade drifting toward the tool, which in zero gravity it can cause by shoving
  it. It was changed to score against the pose the blade was placed at.
- Both were suppressed for 0.30 s after a reset, because a reset writes joint
  positions but leaves the previous episode's actuator targets in place, so the
  arm springs before it holds.
- Extract and insert faked their starting grip by writing the fingers to their
  seated angle. That places the pads around the wedge without preloading the
  collar, and a 40 mm pull then moved the blade 0.1 mm. They were changed to run
  the real two-stage capture inside a 1.0 s settling window.

`TwoStageRobotiqAction` and `hold_two_stage_grip` survive the deletion, in
`mdp/grapple.py`, because they implement the capture/hold split that the
interface specification's 69 N result depends on.

## First training runs: two mis-specified tasks, found by reading past the headline

Both first attempts produced a number that looked like a result and was not.
Both are kept here, because they are the measurements that found the faults and
because two of the three are exactly the pathologies established work already
solves.

### Grasp v1: 99.3% that meant nothing

The first grasp policy measured 99.3% capture at curriculum stage 1 and 35% at
stage 2. The cycle times say why the first number is hollow: successful captures
completed at a **median of 0.30 s and a 95th percentile of 0.77 s**, which is
the first handful of control steps at 30 Hz. The reset was landing the tool
13.8 mm from the grip point, already inside the 20 mm capture tolerance, so the
policy never had to approach anything. It only had to decide to close.

Stage 2 was the only stage whose reset noise put the tool outside that
tolerance, and nothing in the earlier stages had taught an approach, so it timed
out in 588 of 1003 episodes. Those timeouts are worth reading precisely: the
capture-failure predicate almost never fired, so the policy was not diverging or
shoving the blade away. It was sitting still, not converging.

**This is precisely the overfitting IndustReal's sampling-based curriculum
exists to prevent**: an agent exploiting a partially-solved initial state. SBC
exposes the whole initial-state range from the first step and raises only its
easy bound as success improves, so a reset that solves the task can never become
the whole training distribution. The curriculum in this repository does the
opposite, ramping *into* the hard stages through mixtures. Adopting SBC is the
first concrete change the pivot makes.

### Extract v1: 0% success with a grip that never let go

The first extraction policy timed out in 1024 of 1024 episodes. The grip was
never the problem: tool-to-grip error held at 6.7 mm at the median through every
one of them, so the head-on capture carried the load for the whole 15 s. The
blade simply did not travel, reaching a median of **71 mm of the required
494.5 mm**, at 4.7 mm/s against an available 120 mm/s.

The training reward climbed monotonically from 3.3 to 26.0 and had not levelled
off at epoch 700, which is the signature of a policy still learning when the
budget ran out rather than one that has converged on failure. A 495 mm pull is
three and a half times the certified insertion distance and needs about 124
consecutive near-maximum steps before any reward for finishing arrives. That is
a credit-assignment horizon, not a physics result.

**Neither of these was a reach failure.** The extract skill's end pose, with the
wrist folded about 200 mm in front of the robot's own base, was never verified
kinematically, because no policy ever pulled far enough to reach it.

## The pivot: the tasks contained no uncertainty

Decided 2026-08-10. Every RL task in this repository trains against a task with
**no uncertainty in it**. The policy observes `insertion_goal_error`, derived
from `attached_blade_pose_world`, which is simulator ground truth: reset noise
randomizes the initial condition and the policy is then told the exact resulting
error. With a rigid known object on a constrained axis and full observability,
that is a motion-planning and force-control problem, and a scripted controller
would solve it. RL cannot demonstrate its value there, which is why three
hand-rolled skills cost a night of GPU and certified nothing.

The three skills, the eight-phase swap task, and their reward, termination, and
curriculum classes were deleted. What was kept: the grapple pin geometry and
`docs/service_interface_spec.md`, everything in `evidence/` including the
negative results, the contact-force machinery in `mdp/insertion.py`, the
evaluator and its promotion gate, the capture scene the interface spec was
measured on, and the visual-randomization machinery — which was reachable only
from the deleted swap task and is now repointed at the insertion scene as
`Isaac-ZeroG-Blade-Insertion-Vision-v0`. No policy has been trained on that
task; it is scaffolding, and it is labelled as such.

The next result is one falsifiable plot: **success rate against pose-belief
error, force-aware policy versus force-blind ablation**, which is the axis
IndustReal and FORGE are evaluated on and for which this repository already owns
both halves — a working `BladeContactWrenchObservation` and a certified
force-feedback task lineage.

## Insertion under a wrong pose belief: the ablation refutes the hypothesis

The task the pivot exists for, and its result is negative: **force feedback did
not extend the pose error this policy tolerates, and beyond the trained range it
made it worse.** The mechanism, the geometry it forced, the faults found while
building it, and the measured curve are all below.

**The first construction was fake, and measuring it is what showed that.** The
obvious way to inject pose uncertainty is to add a bias to the reported goal
error. On this workcell that is recoverable: the blade is welded to the tool by
the fixed joint, so its pose is a constant offset from the tool frame; the tool
frame is observed directly as `end_effector_pose_local`; and with a fixed goal
the true error is therefore an exactly learnable function of an observation the
actor already has. A network can compute the truth and ignore the bias, and both
arms of the ablation would have scored identically for reasons having nothing to
do with force.

**So the slot physically moves.** Each episode displaces the guide rails, the
upper lips, and the lead-in flares laterally by a magnitude drawn from the
curriculum and a random sign, and the seated goal moves with them. Nothing in the
observation determines that displacement. Verified in the simulator: environments
whose tool poses agree to 1.5 mm disagree about the true lateral error by 5.2 mm.

**The channel had to be relocated, and that is the awkward part of this result.**
The certified slot runs from x = 0.45 to 1.05 and the blade is 450 mm long, so at
*every* reset distance the promoted tasks use the blade already sits inside the
rails, its front face 358 mm past the mouth, with 0.75 mm of clearance per side.
Displacing the rails in place would start the episode with the blade inside a
rail. The mouth is therefore derived to sit 32 mm ahead of the blade's front face
at the staging reset:

| Quantity | Value |
| --- | ---: |
| Blade front face at the staging reset | 0.8079 m |
| Flare opening rate at 12 degrees | 0.2126 m/m |
| Opening a 4 mm offset needs | 3.25 mm |
| Minimum lead-in ahead of the blade | 15.3 mm |
| Lead-in used | 32 mm |
| Resulting channel mouth | 0.8399 m |
| Channel length | 0.2101 m |
| Blade-to-flare clearance at full displacement | about 3.6 mm |

Two consequences are costs, not features. The channel is 210 mm long instead of
600, so a seated blade is engaged over 135 mm rather than its full length. And
there is **one reset distance, not three**: the two nearer stages would start the
blade engaged again, so the stage curriculum is collapsed to a single level.

**The displacement is lateral, and the axis choice is the experiment.** A depth
offset is resolvable without any force sensor: the blade bottoms out and
`blade_velocity` is in both actors' observations, so a force-blind policy finds
the stop as easily as a force-aware one. A vertical offset is unsolvable, because
Level 2 disables the floor collider, and it would still count against the 2.5 mm
lateral success tolerance. Lateral is the only axis where both policies can tell
they have stalled but only a force-aware one can tell **which side**, because
that is carried by the direction of the contact force and by nothing else.

**Training runs at robustness level 2 only.** Level 0 disables rail collision
entirely and level 1 leaves 6 mm of clearance per side, so at both a displaced
channel is untouchable and a policy would spend most of its budget learning that
contact carries no information. The promoted L0 to L1 to L2 walk is right for a
task where contact is a nuisance and wrong for one where contact is the signal.

**What is adopted rather than invented**, with the reason each fixed something
this repository had already measured:

| Adopted | From | Replaces |
| --- | --- | --- |
| Sampling-based curriculum over the displacement | IndustReal (RSS 2023) | Stage mixtures, which produced the 99.3%-in-0.30 s policy |
| Per-episode force threshold, linear hinge penalty | FORGE (arXiv 2408.04587) | Two fixed quadratic profiles measured as ineffective |
| 1 N noise floor on the observed contact force | FORGE and arXiv 2604.19677 | An idealized, noiseless sensor |
| Asymmetric actor-critic | Both | A critic that saw only what the actor saw |

**Refused, deliberately:** force-direction prediction (arXiv 2602.14174) and
hybrid position/force mode selection (arXiv 2604.19677). Both are action-space
changes. Changing the action space and the observation in one experiment makes
the result unattributable; they are the next experiment, and they are the
untested half of roadmap item 7.

**The ablation.** Two policies, from scratch, one PPO configuration, one seed,
one schedule. The force-aware actor sees 58 values and the force-blind one 51,
differing by exactly the seven contact values, against an identical 71-value
critic. `tests/test_belief_curriculum.py` asserts that difference by parsing the
configuration, so the two cannot silently drift apart.

### The result: force sensing did not buy robustness to pose error

Both arms trained to convergence, 1800 PPO epochs each at 512 environments,
robustness level 2, seed 80, one shared configuration. Evaluated on three
held-out seeds (1080/2080/3080) at seven displacements, 33,500 episodes in total.

| Slot displacement | Force-aware | Force-blind |
| ---: | ---: | ---: |
| 0 mm | 100.00% | 99.94% |
| 1 mm | 100.00% | 99.83% |
| 2 mm | 100.00% | 99.83% |
| **4 mm** (trained ceiling) | **99.87%** | **99.77%** |
| 6 mm | 96.94% | 99.56% |
| 8 mm | 87.50% | 94.90% |
| 10 mm | 74.07% | 82.31% |

Reports: `evidence/uncertain_insertion_aware_certification.json`,
`evidence/uncertain_insertion_blind_certification.json`, and the two
`*_envelope.json` files, which carry `evidence_type:
simulation_capability_envelope` because they sweep past the trained ceiling.

**The prediction was that the force-aware policy would hold out further. It does
not. Beyond the trained range it is consistently and substantially worse**, by
2.6 points at 6 mm, 7.4 at 8 mm, and 8.2 at 10 mm, in the same direction on every
one of the three evaluation seeds. Both arms certify at the trained 4 mm
displacement, where they are indistinguishable, so the gate passes for both and
the interesting part of this result is entirely outside it.

Contact load explains the direction, and it is the second surprise:

| Displacement | Aware peak p95 | Blind peak p95 | Aware impulse p95 | Blind impulse p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0 mm | 25.41 N | 12.02 N | 5.25 N·s | 4.90 N·s |
| 4 mm | 27.65 N | 13.72 N | 5.72 N·s | 5.15 N·s |
| 8 mm | 47.47 N | 20.43 N | 79.65 N·s | 7.41 N·s |
| 10 mm | 44.98 N | 30.93 N | 82.58 N·s | 80.72 N·s |

**The policy that can feel contact pushes about twice as hard at every
displacement**, and its failures are force-limit aborts and timeouts while it
grinds. Median cycle time is 7.07 s for both at every point, so this is not speed
traded for anything.

### Why, and what it says about the next experiment

Two mechanisms fit, and they compound.

**The lead-in already solves the alignment.** The guided channel's entry flares
catch a blade arriving up to 16.6 mm per side off centre and walk it in, which is
recorded on this page as the reason they were built. The displacement being
injected is 4 mm. One swallows the other, so neither policy has to work out which
way to correct: the ramp corrects for it. Putting those two numbers next to each
other before launching would have predicted a flat curve, and that is the
methodological miss in this experiment.

**Force in the observation cannot become compliance in a position-controlled
action space.** The arm commands relative Cartesian pose through
damped-least-squares differential IK. A policy that reads a contact force has no
action that yields to it; the one thing it can do with a force reading, given a
per-episode force budget it is conditioned on, is decide how hard to push. It
learned to spend the budget. Spending it is counterproductive here, because
harder pushing into a passive ramp raises contact and trips the abort without
helping the blade slide across.

That is exactly what the two 2026 papers say, and it is the half of them this
experiment deliberately did not adopt. arXiv 2604.19677 selects per axis between
position and force control; arXiv 2602.14174 commands a force *direction* and
lets a fixed magnitude supply compliance. Both make force actionable. This task
made force observable and left the action space stiff, and the measured answer is
that observability alone is worth nothing here and is mildly harmful.

**So the deferred change is the load-bearing one.** Roadmap item 7, an admittance
or impedance action space, moves from "the remaining lever on peak contact force"
to the precondition for force sensing to pay at all.

**Limitations of this comparison, stated plainly.** One training seed per arm.
The three evaluation seeds vary initial conditions only, so the 6-to-10 mm gap is
consistent across evaluation but training repeatability is untested and could
account for part of it. The comparison is otherwise tight: identical
configuration, schedule, seed, reward, and terminations, with one observation
term removed, and a CPU test asserts that.

**Verification before any GPU was spent.** Fourteen simulator checks: the rails
and flares move with the goal to within 0.6 micrometres, the blade starts clear
of the channel, an idle episode develops 4.0 N of contact and no ejection, the
belief is wrong by exactly the displacement and constant to 8.7 micrometres
within an episode, the force threshold samples and varies inside [5, 20] N, and
the tool pose no longer determines the true error. Two bugs were caught this way
that unit tests could not reach: the curriculum term hard-coded three stage
buckets and raised `IndexError` on a single-stage task, and the displacement
event assumed every profile carries lips and flares.

## Demonstration assets

Recorded from the promoted Level-2 checkpoint at full reset distance, 300
control steps each, all episodes successful:

| Clip | View | Environments |
| --- | --- | ---: |
| `artifacts/demo/closeup/` | `grasp`, tool and slot | 1 |
| `artifacts/demo/side/` | `side`, blade entering the rails | 1 |
| `artifacts/demo/array/` | `array`, parallel grid running one policy | 9 |

`artifacts/` is untracked. Publish clips through a GitHub Release. Headless RTX
recording shows visible denoiser speckle; re-record in the Isaac Sim GUI if a
cleaner master is needed.

## Robustness profiles

Cumulative secured-grasp profiles:

| Level | Physics added | State |
| --- | --- | --- |
| L0 | Collision-free insertion | Promoted, seeds 1060/2060/3060 |
| L1 | Wide side rails, larger pose error | Promoted, seeds 1061/2061/3061 |
| L2 | Tight 1.5 mm side clearance, 5–15 kg mass | Promoted, seeds 1062/2062/3062 |
| L3 | Rail friction, 10–120 N breakaway/viscous stiction | Implemented, blocked |
| L4 | Compliant floating mount, wrench pulses | Implemented, blocked behind L3 |

## Known failures and limitations

- L3 reaches valid insertion geometry but cannot consistently settle below
  velocity thresholds under the sampled high stiction. Do not hide this by
  loosening success thresholds or running long PPO. Inspect the force model and
  contact energy first.
- The visible lower shelf collider is disabled in the rigid-grasp task. Tight
  floor contact plus randomized friction caused non-physical lateral ejection.
  Side rails remain physical. Re-enable only after geometry/contact calibration.
- The grasp pose is outside the fingers' reach: the handle is configured
  0.179 m from the flange while the fingers only obstruct on the blade between
  about 0.06 and 0.15 m. This is why the grasp gate fails. Insertion is
  unaffected because the fixed joint welds the blade to the tool frame, but no
  physical grasp can work until the grasp pose is corrected.
- The head-on grapple pin holds 69 N against the 66.4 N requirement with the
  capture/hold split, and 59 N against any single closure command. It is the
  first interface in this project to form a real grip, but no policy has been
  certified on it: the three skills that were trained on it are deleted, and P2
  is where a policy goes back onto it.
- A single-point pin does not constrain yaw once the rails release the blade,
  measured at 0.93 rad of blade rotation in failing extractions. An anti-yaw
  feature is the highest-value change available to the interface specification
  and it has not been made.
- The contact task's finger commands are still inverted. Measured pad separation
  falls monotonically with the command, so `finger_joint` 0 is fully open, and
  the task's "pregrasp 0.80 / closed 0.68" pair opens the fingers by 14 mm. The
  grapple-pin task uses the measured convention; the contact task has not been
  corrected, because doing so changes the physics three promoted certifications
  were produced under.
- The fixed joint is a task abstraction. The real Robotiq pad/handle contact
  task holds 0 N of axial pull because the pads never reach the handle, and must
  not be called learned grasping.
- Primitive blade/rack geometry has no connector, latch, cable, chamfer,
  measured tolerance, or force-displacement curve.
- No wrist force/torque sensing, force limit, damage proxy, real UR10e, HIL rig,
  orbital acceleration data, calibrated camera, or radiation dataset exists.
- Visual noise and lighting ranges are engineering priors. The vision student has
  not been trained from the promoted insertion policy.
- Only one PPO training seed produced each promoted policy. The certification
  seeds are held-out *evaluation* seeds; training repeatability is untested.
- `train.py --smoke`'s scripted axial feasibility probe is now scoped to the
  contact-grasp family it was written for, so the rigid-grasp, force, and vision
  tasks smoke cleanly. Normal training never ran this probe.
- The contact task and the capture-in-slot task both fail `train.py --smoke` for
  pre-existing reasons recorded under *Static validation*: an inverted finger
  command that must not be corrected without re-certifying, and a
  `contact_grasp` flag inconsistent with its parent's disabled handle collider.
- Force sensing did not buy robustness to pose error on this task. Beyond the
  trained 4 mm displacement the force-aware policy is worse than its force-blind
  control by up to 8.2 points, and it uses about twice the peak contact force at
  every displacement. The diagnosis is that the lead-in flares already handle a
  4 mm offset mechanically, and that a position-controlled action space gives a
  policy no way to convert a force reading into compliance. See the section above.
- The pose-belief task's channel is 210 mm long against the certified 600 mm, and
  carries one reset distance instead of three. Both are consequences of moving the
  mouth ahead of the blade's start and are stated in its own section above; its
  numbers are therefore not directly comparable with the promoted Level-0/1/2
  certifications, which used the full-length slot and three distances.
- The authored 64x64 camera resolves a 4 mm slot displacement as **0.13 pixels**,
  so it cannot support the perception stage as configured. `docs/perception_plan.md`
  derives the fix, a narrower field of view rather than more pixels, and requires
  a rendered frame before anything is trained on it.
- No policy has been trained on `Isaac-ZeroG-Blade-Insertion-Vision-v0`. It
  exists so the camera and both Replicator randomizers stay reachable and
  exercised after the swap task was deleted. Its camera pose is the one authored
  for that deleted scene, and whether it frames the slot well enough to regress a
  millimetre-scale pose error is unmeasured.
