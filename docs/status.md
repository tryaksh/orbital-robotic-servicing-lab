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

## Two probes that closed the force line of work

Both follow directly from the flat ablation above, and both are evaluation-only:
no retraining, on the two Stage-1 checkpoints.

### The lead-in was doing the insertion, not helping with the offset

Removing the guided channel's entry flares and changing nothing else:

| Slot displacement | Force-aware | Force-blind |
| ---: | ---: | ---: |
| 0 mm | 0.00% | 0.00% |
| 4 mm | 0.00% | 0.00% |
| 8 mm | 0.17% | 0.00% |

**Both policies fail every episode, including at zero displacement where there is
no uncertainty at all.** The flares were not assisting with the offset; they were
performing the insertion. Neither policy ever learned to align a blade into a
0.75 mm-per-side channel, because with a 16.6 mm-per-side catch in front of it
neither ever had to. That is a complete explanation of the flat ablation: if the
mechanics do all of the alignment, sensing cannot contribute to it.

Both policies trained with the flares present, so this measures *dependence on*
the lead-in, not the impossibility of learning without one. A policy trained
without flares might learn to align; none has been.

The force-aware arm keeps its signature even in collapse, at 21.9 N mean contact
and 35 to 51 force aborts per point against 14.7 N and 1 to 4 for the control.

This echoes the interface specification's central finding from the opposite
direction. There, a parallel-jaw grip could not hold the module and the fix was
geometry on the module. Here, a policy cannot align the module and the fix is
geometry on the rack. **Design-for-serviceability is doing work that control
cannot substitute for**, twice, measured both times.

Raw rows: `artifacts/noleadin/`. No pooled report is published for this probe,
because the runs predate the reporting change that records whether the lead-in is
present, and a 0% result filed as a certification would misread as a failed
policy rather than a removed mechanism.

### A commanded force budget is ignored

FORGE conditions a policy on a maximum allowable contact force so it can be asked
at deployment for a gentler or a firmer insertion. Stage 1 built that mechanism
and sampled the budget uniformly per episode, which averages any modulation away.
Pinning it exposes whether the policy tracks the command. Four budgets, three
held-out seeds each, at the trained displacement:

| Commanded budget | Aware peak mean | Aware peak p95 | Blind peak mean | Blind peak p95 |
| ---: | ---: | ---: | ---: | ---: |
| 5 N | 10.03 N | 31.81 N | 7.05 N | 14.72 N |
| 10 N | 8.90 N | 27.15 N | 7.17 N | 14.67 N |
| 15 N | 9.08 N | 30.09 N | 6.71 N | 12.84 N |
| 20 N | 9.19 N | 30.34 N | 6.82 N | 13.39 N |

Reports: `evidence/uncertain_force_threshold_aware.json`,
`evidence/uncertain_force_threshold_blind.json`.

**Neither policy tracks the command.** The force-aware arm sits at about 9 N mean
whatever it is told, and the only visible response to the tightest budget is the
wrong way: 10.03 N mean, 7 force aborts, and its lowest success rate. The
force-blind arm is flat by construction, since it is handed a limit it has no
sense to respect.

The arithmetic says why the penalty never binds. It is linear, weighted -3.0, and
normalised by 20 N, so a 30 N peak against a 5 N budget costs 3.75 in the step it
occurs. The success term is 35 per step-second, which is 1,050 for a single
terminal step, and the progress term is weighted 12. A transient strike is
therefore worth paying for, every time.

### The finding these three converge on

| Intervention | Effect on peak contact force |
| --- | --- |
| Quadratic penalty, mild and strict profiles | None (2.6% at the mean) |
| Contact force in the observation | None (about 1%), though sustained rubbing fell 59% |
| Per-episode commanded budget, conditioned | None (about 1% across a fourfold range) |

Three independent mechanisms, three nulls. **Peak contact force is not regulable
under position-based differential IK on this workcell, and the binding constraint
is the action space rather than the reward or the sensing.** Roadmap item 7, an
admittance or impedance action space, is now the only untried lever, and both
2026 papers this project read make force *actionable* rather than merely
observable for exactly this reason. Until that exists, further force-reward
tuning here is known waste.

Force feedback's one measured win is unaffected and still stands: contact impulse
fell 59% at the mean and 89% at the median when force entered the observation.
Sustained rubbing is regulable. The first strike is not.

## Certifying the policies the demonstration actually loads

`evidence/` carried certifications for grasp v2, extract v2 and insert v3 while
`scripts/run_workflow_demo.py` loaded grasp **v3**, extract **v4** and insert
**v5**. Every figure this page quoted about the demonstration therefore described
a superseded policy. The three checkpoints the demonstration loads have now been
evaluated on three held-out seeds each and the reports are named after the
version they describe, so the two cannot drift apart silently again.

| Skill | Checkpoint SHA-256 | Report | Episodes | Success | Gate |
| --- | --- | --- | ---: | ---: | --- |
| Capture v3 | `AF579F5A…` | `evidence/grapple_grasp_v3_certification.json` | 9,020 | 95.55% | **fails** |
| Extract v4 | `C0AB5F42…` | `evidence/grapple_extract_v4_certification.json` | 9,078 | **0.00%** | **fails** |
| Insert v5 | `A1567059…` | `evidence/grapple_insert_v5_certification.json` | 3,074 | 6.96% | **fails** |

**All three fail their gate, and one of them fails completely.** The stale files
said 95.5%, 91.5% and 30.7%. Two of those three numbers are not merely out of
date, they point the wrong way.

### Capture v3 is indistinguishable from capture v2

| Stage | v2 | v3 |
| --- | ---: | ---: |
| Near | 3,010 / 3,010 | 3,012 / 3,012 |
| Medium | 93.70% | 94.04% |
| Full | 92.78% | 92.61% |
| Pooled | 95.50% | 95.55% |

The reward change between them raised the blade-disturbance penalty fivefold and
widened its free band past the seating feed. It traded 22 capture failures for 17
extra timeouts and moved the pooled figure by 0.05 points. Recorded because a
change that does nothing is worth knowing about, and because it is the honest
reading of two numbers that look different in a commit message.

### Extract v4 scores zero, and the reason is not the pull

0 successes in 9,078 episodes, at every curriculum stage, against 91.5% for the
superseded v2. The two numbers are not comparable — v2 was certified before the
reset noise was widened 40× for chaining, so it describes an easier task — but
the v4 result stands on its own and the terminal metrics say exactly what fails:

| Terminal quantity | Median | Requirement |
| --- | ---: | ---: |
| Grip **position** error | 12.2 mm | ≤ 20 mm |
| Grip **attitude** error | **0.299 rad** | **≤ 0.20 rad** |
| Distance travelled toward clear | about 458 mm | 495 mm |
| Blade orientation error | 0.142 rad | — |

**The grip holds and the module travels; the tool rotates in the grip.** Position
error sits comfortably inside tolerance for the whole 15 s, and the pull covers
more than nine tenths of the required distance. `capture_established` then fails
on its orientation term, so `extraction_success` can never fire, and 2,842 of the
episodes go on to trip the 0.35 rad grip-attitude failure limit outright.

### Insert v5 is not primarily slow, it is rotating

The handover recorded insert's problem as a clock: 473 of 479 failures were
timeouts with the median module 11.29 mm from a 12 mm tolerance, so "it is slow,
not unreliable". Splitting the 2,860 failures by which success condition each one
satisfies at its terminal step shows that diagnosis is incomplete, and that the
part it misses is the larger one:

| Success condition | Failures satisfying it |
| --- | ---: |
| Lateral alignment ≤ 2.5 mm | 100.00% |
| Blade orientation ≤ 0.0524 rad | 100.00% |
| Grip position ≤ 20 mm | 94.20% |
| Axial depth ≤ 12 mm | 52.06% |
| **Grip orientation ≤ 0.20 rad** | **6.99%** |
| All five at once | 2.59% |

**Ninety-three per cent of failures are out of tolerance on grip orientation at
the moment they end.** The 214 successes make the point from the other side:
their grip attitude has a median of 0.1902 rad, a 95th percentile of 0.1934, and
a maximum of 0.1945, against a limit of 0.20. Every single successful insertion
is pressed up against that limit, and the blade itself is straight — its
orientation error against the goal is 0.0043 rad, satisfied in 100% of episodes —
so the 0.27 rad is the wrist rotating relative to a module the rails hold still.

**The reading taken from that table was that a longer episode could not fix a
condition which was not converging. That was wrong, and lengthening the episode
proved it wrong.** See the next section: at 20 s the same skill scores 95.57%.
Grip orientation *was* converging; it was converging slowly, and the 12 s episode
was cutting it off. A distribution of terminal states says which condition is
unsatisfied when the clock stops. It does not say whether that condition was
still moving, and reading the first as the second is the mistake this paragraph
records.

The pin's free yaw is real and is measured elsewhere on this page. It is not what
was capping the insert skill.

## Insert v6: the clock was the whole of it

`episode_length_s` 12 → 20 s, fine-tuned 800 PPO epochs from the v5 checkpoint at
512 environments, seed 70, one change and nothing else. Certified on the same
three held-out seeds:

| Insert | Episode | Episodes | Success | Gate |
| --- | ---: | ---: | ---: | --- |
| v5 | 12 s | 3,074 | 6.96% | fails |
| **v6** | **20 s** | **3,000** | **95.57%** | **passes** |

Report: `evidence/grapple_insert_v6_certification.json`, checkpoint SHA-256
`7E9A0C33…`. Terminations: 2,867 insertion successes, 90 timeouts, 43 lost grips.
**This is the first head-on grapple-pin skill to pass its promotion gate.**

The reason 12 s was not enough is now plain: successful insertions take a median
of **13.43 s** and a 95th percentile of 16.83 s, so the previous episode ended
before the median success had happened. The old table showing every success
landing at 11.7 s of a 12 s budget was not describing a fast skill with a little
headroom, it was describing a distribution with its right tail cut off.

The residual 133 failures still look the way v5's did — 81% of them outside the
grip-orientation tolerance, 68% short on axial depth — but there are 22 times
fewer of them. Whether the remainder is the pin's yaw or simply more of the same
slow convergence is the question the anti-yaw yoke is built to answer.

Two things this does **not** change. The insert task is single-stage and starts
from the certified staging pose, so this is not a claim about insertion from an
arbitrary approach. And the reconciled chain budget follows this field
automatically, so the workflow now grants its insert phase 20 s because the skill
is certified on 20 s, not because 20 s was convenient.

## Two chained servicing workflows, and the three defects chaining exposed

Three skills certified separately at 92.5%, 91.5% and 100% composed into **zero**
working workflows. None of the three reasons was visible in any individual
certification, and all three are the same shape: a skill that works alone assumes
something its neighbour does not provide.

### The defects

**A grip that relaxes exactly when the part shifts.** `TwoStageRobotiqAction` and
`hold_two_stage_grip` both re-commanded the gentler *capture* closure whenever
`capture_established` went false. That predicate fails as soon as grip error
passes 20 mm, which rail contact does routinely, so the fingers opened about
21 mm and released the module mid-task. Measured effect on the insert skill:

| Insert version | Grip losses | Success |
| --- | ---: | ---: |
| v3, before the fix | 6,023 / 9,014 | 30.7% |
| v4, after the fix | **9 / 512** | 100% near, 7% full distance |

The holding closure now latches for the episode and releases on reset.

**A success criterion looser than the next skill's precondition.** The grasp task
counts a capture from 20 mm of grip error and the driver handed over at the first
qualifying instant, which measured 22.7 mm after seating. The extract policy has
never begun an episode from worse than 12.4 mm. The hand-off now waits for the
capture to close to 10 mm, landing at 12.97 mm after the seating feed.

**A reset too narrow to absorb a predecessor's output.** Extraction trained with
0.0005 rad of joint noise, three hundredths of a degree, so it had seen exactly
one arm configuration. Chained, it reversed into the rack on all eight seeds
tried *even with the hand-off grip error matching its own reset to within a
millimetre*, because the arm's joint configuration was outside anything it knew.
Retrained across 0.010 to 0.020 rad it commits to the pull. Insert was retrained
the same way.

A fourth, smaller one: the grasp policy declares capture at finger angle 0.085
while the pin sits home at 0.223, so the chain adds the 1 s seating pause the
extract task gets free from its own settling window.

### The chain, certified: 0.00% and 15.10%

Nothing in `evidence/` covered a chained run and both workflow videos were n = 1.
`run_workflow_demo.py --episodes` now runs the same driver headless across many
environments and writes the rows `scripts/aggregate_evaluation.py` already pools,
so the chain is gated exactly the way a skill is. Three held-out seeds
(4070/5070/6070), 64 environments, 576 workflows each:

| Workflow | Success | Wilson 95% | Report |
| --- | ---: | ---: | --- |
| Removal | **0 / 576, 0.00%** | [0.00%, 0.66%] | `evidence/workflow_remove_certification.json` |
| Installation, insert v5 | 87 / 576, 15.10% | [12.41%, 18.26%] | `evidence/workflow_install_certification.json` |
| **Installation, insert v6** | **497 / 576, 86.28%** | **[83.23%, 88.85%]** | `evidence/workflow_install_v6insert_certification.json` |

Neither passes the 95% gate, but installation is now within nine points of it,
and the whole of that improvement came from giving the insert skill the clock its
own motion needs. Its remaining failures are 47 insert overruns and **29 capture
overruns** — the capture skill's own 95.55% is now a visible contributor to the
chain rather than being masked by a larger failure downstream. Removal is
untouched by any of this: it is blocked on extraction, which is blocked on yaw.

Both videos below are therefore **demonstrations of capability, not evidence of
reliability**, and every document now says so.

Three things had to be settled before those numbers meant anything.

**Each phase gets the clock its own skill was certified on.** `PHASE_BUDGET_S`
reads `episode_length_s` off the three task configurations, so a phase that
overruns fails the workflow. Before this the chain granted 45 s while the insert
skill was certified on 12 s, and "it completes in the chain" and "it scores 6.5%
alone" were both true statements about different tasks. Reconciled, the overruns
are the headline failure mode:

| Workflow | Overran capture's 6 s | Overran extract's 15 s | Overran insert's 12 s |
| --- | ---: | ---: | ---: |
| Removal | 7 | **569** | — |
| Installation | 40 | — | **448** |

**Success is re-checked after the predicate fires.** The driver holds still for
0.70 s and asks again, which is stricter than the skills' own criteria and is
what separates a module that is seated from one that was briefly in tolerance.
One installation in 576 fired the predicate and then failed the re-check.

**`completed` and `conditions_still_held_after_settling` are now both reported,
and the gap between them is the yaw.** Of the 87 successful installations, only
16 still satisfy every condition including the grip after settling. The module
stays exactly where it was put — its axial, lateral, and orientation errors all
hold — and the *pin* relaxes in the pads. Pooled over all 576 installations the
grip attitude is inside its 0.20 rad tolerance at the end in 24.5% of episodes;
over the removals, in **3.8%**.

That is the same measurement the per-skill certifications above produce, from a
third direction. Removal's median run reaches a module centre of 0.2737 m against
the 0.225 m that clears the rack, so it is 49 mm short after 15 s of pulling
while the grip *position* sits at 12.7 mm — holding fine, rotating badly.

### Removal: capture and extract, both learned

One continuous episode, `scripts/run_workflow_demo.py --workflow remove`. **This
is one run of a workflow certified at 0.00%.** It is kept because it shows the
motion is achievable, and because its own timeline is what shows why the pooled
number is zero: the pull needs 16.7 s and the extract skill is certified on 15 s.

| Phase | Kind | Time | Module centre x | Grip |
| --- | --- | ---: | ---: | ---: |
| Capture | learned | 0.00 - 0.97 s | 0.7195 | 19.4 to 7.4 mm |
| Seat | scripted | 0.97 - 1.97 s | 0.7270 | 12.97 mm at 10 N-m |
| Extract | learned | 1.97 - 18.63 s | **0.2243** | 15.5 mm at 10 N-m |

495 mm of travel on a fully installed module, held by pad-against-pin contact
throughout. No fixed joint and no software fixture in this scene.

### Installation: capture and insert, both learned

`--workflow install`, starting with the module presented at the rack mouth:

| Phase | Kind | Time | Module centre x | Grip |
| --- | --- | ---: | ---: | ---: |
| Capture | learned | 0.00 - 2.73 s | 0.5829 | 68.0 to 1.2 mm |
| Seat | scripted | 2.73 - 3.73 s | 0.5933 | 12.8 mm |
| Insert | learned | 3.73 - 16.57 s | 0.7414 | 11.7 mm |

Seated at 8.63 mm axial and 0.61 mm lateral error, inside the 12 mm and 2.5 mm
tolerances, with blade orientation error 0.0043 rad.

**One honest caveat on this one.** Completion means the task's own success
predicate fired, which it did. Re-checking every condition after a further 0.7 s
of settling shows `grasp_orientation` at 0.229 rad against its 0.20 rad limit:
the module stays seated and the *pin* relaxes slightly in the pads afterwards.
The reports record both, as `completed` and
`conditions_still_held_after_settling`.

### What is not chained, and why it is the interface rather than the controller

The round trip — remove, fly back, re-install — does not work, and the cause is
the limitation `docs/service_interface_spec.md` already records: a single-point
pin does not constrain yaw once the rails release the module. Flying the module
back degrades the grip from 15 mm to 35 mm, and slowing the replay fourfold makes
it **worse**, so this is rotation under sustained load rather than an
acceleration artefact. An anti-yaw feature, a keyway or flats the pads bear
against laterally, is the fix, and it is a second-generation interface result.

A separate kinematic finding sits behind the scripted transit: at full extraction
the wrist sits behind the robot's own base, and driving straight back from there
takes the damped-least-squares IK through a near-singularity, swinging the
shoulder 74 degrees and driving the elbow into its limit. The transit therefore
retraces the path the extraction flew, advancing waypoints on the clock;
advancing them on proximity stalls, because the last waypoint is sampled up to a
stride before the hand-off.

### Insert's remaining limitation, and a diagnosis that was half wrong

The reading recorded here first was that the insert skill is slow rather than
unreliable: 473 of 479 failures were timeouts, the median module was 11.29 mm
from a 12 mm tolerance, and successful insertions took 11.77 s against a 12 s
episode, so a longer episode with retraining looked like the whole fix.

Half of that survives certification and half does not. On 3,074 held-out
episodes, 52.06% of failures do miss axial depth, so the clock is real. But
**93.01% of failures are outside the grip-orientation tolerance at the step they
end on**, and every one of the 214 successes is pressed against that limit, with
a maximum of 0.1945 rad against 0.20. A longer episode cannot fix a condition
that is not converging. Both changes are needed and they are separate
experiments: lengthen the episode, and constrain yaw on the interface. The
per-skill table above has the full split.

## The anti-yaw yoke: designed from measurement, axially validated, yaw untested

Three separate certifications now point at the same property of the interface —
extraction at 0%, 93% of insertion failures out of grip-orientation tolerance,
and the chained removal inside it in 3.8% of episodes — so the second-generation
feature the specification has been asking for was built.

**The measured envelope decides where it can go.** Re-reading
`evidence/gripper_collision_envelope.json` over the whole 0 to 0.8203 rad closure
range rather than at one command gives two numbers that had not been extracted
before:

| Quantity | Measured |
| --- | ---: |
| Deepest any body that is *not* an inner finger reaches from the flange | 0.1245 m |
| Deepest an inner finger reaches | 0.1621 m |
| Widest half-extent of a non-finger body on the third axis (the inner knuckle) | 17.5 mm |
| Inner finger half-width | 13.5 mm |

So there is a **37.6 mm band immediately behind the collar in which the only
gripper body present is a finger**, and inside it a wall narrower than 17.5 mm
cannot foul anything. The pin is already 30 mm across against a 27 mm finger, so
the feature is the wedge's own side faces raised into two walls the fingers run
between, and the pin gains no width at all.

| Feature | Value | Why |
| --- | ---: | --- |
| Wall inner half-gap | 15.0 mm | Flush with the pin's flanks; 1.5 mm per side against a 13.5 mm finger |
| Yoke length from the collar | 34 mm | Puts the mouth 0.128 m from the flange, 3.5 mm clear of the knuckle band |
| Parallel section | 24 mm | The part that constrains yaw |
| Lead-in flare | 10 mm at 20 degrees, to an 18.6 mm half-gap | 5.14 mm of catch per side |
| Wall height | ±45 mm | The collar's own, so the yoke never exceeds the depth stop's envelope |
| Predicted free yaw | 0.125 rad | Geometry: 2c/L. Against 0.93 rad measured with no yoke |

The lead-in is not decoration. A 1.5 mm slot the capture has to hit blind would
trade a yaw problem for a capture problem, and this project has already measured
what a missing lead-in costs: remove the rack's and two fully trained insertion
policies both score 0%. `tests/test_grapple_geometry.py` defends every bound
above against the measured gripper numbers and runs without a simulator.

**The axial pull gate still passes, which was the thing that could have killed
it.** Adding geometry that fixes rotation is worthless if it costs the hold:

| Interface | Best axial force held at 0.48 rad | Angular slip p95 under axial pull |
| --- | ---: | ---: |
| Plain pin | 69 N | 0.1481 rad |
| Yoked pin | **67 N** | **0.1312 rad** |
| Required | 66.36 N | — |

Report: `evidence/grapple_pin_axial_pull_gate_yoked.json`, 3 closures by 121
forces at 1 N resolution, the same grid the 69 N result was measured on. The
yoke costs 2 N of holding capacity, still clears the requirement, and moves
angular slip under axial load in the right direction.

**The yaw gate does not work, and the reason is worth more than the gate was.**
A probe was added to `scripts/grasp_diagnostics.py` to load the interface about
the closing axis and measure how far the payload rotates. On the capture scene it
measures **0.079 rad whatever load is applied, identically with and without the
yoke**, and 200 N of lateral force at the module's centre moves it 1.2 mm.

That is not a null result about the yoke. It is the rails: the capture scene
holds the module in its slot, the slot constrains lateral motion, and a module
that cannot translate sideways cannot yaw. The pull gate's own notes said as much
already — the blade "only starts to lever once the pull has dragged it clear of
the rails that were constraining it" — and that sentence turns out to be a
constraint on what a static probe can ever measure here. **Yaw is not a property
of the seated interface; it is a property of the interface once the rack lets
go.** Measuring it needs a moving extraction, not a held pose.

Reports: `evidence/grapple_pin_yaw_probe_railed_plain.json` and
`_yoked.json`, named for what they are. They carry `gate.applies: false`.

**So the yoke was built, dimensioned from measurement, defended by tests, and
proven not to cost the axial hold.** Whether it fixed yaw was then measured, by
training all three skills against it. It did not, and the section below is that
result.

## The yoke, trained against and turned back off

Decided by measurement on 2026-08-15. All three skills were fine-tuned onto the
yoked pin at 512 environments, seed 70, robustness level 0, and certified on the
same three held-out seeds as their plain-pin predecessors. Resuming was
legitimate: the yoke changes contact geometry and changes neither the
observation nor the action dimension.

| Skill | Plain pin | Yoked pin | Report |
| --- | ---: | ---: | --- |
| Capture | 95.55% | **88.81%** | `evidence/grapple_grasp_v4_certification.json` |
| Extract | 0.00% | **0.13%** | `evidence/grapple_extract_v5_certification.json` |
| Insert | 95.57% | **28.70%** | one seed, `artifacts/certify/insert_v7_s0_seed1070_play.json` |

**It costs 6.7 points of capture and 67 of insertion to buy 0.13 of
extraction.** `GrapplePinBladeCfg.anti_yaw_yoke` and the task-level flag are
both back to off, and the walls stay implemented, dimensioned, and defended by
`tests/test_grapple_geometry.py` and `tests/test_yoke_asset.py`, because the
measurement is the result and the feature may matter on a stiffer gripper.

Two honest caveats on those numbers. Capture's fine-tune was still climbing
steeply when its budget ran out — reward 21.3 to 30.5 over the final 100 epochs
— so 88.81% conflates the yoke's cost with an unfinished retrain, and insertion
was worse still at reward 1.4 against the plain pin's 24.9. A matched plain-pin
control at the same budget was not run. What is not in doubt is the direction.

### The yoke was aimed at the wrong axis, and nobody had measured which

`grapple_grip_attitude_axes` decomposes the capture attitude error into the
gripper's own axes, recorded per episode behind `play.py --grip_axis_metrics`.
Until 2026-08-15 only the *magnitude* was recorded, and a magnitude cannot say
which axis a rotation is about. Measured on extraction with the plain pin:

| Component | Terminal p50 |
| --- | ---: |
| About the **closing** axis — the only axis the yoke's walls oppose | 0.198 rad |
| About the **transverse** axis | 0.199 rad |
| About the approach axis | 0.070 rad |

**The rotation is split roughly evenly between two axes and the yoke addresses
one of them.** That is the whole explanation for recovering 12%. Three sessions
of this project called this failure "yaw" and designed a feature against that
name without ever measuring the axis. The lesson is the same one the railed yaw
probe taught in a different costume: check that the thing you are measuring is
the thing you think it is.

## A modelled latch, and why a torque is not form closure

Flight servicing hardware does not hold a module against extraction by friction
on a passive feature. The SSRMS latching end effector snares a grapple fixture
and then rigidizes it; Dextre's ORU Tool Changeout Mechanism grips a
standardised fixture — H-fixture, micro-fixture, or micro-conical — and carries
a powered socket drive. The load path after capture is form closure through a
latch. So a latch was modelled: `mdp.GrappleLatch` engages the first step a
capture qualifies, applies a rated restoring torque and no force, so the axial
hold is still the wedge's measured 69 N and still has to be earned.

Swept against the **unchanged** extract v4 policy, so nothing in the comparison
can be a training artefact:

| Latch rating | Extract success | Transverse rotation p50 | Module travel p50 |
| ---: | ---: | ---: | ---: |
| none (plain pin) | 0.00% | — | 458 mm |
| 10 N·m | 0.00% | 0.293 rad | 24 mm |
| 20 N·m | 0.00% | 0.296 rad | 22 mm |
| 40 N·m | 0.00% | 0.298 rad | 26 mm |
| 80 N·m | 0.00% | 0.299 rad | 29 mm |

Report directory: `artifacts/latch/`. **An eightfold change in rating moves the
rotation it targets by 0.006 rad and destroys the extraction**, because a
restoring torque applied to a module the rails still hold jams it in the rails:
travel collapses from 458 mm to about 25 mm. `latch_enabled` is off.

One methodological note worth keeping. The first latch had stiffness and no
damping, and it was worse than useless — the module pinned itself against the
0.35 rad failure limit and travelled 84 mm. In zero gravity nothing dissipates
the energy a stiffness injects, so a spring alone is a catapult. The damping
term, sized from the payload's measured inertia, stayed.

## What the evidence actually points at: the wrist, not the module

Three measurements, none of them new, say the same thing once they are put next
to each other:

- The module's own orientation error against its goal is **0.0043 rad** — it
  stays straight — while the grip attitude reads **0.30 rad**. A straight module
  and a wrong grip attitude is the *wrist* rotated relative to it. No change to
  the pin can fix a wrist.
- Extraction stalls at a hard **478 mm of the required 495** whatever clock it
  is given, and a distance ceiling that does not move with time is kinematic.
- Servoing the task's own 6-DoF IK to points along the extraction path leaves
  **0.10 to 0.26 rad** of orientation residual, at poses the trained policy
  traverses successfully as well as at the end point.

At full extraction the tool has to sit 0.336 m horizontally from the robot base
while 0.570 m above it, pointing back over the base, which puts the wrist centre
about 0.20 m in front of the shoulder. That is a folded configuration and it is
where this project's own handover said the risk was: *"inside the UR10e's reach
but folded, and it has not been checked kinematically."*

### It has now been checked, and the workcell is not the problem

Servoing the task's own 6-DoF IK onto the extraction end pose with 2,000 steps
instead of 400, and sweeping the robot base along world x with
`calibrate_grasp_pose.py --robot_base_x`:

| Robot base x | Position shortfall | Orientation residual |
| ---: | ---: | ---: |
| **−0.45 m, as built** | **3.6 mm** | **0.0114 rad** |
| −0.65 m | 13.1 mm | 0.1114 rad |
| −0.85 m | 12.7 mm | 0.1153 rad |

**The arm reaches the extraction end pose and holds the head-on attitude there
to 0.0114 rad, seventeen times inside the 0.20 rad tolerance**, and moving the
base back makes it slightly worse rather than better. The 0.10–0.26 rad
residuals recorded above were an under-converged 400-step servo, not a
kinematic wall, and reading them as one would have been a third interface
redesign aimed at a fourth wrong cause.

So the pose is reachable, the interface holds 69 N, and the module stays
straight. **The grip attitude failure is neither hardware nor kinematics. It is
the objective.** See the next section.

### The reward was paying the policy to give the attitude away

`grip_retention_penalty` charges attitude as `0.25 * ((error - 0.08) / 0.15)^2`.
At the 0.20 rad success limit that is about **0.16 per step**, against an
extraction progress term weighted **12**. The policy was not failing to control
attitude; it was correctly trading an almost-free quantity for a well-paid one,
and then dying on the 0.35 rad limit for a one-off −15.

That is the same class of mistake as the 12 s insert episode: a number that
looked like a physical limitation and was a specification. The extract task now
charges attitude about 3.6 per step at the success limit — free below 0.04 rad,
normalised over 0.06, weighted 1.0 — through parameters on the shared function,
so **insertion keeps the defaults its certification was produced under**.

### Extract v7: the objective was worth 18 points

600 PPO epochs fine-tuned from v6, one change and nothing else. Certified on the
same three held-out seeds:

| Extract | What changed | Episodes | Success | Timeouts | Lost grips |
| --- | --- | ---: | ---: | ---: | ---: |
| v4 | — | 9,078 | 0.00% | 6,236 | 2,842 |
| v6 | 15 s → 25 s | 9,001 | 10.09% | 122 | 7,971 |
| **v7** | **attitude weighted** | **9,002** | **28.48%** | **51** | 6,387 |

Report: `evidence/grapple_extract_v7_certification.json`, checkpoint SHA-256
`58785D8A…`, Wilson 95% [27.56%, 29.42%].

**Extraction went from nothing to nearly a third, and no part of that was a
hardware change.** Two objective defects, each found by reading a distribution
rather than by redesigning a part: an episode shorter than the median success,
and an attitude term two orders of magnitude below the progress term it competed
with. Between them they had absorbed four sessions and two interface features.

One curiosity worth carrying: the stage ordering inverted. Full distance now
scores **32.07%** against 26.88% near and 26.50% medium, where every previous
version fell monotonically with distance. A policy that does better from further
out is not one whose limit is the pull.

**What is still unfixed is unchanged in kind.** 6,387 of the 6,438 failures end
at the 0.350 rad grip-attitude limit, and timeouts are now negligible at 51.
Attitude is still the whole of the remaining failure; it is simply no longer
being given away for free. The next lever is the action space — this repository
has now measured twice that a position-controlled action space cannot convert a
sensed quantity into compliance, once on contact force and once on attitude —
and **not** a third passive interface feature.

### The better skill is the worse chain component, and that is a real result

Both workflows were re-certified against extract v7 on the same three held-out
seeds and 576 workflows each:

| Workflow | With extract v6 | With extract v7 |
| --- | ---: | ---: |
| Removal | **14.06%** | **3.30%** |
| Installation | 86.28% | 86.28% (does not use extract) |

**Extraction alone improved from 10.09% to 28.48% while the removal chain it
sits in fell from 14.06% to 3.30%**, and 557 of the 576 chained failures are
timeouts against 51 in 9,002 when the skill runs alone. Installation is
identical because it never calls extract, which is the control that says the
difference is extract and not drift somewhere else.

This is the same lesson the chain taught the first time it was certified, in the
opposite direction: **a skill certification is not evidence about the chain.**
The plausible mechanism, and it is a hypothesis rather than a measurement, is
that a large attitude penalty makes standing still cheaper than pulling when the
episode begins outside the state the policy is comfortable in — and a chained
extract begins wherever the capture policy's servoing left the arm, never on the
nominal reset. That is the same class of defect as the 0.0005 rad reset noise
that made the first chained extract reverse into the rack.

Until that is diagnosed, **extract v6 is what the removal chain should load and
extract v7 is the better skill**, and those two sentences are both true.

## Extract's clock: 15 s was never enough

Certified on a 15 s episode, extract v4's median cycle time is **15.000 s** —
every episode ran the clock out — while the module reached 458 mm of the
required 495. That is insert v5's situation exactly, and insert v5 → v6 was
fixed by lengthening the episode *and* fine-tuning against the new horizon,
which took it from 6.96% to 95.57%.

The clock alone is not the fix, and that was measured before changing anything.
Replaying extract v4 unchanged at longer episodes:

| Episode | Timeouts | Lost grips | Travel p50 | Grip attitude p50 |
| ---: | ---: | ---: | ---: | ---: |
| 15 s | 449 | 63 | 458 mm | 0.299 rad |
| 25 s | 3 | **510** | 478 mm | 0.350 rad |
| 40 s | 0 | **512** | 478 mm | 0.350 rad |

More time converts timeouts into lost grips at a fixed 478 mm ceiling, because a
policy asked to work past its trained horizon degrades rather than continues.
So the task's `episode_length_s` moved to 25 s **and** the skill was fine-tuned
against it, which is what insert v6 did. `PHASE_BUDGET_S` reads that field, so
the chain's extract budget followed automatically.

### Extract v6: the first extraction this project has ever completed

600 PPO epochs fine-tuned from the v4 checkpoint at 512 environments, seed 70,
one change and nothing else. Certified on held-out seeds 1070/2070/3070:

| Extract | Episode | Episodes | Success | Timeouts | Lost grips |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 | 15 s | 9,078 | **0.00%** | 6,236 | 2,842 |
| **v6** | **25 s** | **9,001** | **10.09%** | **122** | 7,971 |

Report: `evidence/grapple_extract_v6_certification.json`, checkpoint SHA-256
`8B310405…`. Wilson 95% [9.48%, 10.73%]. Consistent across seeds — 10.43%,
8.70%, 11.13% — and falling with reset distance: 12.47% near, 10.40% medium,
7.40% full.

**This is the first time a module has been pulled clear of the rack by a
certified policy.** It is not a working skill and it does not approach the 95%
gate. What it is, is the end of a zero that had survived four sessions and two
interface redesigns, and it converts a dead measurement into a live one.

The clock was genuinely binding: timeouts fell from 6,236 to 122, and successful
extractions take a median of **18.23 s**, which the old 15 s episode made
impossible by construction. The 458 mm the module used to reach was never the
policy's ceiling, it was the buzzer.

**What remains is now unambiguous.** 7,971 of the 8,093 failures end on
`extraction_failed` with the grip attitude at 0.350 rad, which is that
predicate's own limit, while grip *position* holds at 12.5 mm. Every failure is
the same failure, and it is the rotation the section above shows is split across
two axes and which the module itself does not share — the module stays straight
to 0.0043 rad. Fixing it is a workcell or a controller question, and the two
interface features built against it are both measured as harmful.

## Aligning capture worked, moved the bottleneck, and cost the chain 8 points

The diagnosis was right, the fix did exactly what it was designed to do, and the
chain got worse. All three parts of that sentence are measured.

Capture certified at 96.10% alone while causing 68% of the installation chain's
failures, because its success tolerance was 20 mm of grip error and the workflow
refuses to hand over until 10 mm. `capture_success_mask` now defaults to
`WORKFLOW_HANDOVER_GRIP_M` — read, not restated — and the episode went 6 s to
10 s because reaching 10 mm takes longer than reaching 20. 800 PPO epochs
fine-tuned from the v5 checkpoint at 512 environments, seed 70, one change.

| | capture v5 | capture v6, aligned | Evidence |
| --- | ---: | ---: | --- |
| Capture alone | 96.10% | 95.87% | `grapple_grasp_v6_certification.json` |
| **Install chain, state** | **84.38%** | **76.74%** | `workflow_install_aligned_certification.json` |
| Install chain, oracle control | 80.38% | 79.17% | `vision_workflow_oracle_aligned_certification.json` |
| **Install chain, camera** | **80.38%** | **69.97%** | `vision_workflow_camera_aligned_certification.json` |
| Install chain, **blind** control | 43.58% | **59.90%** | `vision_workflow_blind_aligned_certification.json` |

**The intervention hit its target precisely.** Capture-phase overruns fell from
39 of 576 chained installations to **1**. Nothing else in the chain was touched.

**And the failures moved wholesale onto the next skill.** Insert-phase overruns
rose from 43 to **129**, and they are now 129 of the chain's 134 failures:

| Chain failures, of 576 | capture v5 | capture v6 |
| --- | ---: | ---: |
| Capture overran its budget | 39 | **1** |
| Insert overran its budget | 43 | **129** |
| Fired, then failed the 0.70 s re-check | 8 | 4 |
| Total | 90 | 134 |

This is the most repeated defect in this project, for the third time: **a skill
trained across states its predecessor no longer produces.** Capture v6 hands over
a 4.31 mm grip in 1.33 s; insert v6 was trained against the hand-off v5 produced
and certified at 95.57% on it. Making the hand-off *better* moved it out of
distribution, which is the same failure as the 0.0005 rad reset noise that made
the first chained extract reverse into the rack, and the same failure the
capture/insert tolerance mismatch was.

**The blind arm moving is the second result and it is the more uncomfortable
one.** The null control — no image, the module assumed to be exactly where the
rack nominally presents it — improved from 43.58% to 59.90%. The margin that
makes the vision claim mean anything therefore collapsed from **36.8 points to
10.1**, and it collapsed from both ends at once. A capture that converges faster
and tighter needs less information about where the module actually is, so
sharpening capture partially substituted for perception. Nothing was retrained to
flatter either arm; both run identical checkpoints through an identical
observation term.

### The three arms finally separate, and that is the one gain here

Under capture v5 the camera arm and the oracle arm scored **80.38% each, to the
digit**. That reads as a triumph and is closer to a warning: two estimators that
cannot be told apart mean the task was not asking either of them for much. With
capture v6 the three arms separate into a monotonic ladder for the first time:

| Arm, capture v6 | Result | Wilson 95% |
| --- | ---: | --- |
| Oracle — the simulator's own answer | 79.17% | [75.66, 82.28] |
| **Camera — 64x64 RGB through the pose head** | **69.97%** | [66.10, 73.57] |
| Blind — the module assumed to be at its nominal pose | 59.90% | [55.84, 63.82] |

**This is the first non-zero cost of perception this project has measured.** The
oracle and camera intervals do not overlap, so the 9.2-point gap between them is
the estimator and cannot be anything else; the 10.1 points below that are what
having any estimate at all is worth. The mechanism is not mysterious: a capture
that servos to 4 mm needs to know where the module is more precisely than one
that stops at 10, so the pose head's 1.75 mm mean error stopped being free.

The honest framing is that this is a better-structured measurement of a worse
chain. Nothing here recovers the 8 points the chain lost. What it does is retire
a number — camera equals oracle — that could never have supported the claim it
was being read as supporting.

### Measuring the hand-off, and finding the diagnosis was aimed one skill early

The reading above — that capture v6 moved the hand-off out of insert's
distribution — was checked before anything was retrained against it, with a new
instrument. `run_workflow_demo.py --handoff_trace` records the state every phase
actually hands over in, and the state through the settling window;
`scripts/analyse_handoff.py` pools it. Nothing in this repository measured that
before, which is why a rule it has broken three times could keep being broken.

**The hand-off barely moved.** 192 chained installations per arm, same seed:

| State handed to insert | capture v5 p50 | capture v6 p50 |
| --- | ---: | ---: |
| Grip error | 12.64 mm | 12.50 mm |
| Grip attitude | 0.0823 rad | 0.0925 rad |
| Finger angle | 0.2316 rad | 0.2306 rad |
| **Worst-axis arm-joint deviation from nominal** | **0.138 rad** | **0.157 rad** |

The "4.31 mm hand-off" the capture-alignment work was aimed at is capture's
*standalone* terminal state. In the chain the hand-over fires at the 10 mm gate
and a one-second seat follows, so both versions deliver about 12.5 mm. Capture v6
changed the state insert receives by 12 to 20%, not by a factor of three.

**What the instrument found instead is larger and older.** Insert's reset drew
uniform ±0.020 rad of joint noise around one nominal pose. The chain hands it a
pose 0.157 rad away on its worst axis at the median, overwhelmingly `wrist_1`,
with a p95 of 0.284 and a maximum of 0.383. Even the *fifth percentile* hand-off
is three times the widest value that reset could draw on any single joint. Both
captures do this. The certified insert skill has never been trained on the states
its predecessor produces, and the direct measurement of what that costs is:

| Insert v6, unchanged policy | Success |
| --- | ---: |
| On its own certified reset | **95.57%** |
| On 550 measured capture hand-offs | **31.25%** |

That single pair is the chain gap. It is not something capture v6 introduced, it
was not visible in any per-skill certification, and it explains 95.57%-alone
against 84.38%-chained without appealing to anything else.

### A hand-off is a manifold, not a box, and that is measurable

The obvious fix — widen the per-joint reset noise to cover 0.28 rad on `wrist_1`
— was tried first and is **degenerate**. Insert v6 under it scored **0.00%**, and
all 534 episodes ended the same way: grip lost at reset, terminal grip error
65.6 mm.

The reason is the correlation the box throws away. The chain's hand-off has a
large joint deviation *and* a 12.5 mm grip error together, because the capture
policy **servoed** there — the wrist rolled while the tool stayed on the pin.
Independent per-joint noise produces a large joint deviation *and* a large grip
error, so the fingers close on nothing. The states are both "0.28 rad from
nominal" and nothing else about them is alike.

So the reset samples measured poses instead of inventing them.
`src/zero_g_blade_swap/tasks/blade_swap/handoff_poses.py` carries 550 hand-off
arm poses collected on **training-side seeds 70, 170 and 270** — deliberately not
the 4070/5070/6070 the chain is certified on, because a reset distribution drawn
from the evaluation seeds is not a held-out evaluation any more. Regenerate it
with `scripts/build_handoff_pose_bank.py`.

Insert v6 unchanged scores 31.25% on that reset with failures split across
175 lost grips, 177 timeouts and 160 successes, which is a harder task with a
gradient in it rather than a collapsed one. That is the control the retrained
policy is measured against; comparing it with the 95.57% would be comparing two
different tasks.

### Retraining insert on the pose bank made it worse, and the numbers say why

The fix was applied and it failed. 1,000 PPO epochs fine-tuned from insert v6 at
512 environments, seed 70, the reset sampling the 550 measured hand-off poses and
nothing else changed. Resuming was legitimate: a reset distribution is an event
change and alters neither the observation nor the action dimension.

| | control, insert v6 | insert v7align |
| --- | ---: | ---: |
| **On the hand-off reset** | **26.32%** | **24.33%** |
| Install chain, state | 76.74% | **69.10%** |
| Install chain, oracle | 79.17% | **70.66%** |
| Install chain, camera | 69.97% | **55.21%** |
| Install chain, blind | 59.90% | **54.17%** |

Reports: `evidence/grapple_insert_v6onhandoff_certification.json`,
`grapple_insert_v7align_certification.json`,
`workflow_install_insertalign_certification.json`, and the three
`vision_workflow_*_insertalign_certification.json`.

**Training on a distribution made the policy worse on that distribution**, which
is the shape of a mis-specified task rather than an insufficient budget. The
reward trajectory agrees: it fell from +24.91 on the old reset to −1.93 on the
first recorded epoch of the new one and then oscillated between −0.74 and −4.66
for 900 epochs without trend. This is not the capture-fine-tune situation, where
reward was still climbing steeply when the budget ran out. It had converged.

**And the control is what identifies the fault.** Insert v6 succeeds in roughly
80% of the chain's insert phases — 442 of 576 workflows complete, and 184 of 192
reach insert. The *same policy* on the pose bank scores 26.32%. Those are
supposed to be the same states. They are not, and the bank is the thing that is
wrong:

> The reset samples an arm pose from the bank and then places the module at its
> **fixed** nominal pose. In the chain the module sits where the capture left it,
> and the arm sits where the capture servoed it *to reach that module*. Sampling
> one without the other reproduces neither.

So the pose bank fixed one broken correlation and introduced another. The
per-joint noise box broke arm-against-module by randomising the arm; the pose
bank breaks it by randomising the arm against a module that did not move with it.
Both produce grip geometries the chain never generates, which is why one scored
0.00% and the other 26.32% where the real hand-off gets 80%.

**The fix is named and it is small.** `HANDOFF_TRACE_FIELDS` already records the
module's position; it needs its orientation as well, and the reset needs to write
the arm pose and the module pose **as a matched pair drawn from the same recorded
hand-off**, rather than sampling one and defaulting the other. That is a change
to `run_workflow_demo.py`, `build_handoff_pose_bank.py` and one event term, and
it should be measured the same way: insert v6 unchanged on the paired bank should
score near the 80% it achieves in the chain *before* anything is retrained. **If
it does not, the bank is still not the hand-off and no amount of PPO will fix
it.** That check costs five minutes and would have caught this one.

**What is promoted: nothing.** Capture v5 with insert v6 remains the best measured
installation chain at 84.38%, and it is what `certify_workflow.sh` and
`certify_vision_workflow.sh` still default to. Neither capture v6align nor insert
v7align is promoted, and both stay in the record because the measurements are the
result.

**What this does not license.** Reverting to capture v5 would buy the chain back
8 points and restore the vision margin, and it would be the wrong move: v5 leaves
39 capture overruns and the criterion mismatch that caused them, and a chain
propped up by a predecessor's sloppiness is not a chain. The stated fix is to
retrain insert against the hand-off capture v6 actually produces, which is the
same fix that has now worked twice.

## The extract number this project has been quoting is stale

`docs/status.md`, `docs/roadmap.md`, and `CLAUDE.md` all record extraction at
**68.36%**. That figure was produced under a success criterion the code no longer
contains, and under the current one it is not close to true.

`grapple_extract_v8_certification.json` was written at 13:54 on 2026-08-15
(commit `fde6947`). The extraction velocity limits were derived and tightened at
14:58 the same day (commit `3851fa0`), from a chosen 0.30 rad/s and 0.10 m/s to a
derived 0.1429 rad/s and 0.0143 m/s. The certification predates the change by an
hour.

Re-reading v8's own recorded terminal metrics against the limits now in the code:

| Extract v8, 9,005 held-out episodes | p50 | p95 | Within the current limit |
| --- | ---: | ---: | ---: |
| Terminal linear velocity, successes | 0.0710 m/s | 0.0842 m/s | **0 of 6,156** |
| Terminal angular velocity, successes | 0.1517 rad/s | 0.2025 rad/s | 36.3% |

**Not one of the 6,156 episodes counted as a successful extraction would satisfy
the linear velocity limit the predicate now enforces**, and the median exceeds it
fivefold.

Every extract and removal figure on this page shares the defect. Checked against
the 14:58 commit, `grapple_extract_v8_certification.json` (13:21),
`grapple_extract_v9_certification.json` (14:46), and every removal chain run
including the 14.06% (`remove_clock`, `remove_scalefix`, `remove_v9`) were
written before the limits were derived. **The only extraction ever measured
against the criterion the chain enforces is v10, and it scored 0.00%.**

Two consequences, and the second matters more than the first.

- **68.36% must not be quoted again without a re-run.** The honest current number
  for extract under the criterion the chain enforces is unmeasured and the
  evidence points at close to zero. Re-certifying v8 unchanged against the
  present limits is the measurement, and it costs half an hour.
- **The "over-correction" recorded in the handover was not one.** Extract v10 was
  retrained against the derived limits and scored 0.00%, which was read as the
  limits being too strict for a skill that had been working. Against this
  arithmetic, v10 was the first honest measurement of a skill that had never
  satisfied the chain's criterion, and the 68.36% it was compared against was the
  artefact. The derived limits stay — they are what the settling window can
  actually confirm — and the gap they expose is larger than the handover records.

The experiment this points at is unchanged and now better motivated: **nothing in
the extract reward pays the policy to arrive settled.** Velocity enters only
through a sparse terminal predicate, so there is no gradient toward it at any
point in the pull.

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
- **Extraction's 68.36% is stale and must not be quoted, and so is the removal
  chain's 14.06%.** Both were certified before the extraction velocity limits
  were derived and tightened on 2026-08-15, and none of v8's 6,156 counted
  successes satisfies the linear limit now in the code. The only extraction
  measured against the current criterion is v10, at 0.00%. See the section
  above; the honest current number is unmeasured.
- Aligning capture to the chain's 10 mm hand-off removed 38 of 39 capture-phase
  overruns and cost the installation chain 8 points, because insert v6 was
  trained against the hand-off capture v5 produced. The camera arm fell to
  69.97% and the blind control rose to 59.90%, so the vision margin fell from
  36.8 points to 10.1. Insert has not yet been retrained against the new
  hand-off.
- The rotation that fails the grip-attitude tolerance is **not** yaw about the
  closing axis alone, and calling it that for three sessions cost two interface
  designs. Decomposed on 2026-08-15 it is 0.198 rad about the closing axis,
  0.199 about the transverse axis, and 0.070 about the approach axis. Two
  features were built against the closing axis and both are now off: the
  anti-yaw yoke, which bought extraction 0.13 points and cost insertion 67, and
  a modelled latch, which jams the module in the rails and collapses extraction
  travel from 458 mm to 25 mm at every rating from 10 to 160 N·m.
- The leading remaining suspect is the **workcell layout**, not the interface.
  The module stays straight to 0.0043 rad while the grip attitude reads 0.30, so
  the wrist is what rotates; extraction stalls at a fixed 478 mm of 495 whatever
  clock it is given; and the extraction end pose folds the arm to 0.336 m
  horizontally from its own base at 0.570 m height. This has not been proven —
  the IK calibrator under-converges along the whole path and is not yet a clean
  reachability oracle — and proving or refuting it is the next experiment.
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
