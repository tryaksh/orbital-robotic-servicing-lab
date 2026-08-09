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

Ruff passes. 50/50 non-Sim tests pass. Isaac smoke passed for the corrected
evaluator and for a two-iteration checkpoint-resume training run through the
`TerminalMetricsManagerBasedRLEnv` entry point. GPU physics smoke previously
passed for Levels 0, 1, and 2. Sustained environment-only benchmarks passed at
1024 state and 256 vision environments; full PPO memory differs.

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

## Learned grasping is blocked by a 165 mm tool-frame error, not by friction

`docs/status.md` previously recorded only that the real Robotiq pad/handle
contact task "failed its axial pull gate". `scripts/grasp_diagnostics.py`
replaces that pass/fail with a measurement, and the measurement says the gate
was never close to passing for a reason nobody had looked for.

The script holds the arm still, closes the fingers, then applies a constant
axial pull to the blade. Environment `i` takes one point of a 4 closure x 32
force grid from 0 to 120 N, so 128 environments sweep the surface in one run.

| Quantity | Measured |
| --- | ---: |
| Tool frame used by the IK, to actual finger-pad midpoint | 165.6 mm |
| Finger-pad midpoint to handle centre | 164.6 mm |
| Distance at which the pads could touch a 75 mm handle | 37.5 mm |
| Environments with pads in reach of the handle | 0 of 128 |
| Peak gripper drive torque, against a 10 N·m limit | 0.39 N·m |
| Axial force held before slip | 0 N |
| Axial force the promoted insertion policy's contact reaction demands | 66.4 N |

Report: `evidence/grasp_axial_pull_gate.json`.

**There is no grasp to characterise.** The finger pads settle 165 mm away from
the handle and never touch it, the drive joints reach their commanded closure
without developing torque, and the blade is a free-floating body in zero
gravity. Its motion under load confirms this arithmetically rather than by
inspection: 120 N on a 10 kg blade for 1.5 s predicts 13.5 m of free travel, and
the measurement is 12.5 m. The earlier "failed pull gate" was not a weak grip.

The root cause is a frame error, and it is present in the promoted task too. The
tool offset is authored 190 mm along the wrist for the rigid-grasp task and
179 mm for the contact task, while the real pad midpoint sits about 13 mm from
`wrist_3_link`. Both configurations therefore drive a tool frame 165.6 mm away
from the fingers that are supposed to be holding the blade. Two consequences:

- **This is the PhysX startup warning, not a cosmetic issue.** The simulator
  already reports that the fixed joint "connects disjoint transforms and will
  snap them together". The disjoint distance is 165.6 mm. That warning was
  filed as a frame-consistency cleanup; it is the same defect.
- **`tool_to_handle_error_m` cannot detect a grasp problem.** In the rigid-grasp
  task it measures exactly 0.0000 m, because the fixed joint welds the blade to
  the tool frame that the metric compares against. It is a self-consistent
  tautology, not an audit of a grip. `docs/claim_vs_evidence.md` already said
  the near-zero value is "a property of that joint, not of a grip"; the number
  behind that sentence is now measured.

A secondary observation needs confirming before it is acted on: finger-pad body
separation grows monotonically with the commanded value, from 0 mm at 0.00 rad
through 42.7 mm at 0.45 to 58.9 mm at 0.60. The task calls 0.45 "pregrasp" and
0.60 "closed", so the command it treats as closed separates the pad bodies
further than the one it treats as open. That was measured between pad *body
origins*, not certified pad faces, so it is a lead rather than a result.

**What this does and does not invalidate.** It does not touch the promoted
Level-0/1/2 insertion results. Those depend on blade-versus-slot geometry, rail
contact, and blade dynamics, all of which are real, and the fixed joint is
documented throughout as an abstraction rather than a grasp. What it invalidates
is the assumption that the contact-grasp task was a nearly working grasp needing
tuning. It is not: the gripper and the handle are not in the same place, so
Phase-3 grasping starts with a geometry fix, not with PPO.

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
- The tool frame the differential IK drives sits 165.6 mm from the physical
  Robotiq finger pads in both the rigid-grasp and contact tasks. This is the
  cause of the PhysX "disjoint body transforms" startup warning and of the
  failed grasp gate; the blade hangs 165 mm below the fingers that appear to
  hold it. Insertion is unaffected because the fixed joint welds the blade to
  the tool frame, but no physical grasp can work until this is corrected.
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
- `train.py --smoke` runs a scripted axial feasibility probe tuned for the
  contact task. On the rigid-grasp task it exhausts its 300-step budget with
  23.5 mm residual axial error. Verified identical before and after the
  evaluator change, so it is a probe defect, not a task defect: the learned
  policy inserts in 35 control steps at stage 0. Normal training is unaffected
  because it does not run this probe.
