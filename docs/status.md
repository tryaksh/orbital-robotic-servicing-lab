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
- PhysX logs a startup warning that the fixed joint connects disjoint transforms
  and will snap them together. The settled gap is effectively zero, but the
  authored/reset joint frames should eventually be made consistent.
- The fixed joint is a task abstraction. The real Robotiq pad/handle contact
  task failed its axial pull gate and must not be called learned grasping.
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
