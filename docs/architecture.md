# Architecture

## Repository map

| Area | Responsibility |
| --- | --- |
| `assets.py` | UR10e/Robotiq config, blade/rack/caddy collision proxies, and the compliant D6 mount |
| `scene_cfg.py` | Minimal clonable scene and optional 64x64 tiled camera |
| `env_cfg.py` | Arm joint names and the one shared PhysX/render configuration |
| `mdp/actions.py` | Six-axis relative differential IK and one coupled physical gripper command |
| `mdp/grasp_frames.py` | Top-down 2F-85 grasp attitude conventions, including finger symmetry |
| `mdp/observations.py` | Privileged state, deployable proprioception, RGB corruption, and critic state |
| `mdp/insertion.py` | Insertion goal, rewards, terminations, contact-force sensing, and the stage curriculum |
| `mdp/grapple.py` | Head-on capture frames and the two-stage capture/hold gripper action |
| `mdp/uncertainty.py` | Slot displacement, the believed pose error, force-threshold conditioning, and the sampling curriculum |
| `mdp/rewards.py` | Mount deflection and one-shot termination penalties, shared by every task |
| `mdp/randomization.py` | Reset state, rail stiction, mount wobble, Replicator materials, and orbital sun |
| `mdp/terminations.py` | The compliant mount's D6 envelope |
| `uncertain_insertion_env_cfg.py` | Insertion under a displaced slot, and its force-blind ablation |
| `grapple_pin_env_cfg.py` | The head-on capture scene and the three skills it makes possible |
| `workflow_demo_env_cfg.py` | The three skills in one continuous episode; bookkeeping only, never train on it |
| `vision_insertion_env_cfg.py` | The camera task the visual randomizers were repointed at (untrained) |
| `agents/` | RL-Games PPO configurations and the multimodal vision actor |
| `scripts/` | Setup, cleanup, smoke, benchmark, train, play, demonstration, and BC entry points |

## Data flow

```text
ManagerBasedRLEnvCfg
  -> InteractiveSceneCfg (UR10e, rack, one blade, guide rails, optional camera)
  -> CommandManager (the fixed rack goal, kept as a command so it can be
                     randomized later without changing the observation shape)
  -> ActionManager (6D relative differential IK; the capture scene adds the
                    two-stage Robotiq command)
  -> EventManager (reset poses, physics randomization, and, on the vision task,
                   Replicator rack materials and orbital sun)
  -> ObservationManager
       state actor:   proprioception + blade goal error + contact wrench
       vision actor:  proprioception + contact wrench + RGB
       critic:        the above plus ground-truth pose and randomized dynamics
  -> Reward/Termination/Curriculum managers
  -> RL-Games wrapper and PPO
```

The policy runs at 30 Hz while physics runs at 120 Hz. The vision profile renders
at 15 Hz and reuses each frame for two policy steps. State profiles do not
instantiate a camera.

Every task's arm action has six values commanding a relative Cartesian pose
through damped-least-squares differential IK. The head-on capture scene adds a
seventh, the 2F-85 open/capture/hold command, expanded to six explicit
gripper-joint targets with coupling signs `[+,+,-,+,-,-]`. Every insertion task
uses a PhysX fixed joint to represent a blade that was already secured. **This
is not learned grasping.**

## The chained workflow, and how it is judged

`scripts/run_workflow_demo.py` drives one episode with three checkpoints,
switching between them on **measured conditions** rather than on a timer. It runs
one workflow for a video and many in parallel, headless, for evidence: with
`--episodes` it writes one row per completed workflow in the format
`scripts/aggregate_evaluation.py` already pools, so a chain is gated exactly the
way a single skill is.

Three properties make the chain's number comparable with the skills' numbers
rather than merely adjacent to them.

*Each phase gets the clock its own skill was certified on.* `PHASE_BUDGET_S`
reads `episode_length_s` off the three task configurations, so a phase that
overruns fails the workflow. Before this the chain granted 45 s while the insert
skill was certified on 12 s, and "it completes in the chain" and "it scores 6.5%
alone" were both true statements about different tasks.

*Success is re-checked after the workflow's predicate fires.* The driver holds
still for 0.70 s and asks again. That is strictly harder than the skills' own
criteria, and it is what separates a module that is seated from one that was
briefly in tolerance.

*The holding closure latches per environment.* `TwoStageRobotiqAction.hold_latch`
is a per-environment flag the driver sets at the hand-off. Nothing in a training
task sets it, so every trained policy still sees the behaviour its certification
was produced under.

### Measuring what the chain hands each skill

`--handoff_trace` writes a second `.npz` carrying two tables, and it is off by
default and free when off:

- **`handoff`**, one row per phase transition per environment: grip error and
  attitude, finger angle and drive torque, module pose and velocity, tool
  position, and the six arm joint positions.
- **`settle`**, one row per environment per step of the settling window, so a
  module that was settled when its predicate fired and is not settled when it is
  judged shows up as a curve rather than as two numbers.

`scripts/analyse_handoff.py` pools either table and, given the nominal pose a
receiving skill resets around, reports the per-joint deviation between what the
chain delivers and what that skill trains on.

This exists because a per-skill certification cannot see the chain and the chain
cannot see inside a phase, so the hand-off between them was the one thing nothing
measured — and "a skill must be trained across the states its predecessor
produces" is the defect this project has hit four times. The first run of it
found insert scoring 95.57% on its own reset and about 80% on the states the
chain hands it, a gap invisible in every certification either side of it.

Turning those traces into a *reset distribution* is a closed line of work, and
its machinery was deleted on 2026-08-16: four reconstructions scored 0.00%,
26.32% and 47.17% against the ~80% the same unchanged policy achieves in the real
chain, because a hand-off is a trajectory and a gripper-controller state rather
than a pose. `Isaac-ZeroG-Blade-GrapplePin-InsertChain-v0` reproduces it properly
by running the frozen capture inside the environment, and measures 93.06%. The
measurements are in `docs/status.md`; the pose bank, its generator and
`reset_from_handoff_bank` are gone.

## What was deleted, and why it is not coming back

An eight-phase full-swap task (approach, grasp, extract, stow, acquire, align,
insert, retreat) was removed on 2026-08-10 and is not coming back: four of its
five servicing stages carried no physics content — in simulation they are a state
machine — and `tests/test_configuration_contract.py` fails if it returns.

The three head-on grapple-pin skills (grasp, extract, insert) were removed with
it and **restored on 2026-08-11**. Each had failed for a reason established work
already solves — a reset that solved the task, a 495 mm credit-assignment
horizon, and a success predicate written in the wrong frame — but each of those
causes had been identified and corrected in the same session and then never
retested. `docs/status.md` keeps the measurements that found the faults and the
measurements that followed the corrections.

The camera, `RackMaterialRandomizer`, `OrbitalLightingRandomizer`,
`camera_rgb_with_radiation_noise`, and the `VisionActor` were reachable only from
the swap task. They were repointed at the insertion scene rather than deleted
with it; see `vision_insertion_env_cfg.py`.

## Physics and rendering profiles

| Profile | Physics cloning | Camera | Default count | Primary use |
| --- | --- | --- | ---: | --- |
| State | replicated + Fabric | none | 512 training; 1024 environment-tested | state PPO |
| Force | non-replicated USD clones | none | 512 | contact sensing; Fabric clones carry no contact-report API |
| Vision | non-replicated | tiled 64x64 RGB at 15 Hz | 128 training default; 256 environment-tested | untrained P3 scaffold and visual randomization |
| Vision play | non-replicated | tiled RGB | 8 | synchronized state-teacher/RGB collection and video |

All profiles use zero world gravity, 120 Hz physics, 30 Hz policy control, no
ground plane, and no contact-report sensor. Contact solving remains active for
the gripper. The robot base is attached to a static anchor by a D6 joint with
limits of +/-15 mm and +/-2 degrees, translational stiffness/damping 12,000/220,
and rotational stiffness/damping 600/50. Bounded wrench pulses excite the
compliant mount.

## Scaling profiles

The state profile clones shared physics and consumes only vector observations.
The force and vision profiles disable physics replication, the first because
Fabric-cloned prims do not carry the contact-report API and the second because
per-environment Replicator rack materials require individually addressable
prims; their environment counts are therefore lower. A descending first-fit benchmark selects the largest stable
count under a 10.5 GiB total VRAM ceiling. On the development RTX 5070 Ti Laptop
GPU, sustained environment stepping passed at 1024 state environments
(7,378.90 environment-steps/s, 1,037 MiB observed total GPU memory) and 256
camera environments (1,597.65 environment-steps/s, 2,266 MiB). Full PPO uses
additional network, rollout, and optimizer memory, so the vision training
default remains 128.

## Sim2Real boundary

The environment exposes ground-truth blade state to the state actor and to the
asymmetric critic. The vision actor receives only robot proprioception, the
contact wrench, the previous action, and RGB. A state teacher's actions are
recorded with matching camera frames *and with the true blade-to-goal error*,
which is both the behavioural-cloning label and the supervision a P3
pose-regression head needs.

```text
privileged state PPO
        |
        | deterministic action labels
        v
RGB + proprio + true pose + action HDF5
        |
        | behavioral cloning
        v
vision actor initialization
        |
        | asymmetric vision PPO
        v
held-out simulation evaluation -> HIL/real-rig evaluation (future)
```

The diagnostic `blade_pose` vision group is intentionally excluded from the
RL-Games actor mapping. It exists for evaluation and dataset inspection; only
the critic receives privileged object state during vision PPO.

## Failure containment

- Workspace escape, mount overtravel (translation or rotation), non-finite
  state, excessive contact force, exact success, and timeout are separate
  termination terms.
- The benchmark runs candidate counts in isolated processes so a high-count
  failure does not poison the next lower candidate.
- Cleanup validates the retained simulator and CUDA before deleting exact old
  paths.
- Datasets, checkpoints, videos, raw logs, machine locks, and benchmark JSON are
  excluded from Git history; release metadata must identify their source commit.
