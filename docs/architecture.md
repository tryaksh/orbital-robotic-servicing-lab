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
| `mdp/rewards.py` | Mount deflection and one-shot termination penalties, shared by every task |
| `mdp/randomization.py` | Reset state, rail stiction, mount wobble, Replicator materials, and orbital sun |
| `mdp/terminations.py` | The compliant mount's D6 envelope |
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

## What was deleted, and why it is not coming back

An eight-phase full-swap task (approach, grasp, extract, stow, acquire, align,
insert, retreat) and three head-on grapple-pin skills (grasp, extract, insert)
were removed on 2026-08-10. Four of the five servicing stages carried no physics
content — in simulation they are a state machine — and each of the three skills
failed for a reason established work already solves: a reset that solved the
task, a 495 mm credit-assignment horizon, and a success predicate written in the
wrong frame. `docs/status.md` keeps the measurements that found those faults.

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
