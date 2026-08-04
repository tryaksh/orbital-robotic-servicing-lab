# Architecture

## Repository map

| Area | Responsibility |
| --- | --- |
| `assets.py` | UR10e/Robotiq config, blade/rack/caddy collision proxies, and the compliant D6 mount |
| `scene_cfg.py` | Minimal clonable scene and optional 64x64 tiled camera |
| `env_cfg.py` | `ManagerBasedRLEnvCfg`, manager terms, and teacher/vision/play profiles |
| `mdp/actions.py` | Six-axis relative differential IK and one coupled physical gripper command |
| `mdp/commands.py` | Batched eight-phase swap state and phase-dependent goals |
| `mdp/observations.py` | Privileged state, deployable proprioception, RGB corruption, and critic state |
| `mdp/rewards.py` | Reach, alignment, extraction/insertion, milestone, success, and safety terms |
| `mdp/randomization.py` | Reset state, rail stiction, mount wobble, Replicator materials, and orbital sun |
| `mdp/terminations.py` | Exact held success, workspace, mount envelope, timeout, and non-finite guards |
| `mdp/curricula.py` | Four-stage success-rate promotion at 70% over 100 completed episodes |
| `agents/` | RL-Games PPO configurations and the multimodal vision actor |
| `scripts/` | Setup, cleanup, smoke, benchmark, train, play, demonstration, and BC entry points |

## Data flow

```text
ManagerBasedRLEnvCfg
  -> InteractiveSceneCfg (UR10e, rack, two blades, caddies, optional camera)
  -> CommandManager (per-environment swap phase and active target)
  -> ActionManager (6D relative differential IK + 1D binary Robotiq command)
  -> EventManager (physics and visual randomization)
  -> ObservationManager
       teacher actor: state
       student actor: proprioception + RGB
       critic: privileged state
  -> Reward/Termination/Curriculum managers
  -> RL-Games wrapper and PPO
```

The policy runs at 30 Hz while physics runs at 120 Hz. The vision profile renders
at 15 Hz and reuses each frame for two policy steps. The teacher profile does not
instantiate a camera.

The physical action has seven values. Values 0-5 command a relative Cartesian
pose through damped-least-squares differential IK. Value 6 opens or closes the
2F-85 and is expanded to six explicit gripper-joint targets with coupling signs
`[+,+,-,+,-,-]`. There is no simulated magic attachment; blade retention
depends on PhysX collision and friction.

## Full-swap state machine

1. Approach the failed-blade handle.
2. Close the gripper and verify a stable geometric grasp.
3. Extract along the slot axis.
4. Move to the service caddy, release, and retreat.
5. Approach and grasp the spare blade.
6. Align the spare blade with the slot using pose/keypoint error.
7. Insert while maintaining rail alignment.
8. Open the gripper and retreat to the safe pose.

The phase command is a batched tensor. Transitions use geometry and velocity
thresholds rather than CPU callbacks, allowing all environments to advance
independently on the GPU.

Training starts with the replacement pre-aligned for insertion. A rolling
success curriculum promotes to spare acquisition, failed-blade stow plus
replacement, and finally the complete swap after each stage sustains at least
70% success over 100 completed episodes. Terminal success requires the failed
blade in the service caddy, the replacement seated within translational and
rotational tolerances, an open gripper, a retreated end effector, low blade
velocity, and a continuous 0.5 s hold.

## Physics and rendering profiles

| Profile | Physics cloning | Camera | Default count | Primary use |
| --- | --- | --- | ---: | --- |
| Teacher | replicated + Fabric | none | 1024 default/tested | privileged-state PPO |
| Vision | non-replicated | tiled 64x64 RGB at 15 Hz | 128 training default; 256 environment-tested | vision PPO and held-out visual randomization |
| Play | non-replicated | tiled RGB | 8 | synchronized teacher/RGB collection and video |

All profiles use zero world gravity, 120 Hz physics, 30 Hz policy control, no
ground plane, and no contact-report sensor. Contact solving remains active for
the gripper. The robot base is attached to a static anchor by a D6 joint with
limits of +/-15 mm and +/-2 degrees, translational stiffness/damping 12,000/220,
and rotational stiffness/damping 600/50. Bounded wrench pulses excite the
compliant mount.

## Scaling profiles

The teacher clones shared physics and consumes only vector observations. The
vision profile disables physics replication because per-environment Replicator
rack materials require individually addressable prims; its environment count is
therefore lower. A descending first-fit benchmark selects the largest stable
count under a 10.5 GiB total VRAM ceiling. On the development RTX 5070 Ti Laptop
GPU, sustained environment stepping passed at 1024 teacher environments
(7,378.90 environment-steps/s, 1,037 MiB observed total GPU memory) and 256
vision environments (1,597.65 environment-steps/s, 2,266 MiB). Full PPO uses
additional network, rollout, and optimizer memory, so the vision training
default remains 128.

## Sim2Real boundary

The environment exposes ground-truth blade state to the teacher and asymmetric
critic. The deployed student actor receives only robot proprioception, task
phase, previous action, and RGB. Teacher actions are recorded with matching
camera frames for behavioral-cloning initialization before vision PPO.

```text
privileged teacher PPO
        |
        | deterministic action labels
        v
RGB + proprio + phase + action HDF5
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
  state, exact success, and timeout are separate termination terms.
- The benchmark runs candidate counts in isolated processes so a high-count
  failure does not poison the next lower candidate.
- Cleanup validates the retained simulator and CUDA before deleting exact old
  paths.
- Datasets, checkpoints, videos, raw logs, machine locks, and benchmark JSON are
  excluded from Git history; release metadata must identify their source commit.
