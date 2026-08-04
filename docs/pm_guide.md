# PM Guide: What This Project Does and How to Explore It

## The one-minute explanation

We are building a virtual test and training program for a robot that services a
computer rack in space. Isaac Sim is the physics-and-rendering world. Isaac Lab
adds the machinery for running hundreds of copies of that world at once,
defining what the robot sees and does, scoring behavior, and training a policy.

The robot is not yet skilled. The delivered milestone is the validated training
factory: the world, task, randomization, safety checks, observation contracts,
reward system, PPO integration, and repeatable performance tests all run. Long
training and physical validation are the next milestones.

## See it live

Open a normal PowerShell window and run:

```powershell
Set-Location D:\6axis-space-robotics
C:\isaac-sim\python.bat scripts\play.py `
  --task Isaac-ZeroG-BladeSwap-Play-v0 `
  --policy teacher `
  --num_envs 1
```

Do not add `--headless`: that is what makes the Isaac Sim window appear. The
script automatically selects the newest local teacher checkpoint. Close the
Isaac Sim window to stop. This two-epoch smoke policy is deliberately untrained,
so expect motion, not a successful blade swap.

To watch the camera-based policy instead:

```powershell
C:\isaac-sim\python.bat scripts\play.py `
  --task Isaac-ZeroG-BladeSwap-Vision-v0 `
  --policy vision `
  --num_envs 1
```

If no checkpoint exists after a fresh clone, create a quick integration
checkpoint first, then launch the viewer:

```powershell
C:\isaac-sim\python.bat scripts\train.py `
  --task Isaac-ZeroG-BladeSwap-Teacher-v0 `
  --num_envs 8 --smoke --headless --run_name first_teacher
```

In the Isaac Sim Stage panel, expand `/World/envs/env_0`. Select `Robot`,
`Rack`, `Blade`, or `SpareBlade`; press `F` to frame the selected object. Use
the standard orbit, pan, and zoom viewport controls. Start with one environment
because it is much easier to understand visually than a grid of clones.

## What happens during one control step

1. PhysX advances the zero-gravity world four times at 120 Hz.
2. Sensors produce joint state and, for vision, a 64x64 RGB frame.
3. The observation manager packages only the information allowed for that
   policy.
4. The policy outputs seven numbers: six small end-effector changes and one
   open/close gripper command.
5. Reward terms score progress, and termination terms stop unsafe or completed
   episodes.
6. During training, PPO uses many repeated episodes to make useful actions more
   likely.

This repeats at 30 decisions per second. The eight task phases are approach,
grasp, extract, stow, acquire spare, align, insert, then release and retreat.

## The important concepts without jargon

| Concept | Plain-English meaning | Why it matters |
| --- | --- | --- |
| Environment | One virtual robot, rack, and two blades | We can run many independent practice attempts |
| Policy | The learned decision-maker | Converts observations into robot commands |
| Reward | The score used during practice | Teaches intermediate progress before full success is common |
| Episode | One attempt, ending in success, failure, or 45 seconds | Gives training a repeatable unit of experience |
| Curriculum | Start with insertion, then unlock harder phases | Avoids asking a new policy to solve the entire swap immediately |
| Domain randomization | Deliberately vary mass, friction, wobble, light, materials, and noise | Prevents memorizing one perfect simulator setup |
| Teacher | A policy allowed to see exact simulator state | Learns efficiently and creates demonstrations |
| Vision student | A deployable policy that sees RGB plus robot state | Avoids relying on object poses unavailable on real hardware |
| Privileged critic | Extra training-only information used to grade the student | Helps learning without leaking simulator truth at deployment |
| Sim2Real | Transfer from simulation to a physical system | Requires real or hardware-in-the-loop evidence; randomization alone is not proof |

## How the code is organized

Read these files in this order:

1. `env_cfg.py` is the table of contents. It combines actions, observations,
   rewards, events, termination rules, curriculum, and the three profiles.
2. `scene_cfg.py` says which objects and camera exist in every environment.
3. `assets.py` builds the robot, compliant mount, rack, blades, rails, and
   caddies.
4. `mdp/commands.py` owns the eight-phase state machine and current target.
5. `mdp/observations.py` defines what teacher, student, and critic can see.
6. `mdp/rewards.py` and `mdp/terminations.py` define good progress, success,
   and unsafe outcomes.
7. `mdp/randomization.py` creates the physics and orbital-visual variation.
8. `agents/` defines the teacher MLP, vision CNN, and PPO settings.
9. `scripts/train.py`, `play.py`, and `benchmark.py` are the operator entry
   points.

The detailed engineering map is in [architecture.md](architecture.md), while
[sim2real_matrix.md](sim2real_matrix.md) lists every modeled transfer gap.

## Safe experiments for a non-engineer

- Change `--num_envs 1` to `4` in playback and observe the tiled worlds.
- Add `--seed 7` to playback; random masses, friction, poses, and disturbances
  change reproducibly.
- Run `scripts\benchmark.py --profile all --quick` after closing other GPU-heavy
  apps to compare throughput.
- Change one visual range at a time in `VisualRandomizationCfg`, then replay the
  vision task. Keep a Git commit before each experiment.
- Compare teacher and vision playback. The teacher gets exact object state; the
  student must infer the scene from pixels.

Avoid changing success tolerances, collision geometry, and reward weights
together. When several assumptions move at once, a result is difficult to
interpret.

## What is proven and what is not

Proven on the development laptop:

- Isaac Sim 5.1 and Isaac Lab 2.3.2 run on the RTX 5070 Ti Laptop GPU.
- The zero-gravity physics, compliant mount, two randomized blade masses,
  friction/stiction, seven actions, observations, rewards, and safety terms run
  without non-finite state in smoke testing.
- Orbital lighting, black background, rack appearance variation, and camera
  noise produce valid 64x64 RGB.
- Sustained stepping passed at 1024 state environments and 256 camera
  environments under the 10.5 GiB project budget.
- Both RL-Games networks save and reload a two-epoch checkpoint.

Not yet proven:

- A policy can complete the full swap reliably.
- The learned vision policy transfers to a physical arm or flight-like rack.
- The randomization ranges match measured orbital hardware distributions.
- The primitive rack captures connector forces, cable interference, or
  millimetre-scale manufacturing details.
- The system detects and safely recovers from all failure modes.

## Highest-value roadmap

1. **Demonstrate learning, not just infrastructure.** Train three teacher seeds,
   publish success curves by curriculum stage, and show one uninterrupted
   eight-phase video.
2. **Turn the demo into an engineering benchmark.** Report success rate, cycle
   time, insertion pose error, peak force, energy, and categorized failures on
   held-out conditions.
3. **Measure the real mechanics.** Build or source one representative rail and
   blade, record force-displacement/friction data, and replace engineering-prior
   ranges with measured distributions.
4. **Add hardware-in-the-loop before claiming Sim2Real.** Start with the real
   camera and robot controller while the rack remains simulated, then move to a
   guarded physical insertion rig.
5. **Make autonomy operational.** Add fault detection, regrasp/retry, jam
   recovery, keep-out zones, force limits, and a human approval/abort interface.
6. **Tell the portfolio story with evidence.** Publish a 90-second video, one
   architecture graphic, reproducible benchmark JSON, model cards, limitations,
   and a release checkpoint tied to an exact environment lock.
7. **Generalize into a space-service suite.** Reuse the same evaluation harness
   for optical-module replacement, battery or pump module exchange, connector
   mating, filter replacement, and cable routing. Select tasks by avoided EVA,
   downtime, spare-mass, and recovery-value estimates—not visual novelty.

The strongest portfolio claim today is: **a reproducible, GPU-scaled,
Sim2Real-ready training and evaluation stack for orbital rack servicing**. The
claim becomes “Sim2Real validated” only after physical or HIL metrics exist.
