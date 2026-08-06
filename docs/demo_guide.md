# High-Quality Insertion Demonstration

The first reliable presentation milestone is a nominal replacement-blade
insertion: approach the pre-aligned blade, close the Robotiq gripper, align
outside the rack, insert, release, and retreat.

Close every existing Isaac Sim window before launching the demo. Multiple Kit
processes compete for shader caches and make a one-environment presentation run
unnecessarily slow.

## Watch it live

From `D:\6axis-space-robotics`:

```powershell
C:\isaac-sim\python.bat scripts\scripted_demo.py `
  --task Isaac-ZeroG-BladeSwap-Play-v0 `
  --num_envs 1 `
  --steps 900 `
  --seed 42 `
  --device cuda:0 `
  --grasp_mode kinematic `
  --real_time `
  --rendering_mode quality
```

The application exits after formal task success. The nominal validated run
reached the release/retreat phase at step 656 and full success at step 708.

## Record it

```powershell
C:\isaac-sim\python.bat scripts\scripted_demo.py `
  --task Isaac-ZeroG-BladeSwap-Play-v0 `
  --num_envs 1 `
  --steps 900 `
  --seed 42 `
  --device cuda:0 `
  --grasp_mode kinematic `
  --video `
  --video_length 900 `
  --rendering_mode quality
```

Video output is written under `videos/scripted_demo/`; the machine-readable
result is `artifacts/scripted_demo_report.json`.

## What this demonstration proves

- The official Isaac Lab UR10e/Robotiq frame convention is used.
- The Cartesian controller can reach the handle and execute a staged insertion.
- The phase manager recognizes insertion, release, retreat, and success.
- The rendered orbital workcell can produce portfolio footage.

It is not an RL result. `kinematic` mode attaches the blade after the gripper is
precisely aligned and closed, and disables proxy-handle collision for that
nominal presentation. This is explicitly reported in the JSON output.

The next Sim2Real gate is to replace this attachment progressively: kinematic
assist, then compliant assist, then validated finger/handle contact. PPO or
behavioral cloning should learn from successful reference motion during that
curriculum instead of exploring the entire manipulation sequence from scratch.
