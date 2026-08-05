# Training Guide: Teacher to Vision Student

## What is transferred?

There are two PPO training sessions with an imitation-learning step between
them. The teacher and student do not have compatible inputs or neural-network
shapes, so the teacher's weights are not copied into the student.

Instead, the transfer is behavioral:

```text
1. Teacher PPO
   exact simulator state -> teacher -> seven robot actions

2. Demonstration collection
   save camera image + robot state + phase + teacher action

3. Behavioral cloning
   train student to predict the teacher action from image + robot state

4. Student PPO
   start from the cloned student, then improve through its own simulation practice
```

The result of teacher PPO is a `.pth` teacher checkpoint. Demonstration
collection converts that policy into an HDF5 dataset. Behavioral cloning
converts the dataset into a `vision_bc.pt` student initialization. Vision PPO
then converts that initialization into a student RL-Games checkpoint.

Do not run all stages blindly in one unattended command. A poor teacher creates
poor demonstrations, which teaches the student poor behavior. Inspect and
evaluate the teacher before collecting the dataset.

## What one PPO iteration means

Both policies operate at 30 decisions per second. In one PPO iteration, each
environment collects 32 decisions, then PPO reviews that batch four times to
update the policy.

| Run | Environments | Samples per PPO iteration | Configured maximum |
| --- | ---: | ---: | ---: |
| Teacher safe starting point | 512 | 16,384 | 3,000 iterations |
| Teacher maximum tested environment count | 1,024 | 32,768 | 3,000 iterations |
| Vision student default | 128 | 4,096 | 4,000 iterations |

The 1,024-environment benchmark tested sustained environment stepping, not a
complete overnight PPO workload. Begin with 512 for the first long teacher run;
move to 1,024 only after watching VRAM and temperature during a shorter run.

## Before an overnight run

1. Plug the laptop into power and use the Windows performance power mode.
2. Prevent sleep while plugged in. Turning off the display is fine; sleeping is
   not.
3. Close games, 3D tools, video editors, and browsers consuming GPU memory.
4. Ensure the laptop has unobstructed cooling.
5. Check that no old Isaac/Kit process is running in Task Manager.
6. Keep the PowerShell window open. Closing it stops training.
7. Confirm a short run works before committing to an overnight run.

Run every command from the repository root:

```powershell
Set-Location D:\6axis-space-robotics
```

## Stage 0: short teacher learning test

Do not resume or reuse `teacher_pilot_seed42`. That run exposed the old
geometry and learned to end episodes by destabilizing the mount; it is invalid
training evidence. Start the corrected environment from a fresh policy.

This is longer than the two-iteration software smoke test but short enough to
confirm that learning, checkpointing, VRAM, and temperature behave normally:

```powershell
C:\isaac-sim\python.bat scripts\train.py `
  --task Isaac-ZeroG-BladeSwap-Teacher-v0 `
  --num_envs 512 `
  --max_iterations 200 `
  --seed 42 `
  --device cuda:0 `
  --headless `
  --run_name teacher_pilot_fixed_seed42
```

Outputs are written under:

```text
logs/rl_games/zero_g_blade_swap_teacher/teacher_pilot_fixed_seed42/
  nn/           policy checkpoints
  params/       exact environment and PPO configuration
  summaries/    TensorBoard-compatible learning events
```

The policy saves periodically after the initial 50 iterations and writes a
final checkpoint at normal completion. A checkpoint proves that work was saved;
it does not prove that useful behavior was learned.

## Stage 1: overnight teacher PPO

Recommended first overnight command:

```powershell
C:\isaac-sim\python.bat scripts\train.py `
  --task Isaac-ZeroG-BladeSwap-Teacher-v0 `
  --num_envs 512 `
  --max_iterations 3000 `
  --seed 42 `
  --device cuda:0 `
  --headless `
  --run_name teacher_full_seed42
```

After a successful 512-environment run, a higher-throughput experiment can use
1,024 environments:

```powershell
C:\isaac-sim\python.bat scripts\train.py `
  --task Isaac-ZeroG-BladeSwap-Teacher-v0 `
  --num_envs 1024 `
  --max_iterations 3000 `
  --seed 42 `
  --device cuda:0 `
  --headless `
  --run_name teacher_1024_seed42
```

During training, the easiest curriculum stage begins with a pre-aligned spare
blade. A stage advances only after at least 70 of the latest 100 completed
episodes succeed. Stage 3 is the complete eight-phase swap.

### Monitor the run

In another PowerShell window, watch GPU memory, utilization, and temperature:

```powershell
nvidia-smi -l 5
```

RL-Games prints throughput, reward, episode length, and optimization values in
the training window. The curriculum manager also reports its stage and rolling
success through the run summaries. Rising reward is encouraging, but the real
question is whether complete episodes succeed at curriculum stage 3.

The `summaries` directory contains TensorBoard-compatible files. If TensorBoard
is installed, launch it with:

```powershell
C:\isaac-sim\python.bat -m tensorboard.main `
  --logdir logs\rl_games `
  --port 6006
```

Then open `http://localhost:6006`. TensorBoard is optional and is not required
for training.

### Find and replay the latest teacher

```powershell
$teacherCheckpoint = (Get-ChildItem `
  .\logs\rl_games\zero_g_blade_swap_teacher\teacher_full_seed42\nn `
  -Filter *.pth | Sort-Object LastWriteTime -Descending | `
  Select-Object -First 1).FullName

$teacherCheckpoint

C:\isaac-sim\python.bat scripts\play.py `
  --task Isaac-ZeroG-BladeSwap-Play-v0 `
  --policy teacher `
  --checkpoint $teacherCheckpoint `
  --num_envs 1
```

Playback is a qualitative inspection, not a success-rate measurement. Watch for
the entire sequence rather than a single insertion: extract the failed blade,
stow it, retrieve the spare, align, insert, open the gripper, and retreat.

### Resume from a teacher checkpoint

Use a new run name so the original evidence remains intact:

```powershell
C:\isaac-sim\python.bat scripts\train.py `
  --task Isaac-ZeroG-BladeSwap-Teacher-v0 `
  --num_envs 512 `
  --max_iterations 4000 `
  --seed 42 `
  --checkpoint $teacherCheckpoint `
  --headless `
  --run_name teacher_full_seed42_resume1
```

`--max_iterations` is the total target iteration number, not the number of new
iterations to add. For example, resuming an iteration-3,000 checkpoint with a
target of 4,000 requests approximately 1,000 further iterations.

If training must be interrupted, stopping shortly after a scheduled checkpoint
is safer than assuming Ctrl+C will create a new checkpoint immediately.

## Teacher acceptance gate

Do not create the final student dataset merely because teacher reward increased.
At minimum, require:

- curriculum stage 3 has been reached;
- the full eight-phase sequence is visible in playback;
- success is repeatable across randomized mass, friction, mount wobble, and
  initial poses;
- no persistent workspace or mount-limit failures occur;
- results are reproduced with multiple random seeds before making a portfolio
  performance claim.

The current repository provides training and playback but not yet a dedicated
multi-episode evaluation script that calculates a formal success percentage.
Adding that evaluator is the next important engineering step before declaring a
checkpoint ready.

## Stage 2: collect teacher demonstrations

Once the teacher is accepted, run it in the visual Play environment. The
teacher still acts from exact state, while the collector records what the
student would have available at that same moment:

- 64x64 RGB image;
- 47-value robot/proprioception vector, which already includes task phase;
- a separate phase field for dataset inspection;
- the teacher's seven-value action.

Collect 250,000 labeled examples:

```powershell
C:\isaac-sim\python.bat scripts\collect_teacher.py `
  --checkpoint $teacherCheckpoint `
  --samples 250000 `
  --num_envs 128 `
  --seed 42 `
  --output datasets\teacher_250k.h5 `
  --headless
```

The dataset is compressed but can still consume several gigabytes. It is
excluded from Git.

## Stage 3: behavioral-cloning transfer

Behavioral cloning is supervised learning: show the student an image and robot
state, ask it to predict the teacher's action, measure the difference, and
repeat. It does not run Isaac Sim.

```powershell
C:\isaac-sim\python.bat scripts\pretrain_student.py `
  --dataset datasets\teacher_250k.h5 `
  --output checkpoints\vision_bc.pt `
  --epochs 30 `
  --batch_size 512 `
  --seed 42
```

The script reserves 5% of the demonstrations for validation. It saves the
student actor with the lowest validation loss and writes the learning history to
`checkpoints/vision_bc.history.json`.

Low imitation loss means the student can copy teacher actions on recorded
examples. It does not prove that the student can recover when its own action
moves it into a situation absent from the dataset. That is why student PPO is
still required.

## Stage 4: overnight vision-student PPO

The `--bc_checkpoint` option loads only the behavior-cloned vision actor. The
student's privileged training critic starts separately, and PPO then improves
the policy through new camera-based simulation experience.

```powershell
C:\isaac-sim\python.bat scripts\train.py `
  --task Isaac-ZeroG-BladeSwap-Vision-v0 `
  --num_envs 128 `
  --max_iterations 4000 `
  --seed 42 `
  --bc_checkpoint checkpoints\vision_bc.pt `
  --headless `
  --run_name vision_full_seed42
```

The vision actor sees only RGB and deployable robot state. During training, an
asymmetric critic may see exact blade state and randomized physics to provide a
better learning signal. The critic is training scaffolding and is not needed
when the deployed actor selects actions.

Find and play the newest student checkpoint:

```powershell
$visionCheckpoint = (Get-ChildItem `
  .\logs\rl_games\zero_g_blade_swap_vision\vision_full_seed42\nn `
  -Filter *.pth | Sort-Object LastWriteTime -Descending | `
  Select-Object -First 1).FullName

C:\isaac-sim\python.bat scripts\play.py `
  --task Isaac-ZeroG-BladeSwap-Vision-v0 `
  --policy vision `
  --checkpoint $visionCheckpoint `
  --num_envs 1
```

## Stage 5: when is the student ready?

Use three levels of readiness:

1. **Training integration ready:** saves, reloads, and produces finite actions.
   The current smoke checkpoint meets only this level.
2. **Simulation task ready:** achieves a predefined full-swap success rate over
   many unseen randomized episodes and multiple seeds, without safety failures.
3. **Physical-system ready:** passes camera calibration, force-limited HIL,
   guarded real-rig trials, failure recovery, and emergency-stop validation.

A reasonable portfolio target is at least 90% full-swap success in the training
distribution and 80% on held-out harder conditions across three seeds. These
are project acceptance targets, not proof of safe flight deployment.

## Recommended sequence

```text
Teacher pilot (200 iterations)
    -> inspect logs and playback
Teacher overnight (up to 3,000 iterations)
    -> formal evaluation should be added and run
250k demonstration collection
    -> inspect dataset
30-epoch behavioral cloning
    -> check training versus validation loss
Vision PPO (up to 4,000 iterations)
    -> held-out simulation evaluation
HIL and physical test rig
```

For portfolio credibility, repeat the final teacher and student experiments with
at least seeds 42, 43, and 44. Keep checkpoints, exact parameter YAML files,
evaluation JSON, and videos associated with the Git commit that produced them.
