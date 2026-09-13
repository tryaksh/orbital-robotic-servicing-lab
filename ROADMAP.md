# Roadmap

**What is closed, what is open, and what each open thing would cost.** This is
the honest summary; [`docs/NEXT_WORK.md`](docs/NEXT_WORK.md) is the long form,
where every item below has a section with its full reasoning under the task
number given here.

Unlike the other repository in this pair,
[constrained-cable-safety](https://github.com/tryaksh/constrained-cable-safety),
**this project is not finished.** Its headline claim fails its own gate and its
serviceability envelope is not qualified. Nothing below hides that.

Last reviewed 2026-09-13. The most recent measurement is dated 2026-09-04.

---

## What is settled

These are answered, and re-running them would not change anything. Each is
backed by a canonical report; [`README.md`](README.md) explains what they mean
for someone who has not seen the project.

| Question | Answer | Report |
| --- | --- | --- |
| Can a robot do the whole swap in one episode, holding the module throughout, with no simulator tricks? | **Yes.** 22 of 24 episodes on three held-out seeds under the strict rule. It does not clear the 95% gate, but it works. | [`workflow_robot_carried_release_rack_retention_v1_certification.json`](evidence/workflow_robot_carried_release_rack_retention_v1_certification.json) |
| Is rack-side retention worth having? | **Yes, and it is the difference.** 22/24 with it, 17/24 without, on the same code path. | [`..._rack_retention_control_v1_...`](evidence/workflow_robot_carried_release_rack_retention_control_v1_certification.json) |
| Can a plain two-finger gripper on a smooth post hold the extraction load? | **No, by a factor of eleven.** About 6 N held against 66.4 N demanded. Gripping harder made it worse. | [`boundary_failure_modes_v1.json`](evidence/boundary_failure_modes_v1.json) |
| How square can a rack hold a part that is only resting in it? | **`2c/L`**, and no squarer. This rack held a module at 56.40 mrad while demanding 52.36 mrad to accept it — it was asking for something its own geometry forbids. | [`destination_channel_geometry.json`](evidence/destination_channel_geometry.json) |
| Was the seating skill's failure a reward problem? | **No.** Three different objectives ended at 84.26, 84.61 and 84.58 mrad against a 52.4 mrad tolerance. Landing within half a milliradian of each other is the result. | [`insert_attitude_diagnosis.json`](evidence/insert_attitude_diagnosis.json) |
| Do isolated skill scores predict chain performance? | **No, and this is the repository's most transferable finding.** The seating skill certifies at 36.77% alone and scores 0 of 96 on the handoffs its own chain delivers. A certificate describes the states it was measured on, not the states its caller supplies. | [`seating_controller_head_to_head.json`](evidence/seating_controller_head_to_head.json) |
| Which design dimensions actually move the outcome? | Module cross-section and where the robot parks. Not mass. A 120 × 16 mm section takes the chain from 93.75% to 0.00%; a 10 mm park error takes it to 6.25%. | [`chain_robustness_sweep_section_n192_v1.json`](evidence/chain_robustness_sweep_section_n192_v1.json) |
| Can the fiducial marker be seen along the whole seating stroke? | **Yes, now.** The old certificate was retracted — its marker floated 90 mm above the module. Moving and aiming the fixed camera, with the accuracy gates unchanged, took held-out detection of the critical rack from 43.27% to 99.85%. | [`servicing_camera_geometry_v4_datum_pair.json`](evidence/servicing_camera_geometry_v4_datum_pair.json) |

---

## What is open

Sorted by what it would take, not by how interesting it is.

### Done in this session, from what was already on disk

No simulator was run. These were code and record defects that were costing real
runs.

- **The first seating policy that can feel contact was never scored on its own.**
  `verify_insert_skill.sh` ran three times and each run exited in about ten
  seconds having produced nothing, because the play configuration raised
  `TypeError` at construction: it borrowed another class's `__post_init__` with
  its own `self`. Fixed, and a test now rejects that form. **The run still has to
  happen** — see the priced list below.
- **`check_perturbations_bite.py` had never produced a report.** It imported a
  helper that was deleted for being inert, so the script that exists to catch
  inert probes raised `ImportError` before rendering a frame. Fixed, with the
  camera-displacement check replaced by a statement of where that measurement
  actually comes from.
- **`check_destination_channel.py` imported a namespace package to register the
  tasks**, which runs no code, so `parse_env_cfg` would not have found the task.
  Fixed.
- **`promote_checkpoints.py` resolved an ambiguous checkpoint silently, and to
  the wrong file** (task **T8**). Extract epoch 12600 exists under two filenames
  with byte-identical weights; the old `(size, name)` tie-break picks the one the
  current certification was *not* produced from. It now refuses and asks, with
  `--resolve <run>=<filename>`. No published number moves — the weights are equal.
- **Three campaign queues had been exempt from the `$?` lint since 2026-09-03**
  while mid-run. The runs finished; the exemption outlived them by ten days,
  because a skipped test is silent. Fixed and the exemption list is empty.
- **A static import check now covers every Python file in the repository**, so a
  deleted module or a renamed function cannot sit in a simulator entry point
  unnoticed again. It found all three of the defects above on its first run.

### Cheap, and the next thing anyone should do

| # | What | Cost |
| --- | --- | --- |
| — | **Score the contact-feeling seating policy on its own.** Now unblocked. The chain half already ran and scored 4/24 on one seed in a bay 3.897 mm outside its own requirement, which decides nothing. The skill half has never run. Its training reward reached 98.2 against the blind policy's 43.9 on an identical reward function, in a quarter of the epochs — and **training reward is not a success rate and must not be quoted as one.** | three `play` runs, minutes of GPU each |
| **T19** | **The kinematics agreement check fires spuriously**, about one run in fifteen, and costs a sweep point when it does. It takes `.max()` across every environment at one instant, so one environment whose joints are not yet written is enough. Do not widen the tolerance — it is right and it has caught real defects. Run it per environment, or after the first reset has stepped everywhere. | under an hour, plus re-running one sweep point |
| **T6** | **Attribute the 8 points that capture and extraction are missing.** Both sit at 86.90% and 87.64% against 95%, and both overlap the certificates they were supposed to beat. Nobody has asked *which* failure mode accounts for the gap. | evaluation only, no retraining |
| **T7** | **The live demonstration service runs a superseded policy set.** Small if folded into the next certification run. | small |

### Needs a real simulator run, and is worth buying

| # | What | Cost |
| --- | --- | --- |
| **T1** | **Certify the strict chain on the camera-driven task.** This is the strongest claim the project could make, and the one it is furthest from: 4/24 by camera against 20/24 from simulator-supplied poses, with the best configuration found so far at 17/24. | hours, one batch |
| **T11** | **No recording shows the certified chain.** Every clip in the repository is from superseded checkpoints, pre-fix geometry, or both, and none achieved settled seating. One is misnamed: `3_full_chain_seated.mp4` reports a 4.62 mm lateral error against a 2.5 mm tolerance. They are an honest record of the problem and a dishonest record of the solution. | about 8 minutes a clip |
| **T4** | **Exercise robustness levels 1 to 4** to turn a single point into a degradation curve. | evaluation only |

### Needs new collection, and is not being started

Written down with its price so nobody has to rediscover it. **None of this was
begun in this session.**

| # | What | Price |
| --- | --- | --- |
| **T0** | **Ten reports cannot be reproduced.** They were produced from uncommitted code. Any of them a final claim depends on has to be re-run from a clean commit. | CPU plus certification batches, sized by how many claims you keep |
| **T9** | **The seating skill's load path is not the chain's.** The blocking item for a learned seating phase: give the skill the chain's mating compliance rather than just switching the lock on. | half a day of work plus about 2 hours of GPU |
| **T3** | **Every number here comes from one training seed.** Three seeds a skill, so results carry a spread rather than a point. Until this exists, no claim about a *method* is supportable. | four or more training runs |
| **T5** | **Randomise, during training, the variables the robustness sweep says the chain is sensitive to**, so the output is a tolerance band instead of a point. | retrain plus re-certify |
| **T16** | **Re-sweep rack clearance with the entry mouth moving with the walls.** The original sweep moved each bay's guides and left its lips and flares behind; 6 mm per side goes from 0/64 to 36/64 once the mouth moves too. A partial re-sweep landed on 2026-09-03; the axis is not closed. | one flag plus a re-sweep |
| **T20** | **There is no perception certificate for the datum layout actually deployed.** The surrogate's noise model comes from a single-datum certificate and the deployment uses a pair. | one collection |
| **T13** | **Learned insertion interface transfer**, paused until reset states equal real handoffs. | no GPU yet; blocked on T9 |

### Deliberately not doing

- **Moving the 95% gate.** It is 95%, the chain scores 91.67%, and that is the
  result. A gate moved after seeing the number is not a gate.
- **Renaming the `zero_g_blade_swap` package** to something that matches what the
  project became. 35 evidence reports record the hash of a file at
  `src/zero_g_blade_swap/…`; renaming would break every one of those provenance
  links to make a directory listing read better. [`README.md`](README.md)
  explains the name instead.
- **Quoting the 97.92% chain result as current.** It is a legacy
  supported-settle baseline, measured while the robot was still helping hold the
  module. It stays in the records as a comparator and nowhere else.
- **Any hardware claim at all.** Nothing here has run on hardware.

---

## The manuscript

A manuscript draft exists and lives outside both repositories, in
`D:/orbital-servicing-paper` on the workstation, with no git remote. It is not
tracked here, it is not backed up anywhere, and this session deliberately did not
touch it. [`docs/paper_position.md`](docs/paper_position.md) carries the
literature check and says what the paper may and may not claim;
[`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md) is the record of the variables that
were frozen, and its framing has been superseded by the position document.

The honest state: the claim that survived a literature check is the rack
requirement — that a channel may not hold a module outside its own acceptance
criterion — and the claim that the interface, not the reward, sets the seating
angle. The end-to-end chain is not novel and the hybrid architecture has named
prior art.
