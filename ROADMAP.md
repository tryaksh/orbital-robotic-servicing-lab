# Roadmap

**What is closed, what is open, and what each open thing would cost.** This is
the honest summary; [`docs/NEXT_WORK.md`](docs/NEXT_WORK.md) is the long form,
where every item below has a section with its full reasoning under the task
number given here.

Unlike the other repository in this pair,
[constrained-cable-safety](https://github.com/tryaksh/constrained-cable-safety),
**this project is not finished.** Its headline claim fails its own gate and its
serviceability envelope is not qualified. Nothing below hides that.

Last reviewed 2026-09-13. The most recent simulator measurement is dated 2026-09-06.

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
| How square can a rack hold a part that is only resting in it? | **`2c/L`**, and no squarer. This rack can leave a module lying over 69.68 mrad while demanding 52.36 mrad to accept it — it is asking for something its own geometry forbids. | [`destination_channel_geometry.json`](evidence/destination_channel_geometry.json) |
| Was the seating skill's failure a reward problem? | **No.** Three different objectives ended at 84.26, 84.61 and 84.58 mrad against a 52.4 mrad tolerance. Landing within half a milliradian of each other is the result. | [`insert_attitude_diagnosis.json`](evidence/insert_attitude_diagnosis.json) |
| Do isolated skill scores predict chain performance? | **No, and this is the repository's most transferable finding.** The seating skill certifies at 36.77% alone and scores 0 of 96 on the handoffs its own chain delivers. A certificate describes the states it was measured on, not the states its caller supplies. | [`seating_controller_head_to_head.json`](evidence/seating_controller_head_to_head.json) |
| Would a seating policy that can feel contact fix it? | **No, and it is the cleanest version of the finding.** It certifies at **99.20%** on 3,001 episodes — the first learned seating skill here to pass its own gate — and scores **24/96** in the chain against the scripted controller's **23/96** on the same rack. One episode in ninety-six. | [`grapple_insert_v33force_c11065_certification.json`](evidence/grapple_insert_v33force_c11065_certification.json) |
| Why do capture and extraction miss the 95% gate? | Two different reasons, and neither is precision. **1,170 of capture's 1,180** failures never get the gripper within the 10 mm the chain allows, at a median 95.9 mm. **1,024 of extraction's 1,113** leave the module moving faster than the derived settling limit — it comes out of the bay and does not stop. | [`skill_gate_attrition_v1.json`](evidence/skill_gate_attrition_v1.json) |
| Can a hardware engineer check a rack before anyone cuts metal? | **Yes, in one command.** `scripts/check_channel_holds_its_tolerance.py` takes a module and a slot and returns whether the slot can hold the module inside its own acceptance test. On this repository's bay it returns no, on both axes. | [`channel_verdict_shipped_bay_v1.json`](evidence/channel_verdict_shipped_bay_v1.json) |
| Which design dimensions actually move the outcome? | Module cross-section and where the robot parks. Not mass. A 120 × 16 mm section takes the chain from 93.75% to 0.00%; a 10 mm park error takes it to 6.25%. | [`chain_robustness_sweep_section_n192_v1.json`](evidence/chain_robustness_sweep_section_n192_v1.json) |
| Can the fiducial marker be seen along the whole seating stroke? | **Yes, now.** The old certificate was retracted — its marker floated 90 mm above the module. Moving and aiming the fixed camera, with the accuracy gates unchanged, took held-out detection of the critical rack from 43.27% to 99.85%. | [`servicing_camera_geometry_v4_datum_pair.json`](evidence/servicing_camera_geometry_v4_datum_pair.json) |

---

## Closed on 2026-09-13, without a simulator

Record and code defects that were costing real runs. No simulator was needed for
any of them.

- **The seating experiment was already run and its evidence had been lost.** The
  play configuration's `TypeError` was fixed on 2026-09-04, the runs were retried
  the same day, and the certificates went into a branch that was retired two days
  later as a different project. Recovered from the tag, re-verified against the
  episode archives on disk, and written up above.
- **Fifty-four reports came back with it**, including the whole camera-driven
  factorial and a library correction with its tests. See
  [`docs/REPO_MAP.md`](docs/REPO_MAP.md).
- **The design library contradicted itself and the contradiction said yes.**
  `section_verdict` read one of the two bounds `lateral_clearance_window`
  publishes, so the tool accepted a bay the same file called 3.897 mm too wide.
  Recovered with the campaign; accepted cross-sections fall from 7 of 36 to 4.
- **The kinematics agreement check fired when nothing was wrong** (task **T19**).
  It reduced every environment into one verdict, so one environment whose joints
  had not been written yet failed the run, about one in fifteen. It is per
  environment now and names the one that disagrees. The tolerance is unchanged and
  a test pins it.
- **Two clips carried names their own runs contradict.**
  `3_full_chain_seated.mp4` reported `lateral_alignment: false` at 4.62 mm against
  a 2.5 mm tolerance and `4_push_in_and_release.mp4` never released. Both renamed
  to what their reports support; the footage is kept.
- **`check_source_provenance.py` searched only `HEAD`.** A commit reachable from
  any tag here is one a reader can check out, and the recovered campaign's
  bindings were being called lost. Searching every ref recovers six reports
  instead of one, and scoping the walk to the commits that touched each path made
  it seven times faster rather than slower.
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

---

## What is open

Sorted by what it would take, not by how interesting it is.

### Cheap, and the next thing anyone should do

| # | What | Cost |
| --- | --- | --- |
| **T21** | **Fifty-one of the fifty-four recovered reports have not been read.** They arrived classified `historical`, which is the safe default and not a verdict. Among them: the complete camera-driven 2×2×2 factorial, a gravity ladder, paired n=192 arms, and three prediction scorecards. Reading them could close open questions with measurements that already exist. | CPU, a day of reading |
| **T22** | **Verify the nineteen recovered generators against their own reports.** They are restored and the suite passes; what has not happened is running each one and comparing its output to the report it produced, the way `check_reproducible_from_source.py` does. Restoring the code is also what found a published p-value that had the direction wrong. | a few hours, CPU |
| **T7** | **The live demonstration service runs a superseded policy set.** Small if folded into the next certification run. | small |

### Needs a real simulator run, and is worth buying

| # | What | Cost |
| --- | --- | --- |
| **T1** | **Certify the strict chain on the camera-driven task.** This is the strongest claim the project could make, and the one it is furthest from: 4/24 by camera against 20/24 from simulator-supplied poses, with the best configuration found so far at 17/24. | hours, one batch |
| **T11** | **No recording shows the certified chain.** The nearest is one complete RGB-D changeout at seed 6070, which is a different task and n = 1. The misnamed clips are renamed and [`docs/DEMOS.md`](docs/DEMOS.md) states what each one actually is. | about 8 minutes a clip |
| **T4** | **Exercise robustness levels 1 to 4** to turn a single point into a degradation curve. | evaluation only |

### Needs new collection, and has not been started

Written down with its price so nobody has to rediscover it.

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
tracked here and it is not backed up anywhere.
[`docs/paper_position.md`](docs/paper_position.md) carries the literature check and
says what the paper may and may not claim;
[`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md) records the variables that were frozen,
and its framing is superseded by the position document.

The honest state: the claim that survived a literature check is the rack
requirement — that a channel may not hold a module outside its own acceptance
criterion — and the claim that the interface, not the reward, sets the seating
angle. The end-to-end chain is not novel and the hybrid architecture has named
prior art.
