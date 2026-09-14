# Demonstrations

**Two recordings now show one episode each of the chain as it is currently
certified, and both were checked against their own reports rather than against
their filenames.**

A cohort is twenty-four episodes and a clip is one, so a clip can never *be* the
certified 22/24; the most it can be is one episode of the same arm, and that is
what these are. Every other file on disk predates the changes that produced the
current numbers, which was the finding of the 2026-08-25 media audit — that table
is kept below, because publishing a clip that looks like the current system and is
not would overstate exactly the thing this project is careful about.

Videos are **not committed**. `*.mp4` is gitignored and stays that way: git stores
video as opaque blobs, so every re-render adds a permanent full copy and the cost
of a clone grows for everyone, forever, including CI. The repository is about
21 MB and should stay that size. Media belongs on a release, and a release should
only carry footage that is true.

## The certified chain, one episode each

| | seed 5070 | seed 6070 |
| --- | --- | --- |
| File | `artifacts/robotcarried/video_certified_chain_clip_seed5070/rl-video-step-0.mp4` (24 MB) | `..._seed6070/rl-video-step-0.mp4` (23 MB) |
| Run | `artifacts/robotcarried/certified_chain_clip_seed5070_report.json` | `..._seed6070_report.json` |
| Source | commit `db5b79f`, tracked worktree clean | the same |
| Task | `Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0` — the task the 22/24 was measured on | the same |
| All seven insertion conditions | true | true |
| Settled seating still held after 0.70 s | true | true |
| Both robot-side supports released | true | true |
| Rack alone through the full recheck | true | true |
| Terminal lateral error | 0.113 mm | 0.63 mm |

Against a 2.5 mm lateral tolerance, so both are two orders inside it. Neither is a
rate: the recorder refuses more than one environment while the certification runs
eight, so each draws a single environment's reset state rather than one of that
cohort's eight, and lighting is fixed for the recorder rather than randomized. Both
differences are named in the script that produced them.

## The camera-driven changeout

| | |
| --- | --- |
| File | `artifacts/robotcarried/video_datum_pair_rack_clean_seed6070/rl-video-step-0.mp4` (26 MB, not committed) |
| Run | [`evidence/rgbd_strict_rack_retention_datum_pair_seed6070.json`](../evidence/rgbd_strict_rack_retention_datum_pair_seed6070.json) |
| Source | commit `7a82db2`, tracked worktree clean |
| What it shows | trained capture, trained extraction, robot-carried transit, guarded insertion to the derived seated plane at 0.676 m, both robot-side supports released, and the rack alone holding the module for 0.733 s |
| Perception | live throughout: 1,772/1,772 detections, zero failures, both flush plates used |
| What it is not | the certified chain. It is the **camera-driven** task, whose pooled rate is 4/24, and it is one episode at one seed with visual randomization off for recording. It is a favourable sample of a harder task, and it is labelled as one everywhere it appears. |

This is the clip worth publishing for what it shows that the two above do not:
perception driving the guarded advance, live, for the whole stroke.

## The three fields that decide whether a clip may be published

Read them out of the report of the run that produced the clip. Never off the
filename, and never off what the footage looks like.

```
seated_conditions_still_held_after_settling
all_conditions_including_released_gripper
destination_rack_retention.observed_per_environment[0].full_rack_only_recheck_observed
```

All three are true in all three runs above. A clip whose run reports `false` on any
of them is a record of a failure, whatever it looks like.

## Two files carried names their own runs contradict, and were renamed

Both are kept. A failed run is a result and gets to stay; what it does not get is
a name that claims something its report denies.

| Was called | Is now called | What its own report says |
| --- | --- | --- |
| `3_full_chain_seated.mp4` | `3_insertion_missed_the_lateral_gate.mp4` | `artifacts/robotcarried/video_chain_report.json`: `lateral_alignment: false` at **4.62 mm** against a 2.5 mm tolerance, `seated_conditions_still_held_after_settling: false`, `reached_phase: transit`. It did not seat. |
| `4_push_in_and_release.mp4` | `4_push_in_attempt_no_release.mp4` | Same run. `all_conditions_including_released_gripper: false`. Nothing was released. |

The 4.62 mm lateral failure is the blocker that was later **closed**, by deriving
both lead-ins from the rail face. So these clips are an honest record of the
problem and were a dishonest record of the solution.

## What the other files show

Each was checked against the report of the run that produced it.

| File | What it really is |
| --- | --- |
| `artifacts/robotcarried/video/1_grasp_and_extract.mp4` | Learned capture and extraction — but driven by the **superseded w65 checkpoints** (`grapple_grasp_l0_seed70_v6w65`, `grapple_extract_l0_seed70_v16w65`), not the certified v7m130 / v18pin set. Source run: `artifacts/robotcarried/video_full_report.json`, which ends `reached_phase: transit` with 43.2 mm of final lateral error. The two phases in the title did happen. |
| `artifacts/robotcarried/video/2_carry_across_on_the_rail.mp4` | The robot-carried transit from the same run, so the same checkpoint caveat. The carry itself is real and is the claim the form lock supports. |
| `artifacts/demo/vision_clean/`, `vision_install/` | Perception clips from **2026-08-15**, which predates the workcell move, the 130 × 20 mm module and the derived rack. Geometrically obsolete. |
| `artifacts/robotcarried/video/chain/`, `full/`, `artifacts/service_e2e_final/` | Raw uncut recordings, 129–188 MB each, behind the cuts above. |

## Recording one

`artifacts/campaign/queue_certified_chain_clip.sh` runs the certified arm exactly
— state task, robot rail, form lock, rack retention, simultaneous release, the
0.70 s rack-only recheck — and checks the three fields above out of the report it
writes. About eight minutes a seed.

Two differences are forced on it and are named in the script: `--video` refuses
more than one environment while the certification runs eight, so it draws a single
environment's reset state rather than one of that cohort's eight; and the recorder
needs a fixed exposure while the evidence needs the randomization it was certified
under, so lighting is stable. The state task carries no perception, so the second
costs nothing measured here.

For the camera-driven task:

```bash
scripts/run_robot_carried.sh rgbd
```

## The set worth publishing

Four clips, each captioned with the report that backs it:

1. **Learned capture and extraction** — the two phases a policy owns.
2. **Robot-carried transit** — the central claim: the module is held by the arm
   throughout, on a visible robot-side form lock, with no world constraint, no
   teleport and no hidden carrier.
3. **The complete chain on the certified state task**, with a run that reports
   true on all three fields above. Both clips above qualify.
4. **Perception** — RGB-D driving the guarded advance on the current geometry.

All four exist. What does not exist, and cannot, is a clip that carries a rate:
demonstrated capability is claimed from `evidence/` and footage is what it looks
like.

## Why this is not just tidiness

The project's own completion rule requires that "the compute service must save a
clear video and hashed artifacts". A video that is saved and misdescribed
satisfies the letter of that and defeats its purpose. The rule that matters here is
the same one that governs the numbers: **a demonstration is labelled by what the
controller actually did, not by what the file is called.**
