# Demonstrations

**No recording in this repository shows the certified chain.** That is the
finding of the 2026-08-25 media audit, and it is written here rather than worked
around, because publishing a clip that looks like the current system and is not
would overstate exactly the thing this project is careful about.

Videos are **not committed**. `*.mp4` is gitignored and stays that way: git stores
video as opaque blobs, so every re-render adds a permanent full copy and the cost
of a clone grows for everyone, forever, including CI. The repository is ~21 MB
and should stay that size. Media belongs on a release, and a release should only
carry footage that is true.

---

## What the existing files actually show

Every clip on disk was produced before the changes that produced the current
97.92%, and each was checked against the report of the run that produced it
rather than against its filename.

| File | What it really is |
| --- | --- |
| `artifacts/robotcarried/video/1_grasp_and_extract.mp4` | Learned capture and extraction — but driven by the **superseded w65 checkpoints** (`grapple_grasp_l0_seed70_v6w65`, `grapple_extract_l0_seed70_v16w65`), not the certified v7m130 / v18pin set. Source run: `artifacts/robotcarried/video_full_report.json`, which ends `reached_phase: transit` with 43.2 mm of final lateral error and `seated_conditions_still_held_after_settling: false`. |
| `artifacts/robotcarried/video/2_carry_across_on_the_rail.mp4` | The robot-carried transit from the same run, so the same caveat. The carry itself is real and is the claim the form lock supports. |
| `artifacts/robotcarried/video/3_full_chain_seated.mp4` | **Misnamed.** Source run `artifacts/robotcarried/video_chain_report.json` did not seat: `lateral_alignment: false`, 4.62 mm against a 2.5 mm tolerance, settled re-check false. Do not publish this as a seated chain. |
| `artifacts/robotcarried/video/4_push_in_and_release.mp4` | The insertion attempt from that same non-seating run. |
| `artifacts/demo/vision_clean/`, `vision_install/` | Perception clips from **2026-08-15**, which predates the workcell move, the 130 × 20 mm module and the derived rack. Geometrically obsolete. |
| `artifacts/robotcarried/video/chain/`, `full/`, `artifacts/service_e2e_final/` | Raw uncut recordings, 129–188 MB each, behind the cuts above. |

The 4.62 mm lateral failure in the third row is the blocker that was **closed** by
deriving both lead-ins from the rail face. So those clips are an honest record of
the problem, and a dishonest record of the solution.

## Producing honest media

One command, one environment, RGB-D perception active, video written alongside
the report that says whether the run actually seated:

```bash
scripts/run_robot_carried.sh rgbd
```

It runs `Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0` with
`--perception_backend fiducial_pnp` and writes to `artifacts/robotcarried/video/`.
About eight minutes on the measured machine.

**Check the report before publishing the clip.** The field that decides it is
`seated_conditions_still_held_after_settling`. A clip whose run reports `false` is
a record of a failure, whatever it looks like, and every file in the table above
is one.

Note that `run_robot_carried.sh` now defaults to the certified checkpoint set, so
a recording made today uses the policies the published rate was measured on —
which was not true of any recording listed above.

## The set worth publishing, once it exists

Three or four clips, each captioned with the report that backs it:

1. **Learned capture and extraction** — the two phases a policy owns.
2. **Robot-carried transit** — the central claim: the module is held by the arm
   throughout, on a visible robot-side form lock, with no world constraint, no
   teleport and no hidden carrier.
3. **The complete chain, seating and releasing** — with a run that reports
   `seated_conditions_still_held_after_settling: true`.
4. **Perception** — RGB-D driving the guarded advance, on the current geometry.

Until (3) exists, the repository should claim demonstrated capability from
`evidence/` and not from footage.

## Why this is not just tidiness

The project's own completion rule requires that "the compute service must save a
clear video and hashed artifacts". A video that is saved and misdescribed
satisfies the letter of that and defeats its purpose. The rule that matters here
is the same one that governs the numbers: **a demonstration is labelled by what
the controller actually did, not by what the file is called.**
