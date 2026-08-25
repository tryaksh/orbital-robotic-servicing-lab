# Documentation assets

`camera_frames.png` — ten unretouched 64×64 frames from the module pose head's
training set, tiled and shown at 4×. It is what the perception model actually
sees, at the resolution it sees it.

It was produced alongside `evidence/module_pose_head.json`. There is no build
step and nothing generates it on demand; regenerate it from the dataset in
`datasets/` if the camera or the crop changes.

Videos are not committed. `scripts/run_workflow_demo.py --video` writes one to
`artifacts/demo/` when asked, and `docs/compute_service_demo.md` describes the
service job that produces one per run.

A single-page portfolio that used this image was removed in commit `0b28191`;
the image is kept because it is a record of the perception input, not because
that page is coming back.
