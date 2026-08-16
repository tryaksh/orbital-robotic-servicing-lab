# Portfolio page assets

`../portfolio.html` is a single self-contained page — no build step, no
framework, no external requests. It expects one asset beside it:

- `camera_frames.png` — ten unretouched 64x64 frames from the pose head's
  training set, shown at 4x. Referenced as `assets/camera_frames.png`.

The workflow video is **not** committed, because this repository keeps videos out
of Git. It is written to `artifacts/demo/vision_install/rl-video-step-0.mp4` by:

```
scripts/run_workflow_demo.py --headless \
  --task Isaac-ZeroG-Blade-GrappleVision-Workflow-v0 --workflow install \
  --curriculum_stage 2 --num_envs 1 --video \
  --pose_head_checkpoint checkpoints/module_pose_head.pth \
  --grasp_checkpoint ... --extract_checkpoint ... --insert_checkpoint ...
```

To put it on the page, copy it next to `portfolio.html` and drop a `<video>` in
place of the `<figure>` holding the camera strip, or beside it.

Every number on the page names the file in `evidence/` it came from, so each one
can be checked rather than taken on trust.
