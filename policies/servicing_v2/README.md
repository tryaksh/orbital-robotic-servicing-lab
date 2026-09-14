# Mission checkpoints

These three frozen checkpoints (about 4 MB total) make the live mission reproducible
from a clone. They are byte-identical to the original training artifacts; their
SHA-256 values and original paths are in `MANIFEST.json`.

Capture uses v7m130. Extraction uses v19noised, trained with estimator noise.
The v13m130 insertion checkpoint is loaded for compatibility with the recorded
policy set; guarded control produces the insertion actions.

Training used PPO through RL-Games, at seed 70. These are simulation weights,
not hardware-qualified controllers. Full training logs and other experimental
checkpoints remain outside git. See ../../docs/compute_service_demo.md to run them.
