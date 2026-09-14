# Roadmap

The recorded transfer works, but reliability and hardware qualification remain
open. [Current measurements](docs/NOW.md) distinguish the state-driven chain,
camera-driven cohorts, and single recorded application runs.

## Completed

- A continuous zero-gravity transfer with learned capture and extraction,
  robot-carried transit, guarded insertion, and rack-only retention after release.
- A terminal mission application with packaged checkpoints, evidence checks,
  video, motion traces, and offline artifact verification.
- Fresh validation of the deployed flush-marker pair and 640-pixel camera.
- CPU tools for rack clearance, passive alignment, grip reach and entry geometry.
- Paired experiments comparing learned seating with guarded insertion, plus
  preserved failed results and corrected source/provenance checks.

## Next priorities

| Priority | Work | Completion criterion | Expected effort |
| --- | --- | --- | --- |
| 1 | Audit recovered camera and boundary experiments (T21/T22) | Reproduce report generators; promote only supported, correctly scoped results | CPU, about a day |
| 2 | Close selected historical provenance gaps (T0) | Every result used for a final claim rebuilds from reachable source and weights | CPU audit, then selected GPU reruns |
| 3 | Evaluate the camera pipeline on a larger, preregistered cohort (T1) | Report fixed seeds, confidence intervals, all failures and the unchanged 95% gate | Several GPU hours |
| 4 | Repeat training across seeds (T3) | Separate training variation from evaluation variation | Multiple overnight runs |
| 5 | Test dynamic failure mechanisms and serviceability boundaries | Predict failures from pre-outcome measurements; retain counterexamples | Instrumentation plus controlled sweeps |
| 6 | Bench-test the interface and camera geometry | Measure the first failures of the simulation assumptions | Physical fixture and camera work |

The live-service update (T7) is complete. The new camera certificate closes the
missing deployed-layout measurement in T20; checking the training noise model
against that new corpus remains part of T18. Learned seating is not promoted
on its isolated score alone. The serviceability envelope remains not qualified.

[Bounded tasks and reproduction details](docs/NEXT_WORK.md) ?
[Hardware validation plan](docs/sim_to_real.md)
