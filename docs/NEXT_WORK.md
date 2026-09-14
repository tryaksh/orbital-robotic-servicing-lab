# Next work

Priority follows the measured gates in [NOW.md](NOW.md). The application works;
the 95% full-chain gate and the broader serviceability envelope remain open.
Detailed earlier experiments and commands are preserved in the
[backlog archive](handover/backlog_before_mission_application.md). That archive
contains superseded status statements; use this file for current priority.

## T21 / T22: audit recovered experiments

The September campaign's recovered camera factorial, gravity ladder and boundary
reports need individual scope and source checks. Some arms outperform the
canonical recipe, so do not call the selected application configuration optimal.
Recovery from a tag alone does not make a result current.

1. Read the report's scope, cohort definitions, controller flags and provenance.
2. Run its restored generator against the original local episode archives.
3. Compare the generated values with the preserved report, including directional
   paired tests. Keep differences as findings.
4. Promote only verified reports in `scripts/build_evidence_manifest.py`; update
   the owning summary in NOW.md and preserve the previous arm.

This is CPU work, roughly a day. Start here before buying more GPU time.
The [repository map](REPO_MAP.md) records which generators and reports came back.

## T0

Twenty-seven older source-bound reports cannot be fully recovered because at
least one runtime file came from uncommitted code. The new application and
perception reports have reachable source, and the live mission weights ship in
git; this does not repair the older experiments.

```powershell
python scripts/check_source_provenance.py --depth 200
```

Select which claims remain in scope, recover matching files across all refs,
and rerun any still-unrecoverable certification on a clean commit. Store a new
versioned report rather than overwriting the old one. Completion means every
selected claim has reachable source, weights, configuration, seeds and artifacts.
Cost: minutes for the audit; hours per selected GPU certification.

## T1: stronger camera-driven evaluation

The canonical combined camera recipe scores 17/24 and misses the 95% gate.
The new mission is a successful recording at seed 6070, not another pooled
certificate. Its stable lighting and single environment differ from the
randomized, eight-environment research cohorts.

After the recovered factorial audit, preregister a larger set of held-out seeds,
keep the same checkpoints and final predicates, and run the selected camera arm
beside its oracle-pose control. Record success, failures by phase, confidence
intervals, pose-source and velocity-source flags, and source hashes. Preserve
both arms and all episodes. Use one sequential supervisor for dependent stages.
Cost: several GPU hours; rendering is the main expense.

## T3: training repeatability

Headline policies come from one training seed. Repeat capture and extraction
training at additional seeds before claiming a method-level improvement.
Certify all seeds on the same held-out cohorts and compare spread across training
runs. Any reward or criterion change needs its own preserved control.
Cost: multiple overnight trainings; monitor system RAM as well as GPU memory.

## T18 / T20: match sensing during training

T20's missing measurement is closed by
[the new 1,024-frame certificate](../evidence/fiducial_rgbd_service_current_seed287.json).
It uses the deployed 640-pixel camera and flush datum pair, with unchanged gates.
It is a held-pose test, so it does not substitute for moving-robot evaluation.

T18 still needs an audit of the training-time surrogate against this corpus.
Check residuals, cadence, missed detections and derived velocity. The selected
v19noised checkpoint remains unchanged. Recalibrating the surrogate would create
a new training-distribution arm, which must be measured separately.

## T7: live mission application - complete

The service now uses the included v7m130 / v19noised / v13m130 checkpoint set,
checks its hashes against current validation, and executes the camera pipeline
through `zero-g-mission run`. Tests reject stale source, changed commands,
changed weights, incomplete rack hold and altered output files.

Both a clean-source profile validation and a normal worker execution passed.
[Application guide and revalidation commands](compute_service_demo.md)

## T13: learned seating transfer

Force-aware seating passes its isolated test at 99.20%, yet scores 24/96 in the
chain against guarded control's 23/96 on the prescribed rack. The bay/controller
factorial changes the comparison. Preserve the guarded controller until a learned
arm wins pooled and on every shared seed under the same load-transfer criterion.
Further training needs a specific, measured handoff-distribution or interface
change. [Current comparison](seating_controller.md)

## Dynamic mechanism and boundary tests

T16's corrected clearance sweep moves the bay mouth with the guides. T17 asks
whether each criterion predicts its named failure mode. The overall envelope
remains not qualified, and terminal measurements must not be described as
predictions of their own outcome.

Record attitude, lateral offset and velocity at a defined handoff before the
outcome is known. Then rerun paired boundary arms, including nominal, under the
same environment count and rack-retention configuration. Keep judgment-time
values separate. Changes to the driver require fresh source-bound certification.

T4/T5 cover robustness levels and training randomization. T6 covers the missing
capture and extraction skill gates. T8 (checkpoint ambiguity), T10 (test
portability), T12 (derived-rack re-certification), T14 (rack transfer), T15
(datum-pair sight lines) and T19 (per-environment IK agreement) have measured
resolutions in NOW.md or the archive. T11 now has current, report-verified footage.

## Before a hardware or publication claim

Use [sim_to_real.md](sim_to_real.md) for fixture experiments and
[paper_position.md](paper_position.md) for research framing. Audit old manuscript
plans against current evidence before using them: the earlier camera milestone,
legacy supported-settle result, and isolated skill certificates are different
claims. No submission date or hardware readiness is implied by this backlog.
