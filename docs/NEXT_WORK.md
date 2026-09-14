# Next work

Priority follows the measured results in [NOW.md](NOW.md). The fixed-base v3
profile passes its recorded mission checks. A successful development episode
does not close reliability or the broader serviceability envelope.
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

Of 45 source-bound reports, **28 cannot be fully recovered**: 27 inherited
provenance gaps and the preserved
[failed axial-only extraction probe](../evidence/workflow_stability_axial_finish_probe_seed6070.json),
whose exact mixed-line-ending runtime bytes remain unavailable. The other 17
recover through reachable git revisions or exact committed source snapshots.
The current application has byte-stable source and shipped mission weights;
that does not repair the missing historical bytes.

```powershell
python scripts/check_source_provenance.py --depth 200
```

Select which claims remain in scope, recover matching files across all refs,
and rerun any still-unrecoverable certification on a clean commit. Store a new
versioned report rather than overwriting the old one. Completion means every
selected claim has reachable source, weights, configuration, seeds and artifacts.
Cost: minutes for the audit; hours per selected GPU certification.

## T1: stronger camera-driven evaluation

The previous v2 combined camera recipe scored 17/24 and missed the 95% gate.
The fixed-base v3 development recording and profile validation are separate
single-episode demonstrations. Their stable lighting and single environment
differ from the randomized, eight-environment regression comparison.
Seed 6070 informed the new controller's development; reusing it beside 4070 and
5070 does not create an independent controller-development holdout.

The shortened corrected comparison completed only seed 4070: 6/8 versus 5/8
legacy, with no paired losses. Seed 5070 was interrupted and 6070 was not run.
Both 4070 and 6070 have now informed development; the remaining validation
requires fresh, declared conditions rather than treating this partial check
as a complete 24-condition result or a closed 95% gate.

After the recovered factorial and fixed-base comparison audits, preregister a
larger set of new held-out seeds,
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

## T7: live mission application

The v3 service uses the included v7m130 / v19noised / v13m130 checkpoint set,
checks its hashes against current validation, and executes the camera pipeline
through `zero-g-mission run`. Tests reject stale source, changed commands,
changed weights, incomplete rack hold and altered output files.

The initial candidate profile validation passed all nine checks on clean source
`88235d8`. The earlier v2 profile and worker validations remain preserved.
The [fresh-checkout profile validation](../evidence/live_service_stability_validation_seed6070.json)
passed all nine checks on `4a433e0`, with exact current command and raw source
bindings. It includes the guarded-extraction observation-gap hold correction.
The [isolated normal v3 worker](../evidence/live_service_stability_application_seed6070.json)
passed all nine checks and all five output hashes on clean source `e501500`.
[Application guide and revalidation commands](compute_service_demo.md)

## T23: supported-settling command ownership

The absolute-IK insertion path computes corrections during the first supported
settling window, but the final joint-override mask drops DONE environments.
The accepted biased command therefore gives way to zero-action relative IK.
In the selected V5 recording, the transition at step 1250 produced a maximum
joint-target step of 3.868 mrad, 0.084 mm wrist translation on that step and
0.070 mm net wrist motion through the supported interval. Both supports released
at step 1271 and all nine mission checks passed. This is a measured ownership
transition, not a demonstrated failure of the completed mission.

Retain the validated controller while assessing whether this small effect
matters. A proposed follow-up preserves the last accepted absolute target and
freezes further trim while the rack already owns the module, then disables that
override when release occurs. It must preserve the open-hand command and the
passive rack-only recheck. Test the actual control methods, then compare fresh
physical runs under identical criteria before changing the deployed recipe.
The existing paired cohort remains evidence for its original source revision.
[Refinement scope](TRANSFER_STABILITY.md)

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
