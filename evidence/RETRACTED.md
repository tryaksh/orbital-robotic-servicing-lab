# Retracted and superseded reports

`evidence/` keeps everything, including results that turned out to be wrong. That
is deliberate — a project that deletes its mistakes cannot show its reasoning —
but it means a file here is **not** automatically a current number, and five of
them have been quoted after they stopped being true.

**Read this before quoting any figure from `evidence/`.** If a report is listed
below, the number in it describes a system that has since changed. Every one was
a *good measurement* when it was taken; what moved was the code underneath it.

Check currency mechanically rather than by memory:

```bash
python scripts/check_criterion_currency.py
```

That compares each report the handover cites against the last commit to the files
that can define its criterion. `scripts/check_evidence_currency.py` answers the
other half — whether a report describes the checkpoint a run actually loaded.

| Retracted report | Claimed | Why it is wrong | Use instead |
| --- | ---: | --- | --- |
| `grapple_extract_v8_certification.json` | 68.36% extraction | Certified an hour before the settled-enough velocity limits were derived; **none** of its 6,156 counted successes satisfies the limit now in force | `grapple_extract_v14reset_certification.json` — 99.02% |
| `grapple_extract_v9_certification.json` | 67.55% extraction | Same defect | `grapple_extract_v14reset_certification.json` |
| `grapple_extract_certification.json` | 10.09% extraction | Same defect | `grapple_extract_v14reset_certification.json` |
| `workflow_remove_certification.json` and the other pre-2026-08-15 removal runs | 14.06% removal | Same defect: certified before the velocity limits were derived | `workflow_remove_retain_certification.json` — 98.78% |
| `workflow_install_final_certification.json` | 84.38% installation | Describes the *same two policies* by checkpoint hash, but was certified 8.5 h before commit `ffac648` raised the capture phase's budget from 6 s to 10 s | `workflow_install_promoted_certification.json` — 89.41%, and the later clock/retain re-run above it |
| `workflow_install_v6insert_certification.json` | 86.28% installation | Same defect, same commit | as above |
| `vision_workflow_camera_twoslot_certification.json`, the 2026-08-17 run | 65.10% camera; the gate failed by 23.6 points | **One of its three seeds does not reproduce.** Seed 5070 recorded 25.00%; re-run on 2026-08-18 with the identical task, the identical three checkpoints by SHA-256, the identical pose head, 64 environments and 192 episodes, it scores **80.73%**. The other two seeds move within sampling noise, −4.17 and +1.57. The pose head is *best* on the collapsing seed — 2.52 mm mean against 2.65 and 2.53 — and the failures were 142 capture-budget overruns, not the insertion tail the write-up blamed | the re-certification of 2026-08-18, in the same file. The superseded reasoning is kept in `docs/status.md` |
| the 96.10% capture figure, formerly in `grapple_grasp_v5_certification.json` | 96.10% capture | Certified 9.4 h before `ffac648` tightened `capture_success_mask` from a 20 mm grip tolerance to 10 mm. Re-reading its own episodes could only bound it **between 43% and 96%**, because the criterion is the termination: an episode that ended at 15 mm under the old rule would not have ended at all under the new one | **re-measured 2026-08-17: 88.78% pooled and 79.22% in the worst stage, so it FAILS its 95% gate.** The file now holds that run; the number above exists only here. Both bounds were wrong — the lower far too pessimistic, the upper the stale figure itself |

## Not retracted, but scoped: every `main` number describes one workcell

**On branch `industrial-relocation` this is the first thing to know about every
report above and below.** `GRAPPLE_ROBOT_ROOT_POS` moved from (−0.45, 0, 0.15) to
(−0.65, 0, 0.15), and every calibrated spawn pose was re-solved against it. So on
this branch:

| | |
| --- | --- |
| `grapple_grasp_v5_certification.json` (88.78%) | measured on the old cell |
| `grapple_extract_v14reset_certification.json` (99.02%) | measured on the old cell |
| `grapple_insert_two_slot_certification.json` (98.34% worse bay) | measured on the old cell |
| `workflow_remove_retain_certification.json` (98.78%) | measured on the old cell |
| `workflow_install_clock30retain_certification.json` (96.35%) | measured on the old cell |
| `vision_workflow_*_twoslot_certification.json` | measured on the old cell |

None of these is *wrong*. Each was a correct measurement of the cell it ran in,
and each is still current on `main`, which is where that cell lives. What is no
longer true is that **checking one out and re-running it here reproduces it** —
the code in this tree builds a different workcell, so `check_criterion_currency.py`
flags all six, correctly.

The replacements produced on this branch carry a **`w65`** tag in the filename —
`grapple_grasp_v6w65_certification.json` for capture, and the same pattern for
extraction, the two-bay insert and both chains — plus one name `main` has never
had, for the relocation chain itself. The tag is the workcell: base x at
−0.65. (Filenames are given as a convention rather than as citations here,
because a page that cites a report before it exists is the same defect as one
that cites a report after it stopped being true.)

The old files are kept and not overwritten, deliberately: they are the *before*
half of every comparison this branch makes, and once overwritten the comparison
could not be re-made without rebuilding a workcell that no longer exists in the
tree. `run_relocation.sh` and `certify_workflow.sh` gained a `TAG` for exactly
that reason — the first version of this work would have written the new two-bay
insert number straight over the old one.

**This is a different shape of hazard from the retractions below**, and worth
naming as its own. Those are numbers that stopped being true. These are numbers
that are still true about a system nobody can build from this tree. Neither
currency check catches it, because neither asks *"is the geometry this ran on
still the geometry the code makes"* — the honest answer is a tag in the filename
and this page.

## Reports that are measurements, not certifications

These are not retracted and not promotion evidence. They carry
`evidence_type: simulation_capability_envelope` or are explicitly labelled
gates, and their promotion gate is marked non-applicable.

| Report | What it is |
| --- | --- |
| `insert_chain_handoff_gate.json` | The pre-training gate: does the chained-insert task reproduce the chain's hand-off. It is not a promotion of anything |
| `rigid_grasp_l2_envelope_*.json` | Sweeps deliberately past the trained range |
| `uncertain_insertion_*_envelope.json` | Same |
| `grapple_pin_rated_grip_force.json` | A refuted hypothesis, kept because the refutation is the result |

## A label that was wrong on every grapple-pin report — FIXED 2026-08-18

Not a retraction, and it does not move any number, but it misdescribes all of
them and should be fixed rather than remembered.

Every grapple-pin certification carries `evidence_type:
simulation_capability_envelope`, `out_of_distribution: true`, and
`gate.applies: false` — including the ones this project treats as certifications
and quotes as such. The cause is one line in `play.py`, which decides whether the
slot's lead-in flares are collidable with
`bool(...collision_props.collision_enabled)`. That field is a **tri-state**:
IsaacLab documents `None` as "leave as authored", and the grapple-pin scene leaves
it `None`, so a spawned and enabled collider reports as absent. `train.py` reads
the same field correctly, treating only an explicit `False` as disabled, and the
two therefore disagree about the same scene.

The gate values themselves are computed and reported normally — `passed` is
correct — so no rate here is affected. What is affected is the label on top of it.

**Fixed on 2026-08-18.** `play.py` now reads the tri-state properly, treating only
an explicit `False` as disabled, and gains `--no_lead_in` so the field can
actually take the other value — without it `lead_in_present` could never read
false, and the field it reported was dead.

Re-labelling did **not** need a re-run, and that is the part worth reading. The
archived per-episode rows are the measurement; only the metadata stamped on top of
them was wrong. `scripts/relabel_lead_in.py` corrects the `stress` block inside
200 archived `.npz` files and the 23 reports derived from them, in place, keeping
each report's original `generated_utc` — the measurement is not new and must not
read as new to `check_evidence_currency.py`. Every corrected report carries a
`label_correction` block saying what changed and what did not.

The correction is proven rather than asserted: `--verify` re-aggregates a report
from the corrected rows and diffs it field for field against the in-place patch.
Eight reports reproduce exactly, including `grasp_v5`, which re-derives 88.78%
pooled and 79.22% on the worst stage straight from the raw episodes.

Four reports keep `out_of_distribution: true` and are meant to — the two
rigid-grasp envelopes and the two uncertain-insertion belief sweeps are genuine
stress runs. The relabeller decides from the archived rows rather than from the
report, and refuses to touch a checkpoint that appears in any real sweep.

## The pattern, stated once

Four of the five retractions above have the same shape: **a criterion moved after
a number was measured, and nothing re-ran the number.** Not a bad experiment, not
a bad policy — a good measurement of a system that had since changed. The defence
is not care; it is running `check_criterion_currency.py` at the start of a
session and re-running whatever it flags.

The capture retraction is the first one to be *closed by re-measurement* rather
than by a replacement run of a newer policy, and it closed the wrong way: the
skill fails the gate its stale figure passed. That is the mechanism working.

**The two-bay camera retraction has a different shape and needs its own defence.**
Nothing moved underneath it. The criterion was current, the checkpoints were
current, the code was current, and the number was still wrong — because one run
in nine behaved differently and nothing re-ran it. `check_criterion_currency.py`
could not have caught it and neither could `check_evidence_currency.py`.

The only defence against that is replication, and this project had already
written the rule down: *"trusting a single-seed vision sweep"* sits in the
do-not-retry list because one seed once reported a pass that three seeds
overturned. Three seeds then reported a **failure** that a re-run overturns. Three
runs of one configuration are three samples of the configuration, not three
samples of the run. Where a single run can differ by 56 points, the seeds are not
the thing that needs repeating.
