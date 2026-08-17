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
| the 96.10% capture figure, formerly in `grapple_grasp_v5_certification.json` | 96.10% capture | Certified 9.4 h before `ffac648` tightened `capture_success_mask` from a 20 mm grip tolerance to 10 mm. Re-reading its own episodes could only bound it **between 43% and 96%**, because the criterion is the termination: an episode that ended at 15 mm under the old rule would not have ended at all under the new one | **re-measured 2026-08-17: 88.78% pooled and 79.22% in the worst stage, so it FAILS its 95% gate.** The file now holds that run; the number above exists only here. Both bounds were wrong — the lower far too pessimistic, the upper the stale figure itself |

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

## A label that is wrong on every grapple-pin report

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
The fix is to read the runtime collision state from the stage the way `train.py`
does, rather than infer it from a config default, and it needs a re-run of the
affected reports to re-label them.

## The pattern, stated once

Four of the five retractions above have the same shape: **a criterion moved after
a number was measured, and nothing re-ran the number.** Not a bad experiment, not
a bad policy — a good measurement of a system that had since changed. The defence
is not care; it is running `check_criterion_currency.py` at the start of a
session and re-running whatever it flags.

The capture retraction is the first one to be *closed by re-measurement* rather
than by a replacement run of a newer policy, and it closed the wrong way: the
skill fails the gate its stale figure passed. That is the mechanism working.
