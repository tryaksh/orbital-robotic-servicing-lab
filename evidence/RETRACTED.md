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
| `vision_workflow_camera_twoslot_certification.json`, the 2026-08-17 run | 65.10% camera; the gate failed by 23.6 points | **One of its three seeds does not reproduce.** Seed 5070 recorded 25.00%; re-run on 2026-08-18 with the identical task, the identical three checkpoints by SHA-256, the identical pose head, 64 environments and 192 episodes, it scores **80.73%**. The other two seeds move within sampling noise, −4.17 and +1.57. The pose head is *best* on the collapsing seed — 2.52 mm mean against 2.65 and 2.53 — and the failures were 142 capture-budget overruns, not the insertion tail the write-up blamed | the re-certification of 2026-08-18, in the same file. The superseded reasoning is kept in `docs/archive/` |
| the 96.10% capture figure, formerly in `grapple_grasp_v5_certification.json` | 96.10% capture | Certified 9.4 h before `ffac648` tightened `capture_success_mask` from a 20 mm grip tolerance to 10 mm. Re-reading its own episodes could only bound it **between 43% and 96%**, because the criterion is the termination: an episode that ended at 15 mm under the old rule would not have ended at all under the new one | **re-measured 2026-08-17: 88.78% pooled and 79.22% in the worst stage, so it FAILS its 95% gate.** The file now holds that run; the number above exists only here. Both bounds were wrong — the lower far too pessimistic, the upper the stale figure itself |

| `insertion_conditioned_controller_v1.json` | guarded 5.15%, v24 2.96% pooled | Its 27 reset-station pairs disabled the insertion task's fixed-to-compliant form lock, so those arms did not reproduce the v24 reset distribution. The three real chain-handoff pairs remain valid and were isolated without alteration. | `insertion_conditioned_controller_v3.json`; use `insertion_chain_handoff_controller_v1.json` only for the preserved handoff-only subset |
| `fiducial_rgbd_service_plate.json` | passing RGB-D fiducial qualification | It rendered a tilted plate floating 90 mm above the current module. The tag is now a physically flush top-face datum, so that visibility result does not describe the deployed geometry. | `fiducial_rgbd_flush_v2_seed283.json`, which fails the current geometry at 43.27% critical-bay detection |
| `full_chain_rgbd_service_seed4070.json` | one successful RGB-D relocation | It uses the invalid floating-plate evidence, older checkpoints, and the retired payload-stage shuttle rather than the robot-carried chain. | `rgbd_strict_capture_gate_v2_seed5070.json` is the current negative robot-carried RGB-D run; no successful replacement exists yet |

## Retracted claims that were never published as a rate

Not every wrong claim is a number in a report. This one was a *diagnosis*, made
and withdrawn on the same day, and it is recorded because it was committed,
pushed, and written into the README before it was checked.

**"The insert skill is wedged at 84.6 mrad against a channel that admits
20.5 mrad."** — retracted 2026-08-25.

The 20.5 mrad is `SERVICE_DELIVERED_ATTITUDE_RAD`, and its own definition says
what it is: the attitude a carried module was measured to *settle* at inside the
destination channel, after the lead-ins have worked on it. The same comment
records the arm delivering **63 mrad** at that pose. So it was never the angle at
which entry becomes impossible, and "four times past the limit at which the
module can enter at all" was not a statement about a limit.

The second half of the claim — that the objective had been ranking a fatal
attitude below a survivable offset, because it normalised orientation by 0.15 rad
— does not hold either. 0.15 rad is calibrated about right against the real
success tolerance, `INSERTION_ORIENTATION_TOLERANCE_RAD` = 52.4 mrad: at
tolerance the angular half costs 0.031 against the lateral half's 0.063, and at
the *observed* errors lateral is 2.8× its tolerance while orientation is 1.6×
its own. Lateral is the larger violation, so the weighting was not the defect.

The corrected scale was trained for 400 epochs before the error was found.
**It moved the measured orientation from 84.61 mrad to 84.58 mrad**, and the
change is reverted.

**What survives, and it is worth more than the claim was.** Three objectives now
sit beside each other in `evidence/insert_attitude_diagnosis.json` — the baseline
time cost, a 4× time cost trained to convergence, and a 7× orientation penalty —
and all three land within 0.4 mrad:

| Objective | Orientation | Inside the 52.4 mrad tolerance |
| --- | ---: | ---: |
| baseline time cost | 84.26 mrad | 2.3% |
| 4× time cost, converged | 84.61 mrad | 2.7% |
| 7× orientation penalty | 84.58 mrad | 3.9% |

An angle that does not move when the objective is changed three ways is not the
objective's to give. Two flat pads on a pin cannot resist a moment about the
closing axis — this project's central measurement — and the chain reaches
46 mrad at the same seating phase only because it carries the module on a form
lock. The blocker is the load path: `docs/NEXT_WORK.md` **T9**, not T2.

**How it got published.** The constant was read from its name rather than from
its definition, and the diagnosis was written before the arithmetic against the
success tolerance was done. The check that would have caught it is the one this
repository already states as a rule — derive the number from the parts, and say
which part — and it was skipped because the name sounded like the answer.

---

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

---

## 2026-08-23 — the module changed shape, and four reports describe the old one

`BLADE_SIZE` went from 450 x 160 x 35 mm to **450 x 130 x 20 mm** and the
destination bay's seated plane from 0.75 m to a derived **0.676 m**. Both are
requirements rather than tunings and both are argued in `docs/archive/next_session_handoff.md`.
Every report below was a good measurement of the module and rack that existed
when it was taken.

| Report | What in it stopped being true |
| --- | --- |
| `fiducial_rgbd_service_plate.json` | The service plate moved from a tilted stalk 100 mm above the module's centre to flush on its top face, and shrank from a 100 mm tag in a 180 mm quiet zone to 90 in 120. Detection rate, position p95 and orientation p95 all describe the old plate at the old incidence angle. |
| `robot_carried_seating_sweep.json` | Every row is `2c/L` for the 450 x 160 x 35 mm module. The law still holds and `scripts/check_workcell_geometry.py` still checks the sweep against it, using `SWEPT_MODULE_SIZE_M` rather than the current size — deliberately, so a preserved measurement is checked against the rack it was taken in. |
| `robot_carried_rgbd_seed6070.json` | The one full camera-driven run, on the old module, the old plate and the old seated plane. Its 161 mm axial shortfall is not the current chain's. |
| `full_chain_state_16_report.json` | Same, for the state batch. |

Superseded by `evidence/robot_carried_full_chain_seated.json`, which is one seed
and one environment and is labelled as such.

### A caveat on `robot_carried_full_chain_seated.json`

Its `learned_phases` field lists `insert` and its
`loaded_but_not_executed_policies` is empty. **Both are wrong.** The insertion in
that run was performed by `_step_guarded_insert`, a scripted guarded advance; the
learned insert policy was loaded, hashed and never asked for an action. The label
branched on the payload-shuttle flag rather than on the controller that runs, and
is fixed in `scripts/run_workflow_demo.py` and pinned by
`tests/test_robot_carried_contract.py`. Regenerating the file was attempted and
the re-run did not reach the transit-to-insert handoff, so the corrected label is
in the code and not yet in this artifact. Read the run as: capture and extract
learned, seat and transit and insertion scripted.

### A second caveat on `robot_carried_full_chain_seated.json`: it never left transit

Its own `reached_phase` is `"transit"`. So the seating conditions in it —
including the `axial_depth: true` that the handover and `CLAUDE.md` both read as
"the module now reaches full seated depth" — were evaluated on a module that the
**transit's own last leg** drove to the seated plane, with the chain still in the
transit phase and the insertion phase never entered.

That is the overshoot the handover flagged as a reproducibility problem, read
from the other side. Leg 0's axial target is the staging pose at 0.5779 m; the
module in that run finished at 0.6763, which is 98 mm past it and 0.3 mm short of
the seated plane at 0.6760. The hand-off test was two-sided on depth, so a module
that had gone *through* the staging pose could not satisfy it, the leg was ended
by its own timeout, the timeout re-stamped the leg's entry step, and the run sat
there until the step budget expired — while the module happened to be sitting at
the right depth.

So the numbers in that file are a correct measurement of **where the module got
to**. They are not a measurement of an insertion, and the phase that is supposed
to perform the insertion contributed nothing to them. Two of its three
"passes" — depth and orientation — are properties of a leg overshooting, and the
one "failure", 4.2 mm of lateral error, is the transit's delivery accuracy rather
than the seating's.

Read the run as: capture and extract learned; seat and transit scripted; the
insertion never ran.

What replaced it, and why the replacement looks worse:

| | `robot_carried_full_chain_seated.json` | the solved-IK run, same seed |
| --- | ---: | ---: |
| Reached phase | transit | insert |
| Destination squaring leg | forced by timeout, 11.4 mrad | **met its gate** |
| Insertion approach leg | forced by timeout, 13.5 mrad | met its gate |
| Lateral alignment | 4.19 mm — **fail** | 1.62 mm — pass |
| Axial depth | 0.3 mm short — pass | 52.9 mm short — fail |
| Orientation | 13.6 mrad — pass | 52.4 mrad — fail |

The transit got strictly better and the reported outcome got worse, because the
chain now reaches a phase it had never reached. The module is handed to the
guarded advance at 52.4 mrad — inside the 52.36 mrad gate the hand-off used, and
about twice what the destination channel admits over the length it is engaged
over — and it wedges. The guarded advance's stall detector then holds the target
rather than pushing, which is the behaviour it was built for. The gate is the
defect and it is fixed; see `docs/archive/next_session_handoff.md`.

**And the comparison above is itself a snapshot.** It was taken mid-session, with
the solved-IK transit in place and the seating stroke still being performed by
the transit's last leg. The chain closed later the same session, once the
insertion phase was given the stroke and its own action scale:
`evidence/robot_carried_full_chain_complete.json` reaches `done` with all seven
seating conditions held through the settle. Read the table as "what the transit
fix did", not as the state of the chain.

## 2026-08-24 — the skill certifications describe the previous module

`BLADE_SIZE` is 450 x 130 x 20 mm. These were taken on the 450 x 160 x 35 mm one,
which is a third heavier with a different inertia tensor, in tasks whose entire
difficulty is a contact transient on a free-floating payload.

| Report | Claimed | Measured on the current module |
| --- | ---: | ---: |
| `grapple_grasp_v6w65_certification.json` | 94.46% | **63.43%** pooled, 42.71% worst stage |
| `grapple_extract_v16w65_certification.json` | 94.89% | **72.92%** pooled, 63.54% worst stage |

Neither is a bad measurement; each is a correct measurement of a module that is
no longer in the scene. The controls that establish the second column are
`grapple_grasp_v6w65_on_current_module_control.json` and
`grapple_extract_v16w65_on_current_module_control.json` — one evaluation seed and
384 episodes per stage, which is a control and not a certification, and labelled
so in the files.

Superseded by `grapple_grasp_v7m130_certification.json` (86.64%) and
`grapple_extract_v17m130_certification.json` (74.27%), both fine-tuned on the
current module and both still below this project's 95% gate.

`grapple_insert_two_slot_w65_certification.json` (10.50%) is in the same
position, and its successor `grapple_insert_v13m130_certification.json` is worse
at 0.00% — from a fine-tune whose reward was still climbing when it was stopped,
so read it as unconverged rather than as a refutation of the approach. A longer
fine-tune, `v14m130long`, was started and abandoned uncertified when the decision
was taken to retrain from scratch instead; nothing may be quoted from it.

## 2026-08-24 — "the robot at the end of its own reach into the rack"

`assets.py` explained the destination bay's seated plane, 0.676 m, by saying the
chain stops at 0.6763 because that is as far into the rack as this arm reaches.
The reach was never checked.

It is not the reason. `zero_g_blade_swap.arm_kinematics` — the same closed-form
solver the transit legs are commanded from, validated against every
configuration `evidence/workcell_reach_solution.json` recorded from the
simulator — solves the head-on tool pose for a module centre of **0.75 m** with a
position residual of 0.0001 mm, a realised DLS authority of 0.9998 and a
smallest singular value of 0.327. Every station from 0.147 to 0.75 solves the
same way; `evidence/insert_reset_bank.json` records nine of them.

**The number is unaffected.** `SERVICE_DESTINATION_SEATED_X` is derived from
`service_latch.release_before_blade_centre_x_m`: past that depth the engaged jaw
would enter the slot mouth, so the lock has to be released there whatever the
arm could still do. The retraction is of the mechanism, not of the plane.

## 2026-08-24 — `workflow_robot_carried_m130pin_check_certification.json` is 0.00% and is not a chain result

That report says the robot-carried relocation succeeded in 0 of 32 episodes. It
is kept because it is the measurement that found the defect, and it must not be
read as a rate for this chain.

`GUIDE_CENTER_OFFSET_Y` was derived and the side rails moved inboard 3.061 mm.
Two other pieces of the rack were positioned *from* that constant and were not
re-derived with it: `_FLARE_CENTER_Y`, an authored literal placed so the lateral
lead-in's inner face met the rail face at the mouth, and `_RAMP_SURFACE_OFFSET`,
which is the difference between those two and therefore moved the vertical
lead-in 3.061 mm the *other* way. The run above is that rack.

The signature is unusual and worth keeping: the module still arrived 1.0 mm from
the seated plane and 47.1 mrad square, and every episode still ran its clock out,
because the failing condition was **4.04 mm of lateral against a 2.5 mm
tolerance** — against 1.85 mm on the same chain the day before. A lead-in that
does not track its own channel surface is a step, not a lead-in.

`evidence/workflow_robot_carried_leadin_check_certification.json` is the same
seed with `_FLARE_CENTER_Y` derived from the rail face: **93.75%**.
`tests/test_workcell_geometry.py::test_the_lead_ins_move_with_the_rails_they_continue`
holds the relationship so the next channel change cannot break it silently.

## 2026-08-25 — "the module rests corner-to-corner against the channel walls" was the wrong surface

**Corrected, not retracted: the conclusion held and the mechanism named in it did
not.** T9 read the insert skill's terminal-attitude band -- 56.03 to 56.92 mrad
over the 91 episodes that reached seated depth -- as the module "resting
corner-to-corner against the channel walls at the largest angle the clearance
permits", and identified that angle as `2c/L` on `GUIDE_CENTER_OFFSET_Y`'s
12.689 mm, which is 56.40 mrad.

Two things were wrong with the mechanism.

**The channel walls were nowhere near it.** The destination bay is relieved by
4.6125 mm per side, so its walls admit 76.90 mrad of yaw and 56.06 mrad of pitch
-- and the runs in question were taken through `play.py --latch_enabled`, which
applied the relief a *second* time and opened them to 97.40 and 76.56 mrad. A
module at 56 mrad in that channel is not touching its walls. What holds it is the
lead-in throat at the mouth, which is authored from the rail face and
deliberately does not move with the relief.

**The band is a floor, not a ceiling.** A wedge is an upper limit and would show
values crowding up to it; the measured p5 is 56.035 mrad and the minimum is
56.033, with a tail running to 86.5. That is a surface the module cannot get
*past*, which is what a throat is.

The conclusion survives both corrections, because the throat is set by the same
constant: `GUIDE_CENTER_OFFSET_Y` places the guides and the flares are derived
from the rail face, so narrowing it narrows the throat. Tested rather than
argued, same checkpoint and seed and episode count, in
`evidence/insert_attitude_wall_moved.json`: 12.689 -> 11.065 mm moves the floor
from 56.03 mrad to 45.75 mrad.

**And the relief was being applied twice by half the callers**, which is its own
finding rather than a detail of this one.
`evidence/destination_channel_geometry.json` measures the destination bay out of
the built configuration for each entry point: the chain and the skill
certification built a 17.30 x 12.61 mm channel while skill *training* and every
lock-on diagnostic built a 21.91 x 17.23 mm one. **The insert skill was trained
in a rack 4.6 mm per side wider than the rack it was then certified in, on both
axes.** Every insert number taken before 2026-08-25 carries that, including the
v23lock row in `insert_attitude_diagnosis.json`; the direction of the resulting
bias is not known, and the retrain and re-certification below it are on a single
rack that `scripts/check_destination_channel.py` verifies.
