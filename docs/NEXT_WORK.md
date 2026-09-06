# Next work

Bounded tasks for the correction-aware handoff pivot. The question, what is
settled, and the day-14 decision are in [`CHARTER.md`](CHARTER.md); read that
first. Current state is [`NOW.md`](NOW.md).

The pre-pivot backlog — thirty-one tasks, most of them closed, refuted or
absorbed — is
[`archive/NEXT_WORK_pre_pivot_2026-09-05.md`](archive/NEXT_WORK_pre_pivot_2026-09-05.md).
It was retired on 2026-09-05, not deleted: three of its tasks became the pivot's
evidence and the rest record why lines were abandoned. Nothing below restates it.

**What carried over, and why only these.** A task survives the pivot only if the
day-14 decision needs it. T0 (source provenance) survives because a result whose
code cannot be recovered cannot be published. T21 (the delivered attitude)
survives as *done* and load-bearing: the 46 mrad premise is measured at 8.19 mrad
median, and every geometry number derived from the old constant is suspect. The
rest are archived.

---

## H0 — Release the module when it is not moving (**answered 2026-09-05: fit the pawls**)

Superseded by its own answer, and kept because the route to it is the point.

The settle traces said the module is placed correctly and drifts out after
release, with drift tracking release velocity at rho = +0.84. The proposed fix
was to condition the release on velocity. The actual cause was one level down:
**the cohort had no retention mechanism at all**. Fitting it —
`--rack_retention`, same point, same three seeds, nothing else changed — takes
the nominal point from **110/192 to 187/192**, 77 gained and 0 lost,
p = 1.3e-23, with drift going to zero and release velocity to 0.000 mm/s.

No velocity-conditioned release is needed. The 77 gained is exactly the count of
`missed_the_terminal_gate` failures, so the fixture accounts for all of them.

## R1 — Re-run the boundary family with the pawls fitted (**now the priority**)

Thirty `robustness64*` cohorts, `relief0*` and `envcount*` were run with
`destination_rack_retention.enabled = false`. Every boundary verdict this
project has published rests on them: the module-section points, the clearance
axis, the "not qualified" decision, and the rack-prescription refutation at
110/192 against 64/192.

**None of those numbers is wrong.** Each measured a bay with no pawls. They
cannot be quoted as properties of the workcell that was designed until the same
points are run with the fixture that workcell has.

**The prescription arm is done, 2026-09-05.** With the pawls fitted it costs 12
episodes in 192 — 0 gained, 12 lost, p = 4.9e-04 — against 46 without them. The
refutation survives at a quarter of its size and its mechanism is named: 12
episodes that **never seat**, stopping 32.0-32.9 mm off centre with
`predicate_fired = 0` and the pawls never in the path. That is
`jammed_in_the_bay`, whose count at the shipped relief is zero, and it is the
mode the library's **entry** criterion predicts.
See `CHARTER.md`. **The mechanism is now named and it is angular, not lateral.**
Every insertion swings to ~76 mrad in both configurations; the relief does not
change that median at all, it bounds the tail. Shipped relief never exceeds
80.01 mrad in 187 episodes; all 12 wedges exceed 83.6 mrad and stop at
x = 0.220 m against a seated plane at 0.676 m. `evidence/entry_swing_v1.json`.

**R1a — derive a peak-entry-swing bound.** The geometry checker derives what a
seated module may rest at and says nothing about what the stroke passes through,
which is the quantity that actually governs here. Deriving it belongs in
`servicing_design.py` beside the existing bounds, validated against these
traces. No simulator; the traces exist.

**Cost of the rest.** The nominal point is 23 minutes for three seeds. The
remaining sweep points are roughly eight, so under three hours.

**Read every point paired** against the retention-absent arm that already
exists. Same seeds, same checkpoints, one flag.

## R1b — The remaining sweep points (**clearance axis needs seeds first**)

`section_120x16` and `section_140x26` are done and reported in `CHARTER.md`.
What is left of the sweep:

* `rack_lat_6mm`, `rack_lat_16mm` — **the clearance axis, and the one that most
  needs re-running**. A 2x2 was started on 2026-09-05 and **stopped seven runs
  of twelve in**, leaving `artifacts/clearance_factorial/` partial: the
  retention-absent arm at three seeds, the fitted arm at one, so nothing in it
  is paired. Finish from there rather than pairing against `robustness64*` —
  every run in it states `--rack_clearance_scope channel` explicitly, which the
  historical cohorts do not record. Five runs, about 45 minutes.
* `base_x_-0.70`, `base_y_+10mm`, `mass_20kg`, `mass_40kg` — lower priority. No
  published verdict rests on them.

**Read the delivery column first on every one.** The section axis turned out to
be entirely a capture failure once the fixture was fitted, and a clearance point
may be the same. `scripts/qualify_handoff.py` splits it.

**A process note that cost two hours of GPU throughput.** That supervisor was
believed stopped on 2026-09-05 and was not: `pkill -f
supervise_clearance_factorial` did not kill it, it survived as PID 1870, and it
kept launching cells alongside `supervise_jam_mechanism.sh` from 17:47 to 18:50.
Both campaigns shared one GPU for that window and runs took about 11 minutes
where the same work alone takes 8. No episode is wrong — contention costs wall
clock, not physics — but **any timing read from either campaign's logs in that
window is not a clean measurement**, and a run described at the time as
time-boxed was quietly executing at two-thirds speed.

The rule this repository already has for runs applies to kills: verify by the
artifact, not by the claim. A kill is confirmed by matching each surviving
process to its `--episode_metrics` path, not by the absence of an error from
`pkill`.

## R1c — Why the module never arrives (**capture *and* extract**)

The finding the section axis actually produced. With the pawls fitted, seating
is perfect at every cross-section — precision given delivery 1.000 — and the
whole axis is delivery: 0.974 at nominal, 0.911 at 140x26, **0.792 at 120x16**.
A 120 x 16 mm module is captured four times in five.

**It is not one failure mode and calling it "capture" was wrong.** The phase
breakdown: nominal loses 3 to capture and 2 to extract; 120 x 16 mm loses 33 to
capture and 7 to extract; 140 x 26 mm inverts to 1 and 16. A thinner module is
hard to grasp, a thicker one hard to pull clear, and no clearance or seating
work moves either. This is where the chain's remaining headroom is: nominal
itself delivers 0.974.

**Start from the five nominal failures.** They are identical across every arm
run today -- 384.6, 216.5, 364.9, 1212.5 and 833.7 mm -- so they are
deterministic and reproducible, and the capture traces for them already exist.

## R2 — Re-run the gravity sweep with the pawls fitted

Sixteen cohorts, none with retention. The published reading — 14/48 in orbit and
0/48 at lunar, Mars and Earth, with the failure mode inverting — was measured on
a bay that holds nothing. A module released free under gravity with no pawls
falls out, so the sweep has not yet tested the interface it claims to test.

**Falsified if** the rates do not move: that would mean gravity defeats the
pawls too, which is a real and much stronger interface result.

## R3 — Re-run the `install` second workflow with the pawls fitted

Two cohorts, no retention, and only one of three seeds ever completed. Its 0/16
and the "needs a 10.41 mm tolerance" figure derived from it are both
retention-absent numbers. Re-run all three seeds with `--rack_retention` and
`TRACE=1`, and verify by the artifact rather than the log line -- seeds 5070 and
6070 died in Isaac startup when the session closed on 2026-09-05 and
`supervise_secondtask.sh` printed `exit=0` for one of them anyway.

**Why it still matters more than a public second task.** Same frozen controller,
same bay, a different approach path. That is a two-configuration result on
machinery that exists, where a Franka port would cost a baseline training
campaign to reach the same point.

## H1 — Join pre-handoff state to outcomes that vary (**done 2026-09-05**)

The traced nominal cohort landed at 13:29, three seeds, 23 minutes of GPU.
**The control passes exactly**: 35/64, 36/64, 39/64 — 110/192, the published
number to the episode — with all 192 episodes agreeing individually and a
maximum per-episode residual difference of 0.000000 mm. The trace is
non-perturbing and 187 of 192 episodes carry a pre-handoff row.
`evidence/trace_non_perturbing_v1.json`.

## H2 — Fit the pre-handoff predictor (**falsified 2026-09-05**)

Largest of 24 feature correlations with the terminal residual: **0.135**,
against a null whose largest-of-24 at n = 187 has a median of 0.159. Held out by
seed, the model's Brier is 0.2407 against the base rate's 0.2427 — a gain of
+0.002 where the declared margin was +0.10.
`evidence/pre_handoff_predictability_v1.json`, verdict `falsifies_h2`.

**Do not read this as "the fixture erases incoming error".** The delivered
distribution is narrow — 2.131 mm of lateral spread at handoff, 0.124 mm of
latch relative position error — against a 3.931 mm terminal residual spread. The
variance that decides the outcome is generated *during* the contact interval,
not inherited. A correction model has nothing to condition on because the
conditioning variable barely moves.

## H2b — Widen the handoff distribution on purpose (**the one experiment worth running**)

The single recommended next experiment, and the only way to learn whether a
correction model could work at all. Inject lateral and attitude offsets at the
transit-to-insert boundary, well beyond the 2 mm the transit naturally delivers,
and find where the fixture stops absorbing them.

**Why it is the right next step.** H2 did not show that incoming error is
uninformative; it showed that this cohort never varied it. The boundary the
whole pivot is about has never been sampled.

**Machinery that already exists.** `scripts/solve_insert_reset_bank.py` solves
paired arm and module poses along the seating stroke in closed form, and
`play.py --legacy_unbounded_reset` widens a reset distribution.

**Falsified if** the pass rate stays flat across the full injected range — which
would mean the fixture's acceptance region is wider than anything reachable, and
the qualification question is answered trivially — or if it collapses the moment
any offset is injected, which would mean the delivered distribution is already at
the boundary and the margin is zero.

**Cost.** The traced cohort was 23 minutes for 192 episodes. A five-point
injection sweep at three seeds is under two hours.

## H3 — Transfer the criterion curve between configurations

The criterion curve is the advantage that survived day one. Test whether it
transfers: fit on two of `nominal`, `section_120x16`, `section_140x26`, predict
the third's pass rate at a criterion it was not run at.

**Done when** the prediction and its interval are compared against the
third cohort's measured rate, with the interval declared before the third is
read.

**Falsified if** the predicted rate misses by more than ± 0.10 absolute or the
interval fails to cover.

This needs no new GPU. The three cohorts exist.

## H5 — Why a run exits 0 having written nothing (**diagnosed, mostly closed**)

`supervise_secondtask.sh` reported `exit=0` for two runs that wrote no episodes.
Audited 2026-09-05: **the `rc=$?` rule is intact and there is no hole in it.**
The script captures the status correctly and, more importantly, builds its
aggregation list from `[ -f "${out}.npz" ]` and refuses to aggregate below three
seeds. No certification was ever at risk. The log was the only thing that lied,
and it now names a missing artifact instead of printing a bare `exit=0`.

What remains open is the smaller and stranger half: **`run_workflow_demo.py`
exited 0 after 36 seconds having written no `.npz`.** Both logs truncate at the
same byte, 13.7 s into scene setup, and both carry `Disabling key-value database
because another kit process is locking it` — a second Isaac process was running.
The session closed at 12:33 and took them with it, so the likely story is a kill
that surfaced as a zero status through Git-Bash. That is a guess.

**Done when** either the driver is shown to return non-zero on a scene-setup
abort, or the story above is confirmed and the exemption written down. Low
priority: the artifact check above already prevents it from producing a wrong
number.

## T0 — Source provenance for results that remain in scope (carried over)

Ten older reports were produced by uncommitted code and cannot be reproduced.
The pivot narrows what must be recovered: only the reports the day-14 decision
cites. Everything else may stay lost and labelled.

**Done when** every report cited by `CHARTER.md` carries a runtime source
binding that resolves, or is dropped from the citation list.

---

## Two rules this list does not get to bend

**Never widen a tolerance to make a gate pass.** H3 reports what tolerance a
configuration would need. That is an input to the interface specification, not
permission to move `INSERTION_LATERAL_TOLERANCE_M`. The criterion is unchanged
and every rate here is quoted against 2.5 mm.

**Never quote a criterion change and a policy change as one number.**

Compute figures assume the measured machine: RTX 5070 Ti Laptop, 12 GB. A
1024-environment PPO run fits alongside a small evaluation process; two full
training runs do not.
