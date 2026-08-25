# Now

The verified state of this repository. **This file is canonical.** Read it
before quoting a number or changing a constant. Work to do is in
[`NEXT_WORK.md`](NEXT_WORK.md); every measurement is indexed in
[`../evidence/MANIFEST.json`](../evidence/MANIFEST.json).

Last verified: 2026-08-25. Branch: `main`.

Everything here is simulated. Nothing has been run on hardware.

---

## 1. What runs

**The mission this stands in for:** replacing a failed compute module on an
orbiting platform, autonomously, without an astronaut EVA. The whole simulation
runs at `gravity=(0.0, 0.0, 0.0)`, which is not decoration — it is why capture is
learned rather than scripted. Nothing settles, and squeezing a free-floating mass
in zero gravity ejects it before the pads close. The README says why that changes
the problem; the specification is the design output it exists to produce.

One continuous episode, no cuts: a UR10e locates a compute module in a rack bay,
grips it, pulls it clear, carries it to the neighbouring bay, drives it home, and
opens its hand only after every seating condition has held for 0.70 s. The robot
holds the module the whole way — no world constraint, no teleport, no direct
module pose write, no hidden carrier.

| Phase | Controller | Certified separately |
| --- | --- | --- |
| Grasp | RL policy | yes — `grapple_grasp_v7m130_on_derived_rack_certification.json` |
| Seat (0.03 s dwell) | scripted | — |
| Extract | RL policy | yes — `grapple_extract_v18pin_certification.json` |
| Transit, 5 legs | solved inverse kinematics, actuator targets | — |
| Insert | guarded advance on the deployed RGB-D estimate | the learned alternative is measured against it, §5 |
| Back off | scripted, after the settled re-check | — |

A phase may not be called learned unless a policy produced the actions. The
report keys that label on the controller that stepped, never on a flag.
`docs/service_interface_spec.md` §10 states the split as a requirement.

## 2. The numbers

### Chain

**97.92% pooled** — 94 of 96 episodes, 32 environments on each of three held-out
seeds, Wilson 95% **[92.7%, 99.4%]**. Per seed: 93.75%, 100%, 100%. The gate is
95% pooled and 95% worst-case; both pass. Tool-to-module drift through the carry
is 0.9 mm and 2.5 mrad at the median, 2.3 mm and 6.3 mrad at worst.

`evidence/workflow_robot_carried_m130pin_guarded_c11065_certification.json`

**Certified twice, at two rack clearances, and the two are identical.** The
destination channel's throat was re-derived on 2026-08-25 and moved from
12.689 mm to 11.065 mm per side, which moves a published number, so the chain was
re-run under it before anything was allowed to depend on it:

| | throat 12.689 mm | throat 11.065 mm |
| --- | ---: | ---: |
| pooled | 97.92% (94/96) | **97.92% (94/96)** |
| seed 4070 | 93.75% | 93.75% |
| seed 5070 | 100.00% | 100.00% |
| seed 6070 | 100.00% | 100.00% |

Seed for seed, which is stronger than the pooled agreement on its own. The chain
never used the headroom that moved: it delivers 46 mrad and its form lock holds
the module there, so a throat that accepts 49.18 mrad instead of 56.40 is still
3.2 mrad clear, and the guarded advance's own gates come from
`SLOT_ENTRY_RAMP_CATCH_M` on the *vertical* ramp, which did not move. The
12.689 mm run is preserved as
`evidence/workflow_robot_carried_m130pin_guarded_certification.json`.

The two befores are preserved too, because a before is what makes an after mean
anything: 31.25% (`workflow_robot_carried_relocate_certification.json`), then
96.88% (`workflow_robot_carried_m130_guarded_certification.json`).

> **Provenance: the 11.065 mm run is the first one here that recovers.** Every
> report that records source hashes had been produced on **uncommitted
> working-tree code** — nine of them, across three sessions, none reproducible
> from this repository. This one was launched from a clean commit, and
> `python scripts/check_source_provenance.py --depth 20` reads
> `robot_carried_full_chain_c11065.json` as **RECOVERED** where the other nine are
> LOST. That is `NEXT_WORK.md` **T0**'s first recommendation, and it cost nothing
> but ordering. The nine are still lost and every number that rests on them still
> carries the caveat.

### Skills

Three curriculum stages on each of three held-out seeds, 500 episodes a point.

| | Pooled | Stage 0 (what the chain runs) | Worst stage |
| --- | ---: | ---: | ---: |
| Grasp v7m130, on the derived rack | 85.69% | 99.14% | 78.68% |
| Extract v17m130, the checkpoint replaced | 87.78% | 91.53% | 82.80% |
| Extract v18pin, 2,000 further epochs | 87.75% | 91.08% | 84.08% |

**Neither passes the 95% gate, and the chain exceeding both is not a
contradiction.** The chain is not the product of the skill certifications: those
pool three curriculum stages while the chain runs stage 0, each phase hands over
on the *next* phase's precondition rather than on its own success criterion, and
the guarded seating recovers deliveries a skill certification scores as failures.
Report both. Quote neither alone.

### Perception

Certified separately, on 1,024 rendered frames
(`evidence/fiducial_rgbd_service_plate.json`). The chain's 97.92% runs on the
**state** task, where the module pose comes from the simulator and the guarded
advance's "deployed estimate" is the deployed code path reading ground truth.
The RGB-D chain has been run end to end at one seed
(`evidence/full_chain_rgbd_service_seed4070.json`) and not since the changes that
produced the current rate. **Putting the two numbers side by side without saying
this is the easiest way to overstate what is built.** See `NEXT_WORK.md` T1.

## 3. Checkpoints, and what is not in the clone

`logs/` and `checkpoints/` are **gitignored**. A clone carries every report and
no weights, so every learned number here is readable and not reproducible
without the files below. Nothing in this repository should claim a capability
whose checkpoint is unreachable.

```
logs/rl_games/zero_g_blade_insertion_contact/
  grapple_grasp_l0_seed70_v7m130/nn/last_..._ep_3100_rew_30.262873.pth     capture
  grapple_extract_l0_seed70_v18pin/nn/last_..._ep_12600_rew_172.70488.pth  extraction
  grapple_insert_l0_seed70_v13m130/nn/last_..._ep_8000_rew_-42.01845.pth   loaded, never stepped
checkpoints/module_pose_head*.pth                                          perception
```

The chain runs **two** policies. The insert checkpoint is loaded because the
certification loaded it and the policy-set hash covers it; the seating is the
scripted guarded advance, so it never acts. `--insert_checkpoint` is optional and
`--insert_controller policy` is what makes it act.

Provenance is mechanical, not remembered: `evidence/robot_carried_full_chain_c11065.json`
records all three paths and their SHA-256, the pooled report records the combined
`policy_set_sha256`, and `tests/test_reproduction_path.py` fails if
`scripts/run_robot_carried.sh` stops defaulting to that set. That file is also the
one report here whose *source* bindings recover from git;
`evidence/robot_carried_full_chain_pin.json` is the same record for the 12.689 mm
run and does not.

## 4. Settled — do not re-litigate

Each of these was measured, and re-deriving them costs GPU hours that buy nothing.

- **The mating stroke must be compliant.** Rigid reaches 0.2275 m and 269 mrad
  against compliant's 0.6753 m and 7 of 7 conditions.
  `evidence/robot_carried_rigid_mating_refuted.json`.
- **The rail carries the robot, never the module.** Parked opposite a bay the
  arm's configuration is the one it has at bay 1, so no bay needs a skill another
  bay does not already have. The world-mounted payload shuttle behind
  `--base_rail_on_relocation` is a labelled historical baseline, unreachable from
  the live preset, and `tests/test_robot_carried_contract.py` keeps it out.
- **Do not widen the channel and do not shorten the module.** Both measured, both
  dead ends. Narrowing is a different question and is now derived.
- **Extract's ceiling is the task, not the training budget.** 900 epochs moved it
  1.4 points; 2,000 more moved it 0.0. What moved it — 13 points, on an
  *unchanged* checkpoint — was fixing the grip criterion, deriving the rack
  clearance, and bounding the reset. `evidence/extract_attribution.json` separates
  the four contributions one row at a time.
- **The module's cross-section is what made extraction hard.** The same unchanged
  policy scores 99.02% on the section it was built for and 76.95% on the current
  one, one seed, nothing else different.
- **A tapered pin holds by feeding.** The pads come to rest 12.0 mm along the pin
  from its drawing pose on every loaded pull, measured over 433 extractions in a
  band 0.8 mm wide. The grip criterion is three bounds on the pin's own axes now,
  two of them *tighter* than the ball they replaced.
- **"The robot at the end of its own reach into the rack" was wrong.** The seated
  plane is set by the latch's release interlock, not by reach.
- **A depth-dependent attitude envelope for the guarded advance** was built,
  checked against data in hand, and refuted before it ran. It is attractive enough
  to be reinvented; the reasoning is in the guarded advance's own report under
  `why_not_depth_dependent`.
- **Lead-ins have to move with the rails they continue.** Deriving the guide
  offset moved the rails 3.061 mm inboard while an authored literal stayed put,
  and the chain scored **0.00%** over 32 episodes on that rack. Both lead-ins are
  derived from the rail face now and `tests/test_workcell_geometry.py` holds it.
  See `evidence/RETRACTED.md`.
- **A rack may not hold a module outside its own acceptance criterion.** A rigid
  part fully inside a channel with `c` per side wedges at `2c/L` and cannot be
  squarer, so the channel decides the attitude of a module that merely rests in
  it. At 12.689 mm that was 56.40 mrad against a 52.36 mrad seated criterion, and
  no controller fixes a rack that specifies a part it will then reject.
  `GUIDE_CENTER_OFFSET_Y` is derived from that requirement now, and the chain is
  certified at both clearances.
- **Do not place a derived constant on a bound.** Both values this project has
  used for `GUIDE_CENTER_OFFSET_Y` sat exactly on one: 15.750 mm was outside the
  pads' bound and cost measured grips, and the 12.689 mm that replaced it sat on
  the pads' bound and 4.04 mrad outside the seated criterion. The value is placed
  at the clearance that maximises the smaller of the two margins — the midpoint
  *in attitude* between the 46.0 mrad the transit delivers and the 52.36 mrad a
  seated module is accepted at — which is **11.065 mm**, 3.18 mrad clear on each
  side. Specification §6.5.
- **Configuration that mutates the scene must write absolute poses.**
  `configure_service_destination` added the channel relief as an increment, and
  it is called from `configure_robustness`, which `__post_init__` has already
  run — so `train.py --robustness_level` and `play.py --latch_enabled` each
  applied it twice. **The insert skill was trained in a 21.91 x 17.23 mm channel
  and certified in a 17.30 x 12.61 mm one**, on both axes, for as long as both
  paths have existed. `scripts/check_destination_channel.py` reads the built
  configuration rather than the source and reports the applied relief as a
  multiple; `evidence/destination_channel_geometry.json` is the measurement.

## 5. Open

Every item below is a bounded task in [`NEXT_WORK.md`](NEXT_WORK.md), with
evidence, reproduction path, acceptance gate and expected compute. Summarised
here so this file stays the one place the state is described:

- **No certification is reproducible from committed code.** All nine reports that
  record source hashes were produced on uncommitted working-tree state, across
  three sessions. This does not make a number wrong; it makes it uncheckable
  (T0, and it outranks the rest).
- The chain is certified on the state task and perception on rendered frames, and
  the two have never been combined at scale — **the highest-value missing
  measurement** (T1).
- **The learned insert policy does not seat, and the reason is now measured
  rather than inferred.** 0.00% over 1,536 episodes, a median of 204 mm short.
  The long-standing reading — that it *creeps*, at 3.65 mm/s against 120 mm/s of
  authority — is **refuted**: the time cost sized against the failure penalty,
  trained to convergence at 1,400 epochs, moves the shortfall 1.4 mm and every
  episode still spends its whole clock. The cost is paid, not avoided.

  What the same episodes show is attitude, and it is not the objective's to fix.
  The module ends at a median **84.5 mrad** against the **52.4 mrad** a seated
  module must be inside, with only 2–4% of episodes inside it. Three different
  reward configurations land within 0.4 mrad of each other:

  | Objective | Orientation | Inside tolerance |
  | --- | ---: | ---: |
  | baseline time cost | 84.26 mrad | 2.3% |
  | 4× time cost, converged | 84.61 mrad | 2.7% |
  | 7× orientation penalty | 84.58 mrad | 3.9% |

  The reset is square (`attitude_residual_rad` 0.0 at every station) and the grip
  holds (12.2 mm, p95 12.48), so the episode takes it there and no objective
  brings it back. **The attitude is set by the interface**: two flat pads on a
  pin cannot resist a moment about the closing axis, and the chain reaches
  46 mrad at the same phase only because it carries the module on a form lock.
  The blocker is the load path — **T9**, not T2.
  `evidence/insert_attitude_diagnosis.json`.

  **T9 landed on 2026-08-25, and it moved the skill further than anything since
  the task was built.** Trained on the chain's mating compliance, the median
  shortfall falls **202.2 → 98.6 mm** and **35.5%** of episodes now reach the
  seated plane, against essentially none before; lateral improves 7.10 → 4.51 mm.
  Success is still 0.00%, and **orientation is the only condition left**.

  It is not the policy's. Among the 91 episodes that reach depth, orientation
  clusters between **56.03 and 56.92 mrad** — a band under a milliradian wide,
  because the module is resting corner-to-corner against the channel walls at the
  largest angle the clearance permits. `check_workcell_geometry.py` names that
  angle independently as 56.40 mrad. The success criterion is **52.36 mrad**, so
  **this channel holds a module 4.04 mrad outside its own acceptance criterion**,
  and a module that merely rests in it cannot pass. The chain reaches 46 mrad
  because its form lock holds the module squarer than the channel does.

  That makes it a rack requirement, and the geometry check already brackets it:
  the clearance window is 10.35–12.69 mm, `GUIDE_CENTER_OFFSET_Y` sits at the top
  at 12.689 mm, and `2c/L` = 52.36 mrad at **c = 11.78 mm**. **Not changed here** —
  the chain was certified at 12.689 mm, so moving it moves a published number.

  **T9 is unblocked as of 2026-08-25.** Reaching the chain's load path from the
  skill needed three fixes that each masked the next: anchor the lock after the
  reset settles, run `joint_mode="fixed"` (with `"compliant"` the mating joint is
  never installed and softening is a no-op), and turn `replicate_physics` off
  (PhysX copies only env 0's authored joint). `--latch_mating_compliance` sets
  all three. Zero-shot on a policy trained without the lock, episodes survive and
  the module gets 20 mm further in; attitude is worse, which is expected for an
  unseen load path. A policy trained under it —
  `grapple_insert_l0_seed70_v23lock` — is the measurement that matters.
- Grasp and extract both miss the 95% gate (85.69% and 87.75% pooled), and
  neither responds to more epochs on the evidence available (T6).
- Every policy is **one PPO training seed**, so no number carries a spread (T3).
- Every certification is at **robustness level 0**; levels 1–4 exist and are
  unexercised (T4).
- **Training randomizes none of the variables the sweep says the chain is
  sensitive to.** A 10 mm error in where the robot parks across the bay takes it
  from 93.75% to 6.25% (T5).
- The insert skill still differs from the chain's seating phase in the **load
  path**, and the measurement is recorded in
  `tests/test_skill_chain_agreement.py` (T9).
- The **live compute service runs the superseded w65 policy set**, two promotions
  behind the certified chain (T7).
- **Grasp and extract are certified on the *old* rack.** The derived clearance
  narrows the source bay as well as the destination, so 85.69% and 87.75% are
  numbers taken in a channel this repository no longer builds. The chain re-run
  exercises both skills in situ and reproduces seed for seed, which bounds the
  effect at the chain level and says nothing about the skill level. Re-certifying
  them is `NEXT_WORK.md` **T12**. The predicted direction is favourable: the
  channel corner the pads have to follow falls from 15.000 mm — exactly their
  limit — to 13.654 mm.
- **Delivered angle has about 3 mrad of margin** — modules seat at 46.0 mrad
  (p95 46.5) against a channel throat that now accepts 49.18. It was ~10 mrad
  against the 56.40 mrad throat. The margin was deliberately spent: it is the
  same margin the seated criterion needed on the other side, and §4 records why
  the value sits between them rather than on either. Still the only quantity in
  the certification operating against a limit.

### What breaks the chain first

One variable at a time around the certified point, 16 environments and 16
episodes a point, one seed. Coarse on purpose — the Wilson interval on each point
is about twenty points wide, so this **ranks** variables rather than measuring
them. `evidence/chain_robustness_sweep.json`.

| Point | Success | Below nominal |
| --- | ---: | ---: |
| nominal | 93.75% | — |
| module 120 × 16 mm | **0.00%** | 93.75 |
| robot base +10 mm across the bay | **6.25%** | 87.50 |
| rack lateral clearance 16 mm | 75.00% | 18.75 |
| module mass 40 kg | 81.25% | 12.50 |
| module 140 × 26 mm | 87.50% | 6.25 |
| module mass 20 kg, rack clearance 6 mm, relief 0 mm | 93.75% | 0.00 |
| robot base 50 mm further back | 100.00% | −6.25 |

The closed-form envelope in `check_workcell_geometry.py` called every
cross-section result before the simulator was started. Mass is nearly free — a
much wider payload band than the interface specification claims.

## 6. Reproducing

```bash
# The whole job, one environment, end to end. About eight minutes.
scripts/run_robot_carried.sh rail

# The published 97.92%: 32 environments on each of three held-out seeds.
# CERT_TAG is required -- the default names preserved evidence.
CERT_TAG=<name> scripts/run_robot_carried.sh certify

# One skill, three stages, three held-out seeds.
SKILL=Extract CKPT=<path> TAG=<name> scripts/certify_grapple_skills.sh

# One variable at a time around the certified point.
scripts/sweep_chain_robustness.sh && python scripts/report_chain_robustness.py

# Geometry, no simulator, about a second.
python scripts/check_workcell_geometry.py

# Is any published number stale, and can it still be reproduced?
python scripts/check_criterion_currency.py         # did the criterion move under it
python scripts/check_source_provenance.py --depth 200  # do the source bytes still exist
python scripts/build_evidence_manifest.py --check
```

Two flags exist only so an archived checkpoint can be re-run under the criterion
it was certified against — which is what keeps a criterion change and a policy
change from being quoted as one number: `play.py --legacy_grip_ball_m 0.030` and
`--legacy_unbounded_reset`.

## 7. Where the rest of the reasoning is

`docs/archive/` holds the session handoffs in the order they were written, kept
for the reasoning and the negative results in them, not for their status claims —
every one is superseded by this file. `docs/archive/README.md` says what each
contains.

### Branches

`main` is canonical and carries everything. Two exploration branches are kept as
refs, and **neither holds any evidence file that `main` does not** — that was
checked file by file before consolidating, so nothing is lost by ignoring them.
They are kept for the reasoning in their commit messages:

| Branch | What is in it | Why kept |
| --- | --- | --- |
| `keyed-interface` | A keyed pin instead of a tapered one. Five real defects on that geometry, each found and fixed, **none of which moved extraction off 0.00%**. | Its final commit message is the write-up: it ends by naming parameter search as the wrong tool and the missing lead-in geometry as the actual gap. What stands — seated grip offset 19.4 → 0.7 mm, attitude 63.7 → 1.3 mrad, 40 N·m of lateral load held without slip — is worth reading before anyone re-opens a keyed interface. |
| `agent/zero-g-blade-swap` | The original eight-phase swap line, before the workcell move. | Historical origin. Superseded in every respect; merging it would delete current work. |

`industrial-relocation` was the working branch through 2026-08-25 and is now
identical to `main`.
