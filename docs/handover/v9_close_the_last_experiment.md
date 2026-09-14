# V9 — Close the one experiment that can still change the story

**This project is not finished, and that is fine.** The chain scores 91.67%
against an unchanged 95% gate and the serviceability envelope returns *not
qualified*. Neither of those is a problem to hide. What *is* a problem is that one
decisive experiment has never run because of a defect, the most transferable
finding in the repository is buried inside a library, and the only recording in
the repository misrepresents what it shows.

This session runs the experiment, surfaces the finding, and makes the honest state
legible. Then it stops.

**EXECUTED 2026-09-13. This file is now history; do not run it again.**

The experiment had already run and its evidence had been lost. The play
configuration's `TypeError` was fixed on 2026-09-04, the runs were retried the same
day, and the certificates went into a branch retired two days later as a different
project. Fifty-four reports came back out of `archive/assembly-recovery-training`
along with nineteen generator files and a library correction; the premise that the
skill half had never been scored was wrong on both counts, and the recovery is the
larger part of what this session did. `docs/REPO_MAP.md` has the inventory.

What the experiment decided: the skill certifies at **99.20%** on 3,001 episodes,
the first learned seating skill here to pass its own gate, and scores **24/96** in
the chain against the scripted advance's **23/96**. The missing cell of the 2x2 ran
on 2026-09-13 and the answer is that the *bay* is the variable and the effect
crosses over -- the shipped relief is worth +41.7 points to the scripted controller
and -16.7 to the learned one. `evidence/seating_bay_factorial_v1.json`.

Also done: the two skills' gap attributed to a failure mode each; `2c/L` given a
single runnable entry point with the measured rack pinned in a test; the kinematics
check made per environment without touching its tolerance; two clips of the
certified chain recorded and checked against their own reports; two falsely named
clips renamed; the document counts pinned by tests; and the working frame removed
from the reader-facing documents with a lint to keep it out. No gate moved and
nothing was produced for a website.

Still open, with prices, in `ROADMAP.md`: fifty-one recovered reports unread
(**T21**), the nineteen restored generators unverified against their own reports
(**T22**), and the pooled camera-driven certification (**T1**).

Prepared 2026-09-13, after the two-repository reorganisation
([v8](v8_two_repo_reorganisation.md)) left this repository with one branch, 1,049
tests and a README a stranger can read.

---

## What this session must NOT produce

- **Nothing for a website.** No HTML, no hosted page, no published artifact, no
  demo site, no slide deck, no marketing copy. The owner is writing a Robotics Lab
  page on their own portfolio themselves, later, from what is in this repository.
  If you find yourself writing anything whose audience is a web visitor rather
  than an engineer reading the repository, stop.
- **No new training.** Not one epoch. Every run below is evaluation or a play run
  against weights that already exist.
- **No moved gates.** 95% is 95%. A gate changed after seeing a number is not a
  gate.
- **No touching `D:/orbital-servicing-paper`.** The manuscript draft lives outside
  both repositories and is out of scope, as it was last session.

---

## Why this session exists

Three things are worth more than anything else left on the list.

**One experiment can still overturn a standing result.** The seating skill has
been doing contact-rich assembly blind for this project's entire history — there
was no force channel in its observations and the scene had no contact sensor.
`v33force` is the first policy able to feel the contact it is making, and on an
identical reward function its training reward reached 98.2 where the blind policy
plateaued at 43.9 after five times the epochs. **It has never been scored.** The
three runs meant to score it each exited in about ten seconds with no episodes,
because the play configuration raised `TypeError` at construction. That defect was
fixed on 2026-09-13 and the runs have not been retried.

Either the attitude comes down, and a negative result that has stood since the
beginning is wrong in an interesting way — or it does not, and the interface bound
survives its strongest remaining challenge. Both outcomes are worth having. Not
knowing is not.

**The best finding is invisible.** `2c/L` — a rigid part of length `L` in a
channel with `c` of clearance per side cannot be held squarer than `2c/L`, so the
rack and not the robot decides the attitude of a part resting in it — is the one
result here that transfers to hardware nobody has built yet. It is implemented, in
`servicing_design.py`, as a library function. Nobody can run it.

**The media lies.** Every clip in the repository is from superseded checkpoints,
pre-fix geometry, or both, and `3_full_chain_seated.mp4` is misnamed: its own run
reports a 4.62 mm lateral error against a 2.5 mm tolerance. A repository that is
this careful about its numbers should not carry a file whose name is a false claim.

---

## What to do, in order

Commit at every stage boundary and push. Register a gate before a run, not after.

### Stage 1 — Score the seating policy that can feel contact

The blocking defect is fixed. Run the skill half of `verify_insert_skill.sh` — the
policy alone, on three held-out seeds — and then the same weights inside the chain
against the scripted guarded advance, which is the arm that actually decides.

Before launching, write down the gate and commit it. After, write the certificate
into `evidence/` the way every other certification is written, and update the
manifest.

**Report a rate, never a reward.** Training reward is not success and must not be
quoted as one; a policy can collect reward in ways that never seat a module. The
existing chain-half result of 4/24 was measured on one seed in a bay 3.897 mm
outside its own requirement and decides nothing — say so rather than averaging it
in.

If the run cannot be made to happen, that is a result too: record what blocked it
and what it would cost.

### Stage 2 — Say why capture and extraction miss their gate

Both sit around 87% against 95%, and both overlap the earlier certificates they
were meant to beat, so neither retrain can be called an improvement. Nobody has
asked *which failure mode* accounts for the gap. This is evaluation only, on
weights that already exist, and it turns "misses the gate" into "misses the gate
because", which is a far stronger thing to be able to say.

### Stage 3 — Make the design rule something an engineer can run

`interface_regime()` and the `2c/theta` depth limit already exist in
`src/zero_g_blade_swap/servicing_design.py`. Give them one entry point: module
geometry and channel clearance in, and out comes the attitude that channel can hold,
the acceptance tolerance it would need, and whether those two are compatible at all.

Pin the measured rack in a test — it held a module at **56.40 mrad** while
demanding **52.36 mrad** to accept it as seated, which is the case where the rule
bites. CPU only, no simulator.

This is the piece of this project a hardware engineer could use before anyone has
cut metal, which is the whole stated point of the repository. It should not take
reading a library to find it.

### Stage 4 — Make the media honest

Either record one clip of the chain as it is currently certified, or delete the
clips that misrepresent it and say plainly in `docs/DEMOS.md` that no recording
shows the certified chain. **Do not leave a file whose name is a claim its own run
contradicts.** Both endings are acceptable; leaving it as it is, is not.

If you record: it is one run, about eight minutes, and the clip must be checked
against the report of the run that produced it rather than against its filename.

### Stage 5 — Fix the check that fires when nothing is wrong

The closed-form kinematics agreement check fails about one run in fifteen and costs
a sweep point when it does. It takes the maximum across every environment at one
instant, so a single environment whose joints have not been written yet is enough
to trip it. **Do not widen the tolerance** — it is right and it has caught real
defects. Run it per environment and report which one disagreed, or run it after the
first reset has stepped everywhere. Under an hour.

### Stage 6 — Make the documentation match the evidence, and make it shorter

`docs/NOW.md` says *53 canonical, 11 retracted, 159 historical*. The generated
manifest says **64, 12 and 167**. Nothing catches that, which is the same class of
drift `tests/test_documented_numbers.py` exists to prevent — extend it so the
counts are pinned too.

Then prune, and **rule 2 below is most of the pruning**: no "session", no
addressing an agent, no naming the writing standard, and nothing explained twice
across documents that should each own one thing. There are over six thousand lines
across fourteen documents, and `README.md` and `ROADMAP.md` are now the front
doors. Superseded prose is what gets deleted; a result is never deleted to tidy
up. If you cannot tell which a passage is, it is a result.

---

## Rules

1. **Plain English.** `~/.claude/CLAUDE.md` carries the owner's writing preference
   and it applies to every document you touch: natural English, explain unfamiliar
   terms where they first appear, concrete over abstract, focused but not cryptic.
   The current `README.md` is the standard being asked for — read it first.
2. **Write for the reader, not about the work.** These documents still read in
   places as though they were produced by and for a working session, and a reader
   has no idea what a session is: a `ROADMAP.md` heading that says *"Done in this
   session"*, *"this session deliberately did not touch it"*, *"the open task list
   with costs, as the sessions wrote it"*, and a repository map that opens by
   addressing an agent. Delete that frame. State what is true now and what was
   found; git history already records who changed what and when. Three more, while
   you are there:
   - **Do not name the writing standard inside the document.** "Plain English" is
     an instruction to whoever is writing, not content for whoever is reading.
   - **Say each thing once, in the document that owns it.** The README owns what
     the project is, `ROADMAP.md` owns what is open, `docs/REPO_MAP.md` owns where
     things went, `docs/NOW.md` owns the detailed verified state. They currently
     re-explain each other. Cross-link instead — and this is most of the six
     thousand lines Stage 6 is asking you to cut.
   - **Cut the paragraph that restates the paragraph above it.** If a passage
     survives being deleted, it was not carrying anything.

   Two exemptions, and they are real: `AGENTS.md`, `CLAUDE.md` and everything
   under `docs/handover/` are addressed to whoever picks the work up next, so
   process words belong there. And **keep genuine domain vocabulary** — a
   *handoff* here is one skill handing state to the next skill in the chain, which
   is a technical term and is load-bearing in several results. Do not sweep a word
   away because it looks like process-speak without checking what it means.
3. **Never delete a result.** Failed runs, retracted certificates and losing arms
   stay with their scope. `evidence/RETRACTED.md` is part of the record, not an
   embarrassment.
4. **Every claim keeps its evidence link**, and quote canonical reports only.
5. **Report the controller that actually stepped the phase**, never a
   configuration flag, and never an inherited historical rate.
6. Capture commit, source hashes, config and seeds **before** launch. A primary
   result needs a clean tree.
7. Run the full CPU suite and `scripts/build_evidence_manifest.py --check` before
   every commit.

## Traps already paid for

- **Isaac Sim's shutdown takes the process with it.** Anything after
  `app.close()` does not run. Write records before closing the app.
- **There is no display on this machine.** Anything that opens a window cannot be
  run here. Render to a file and inspect the file, and say in the summary what was
  never executed.
- **`import zero_g_blade_swap.tasks` registers nothing** — it is a namespace
  package with no `__init__.py`. Import `…tasks.blade_swap`, which is where the
  `gym.register` calls live.
- **A clone does not carry the weights.** They are under `logs/` and
  `checkpoints/`, which are not in git, and about 865 MB of raw run artifacts are
  untracked in `artifacts/` on this workstation only.
- **Multi-line edits:** write a small Python patch script with `assert old in s`
  before each replace. Bash heredocs fail on this machine on apostrophes and
  triple quotes.

## Done conditions

1. The contact-feeling seating policy has a rate, or a written record of exactly
   what stopped it and what finishing would cost.
2. The gap between the skill rates and their gate is attributed to a failure mode
   rather than left as a number.
3. The `2c/L` design rule has a single entry point an engineer can run, with the
   measured rack pinned in a test.
4. No file in the repository carries a name its own run contradicts.
5. Every count and rate in the maintained documents is defended by a test.
6. **No maintained document mentions a session, addresses an agent, names the
   writing standard, or explains the project a second time.** `AGENTS.md`,
   `CLAUDE.md` and `docs/handover/` are exempt.
7. Nothing is uncommitted, nothing is unpushed, no gate was moved, no website
   material was produced.

Finish by telling the owner, in plain English: what the experiment decided, what
the design rule now lets someone do, what you deleted and why it was safe, and what
is still open with its price.
