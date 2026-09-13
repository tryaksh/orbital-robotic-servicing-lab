# Repository map

What this repository holds, what its branches are, and where everything that used
to be here went. If you are an agent or a person picking this up cold, read this
before anything except [README.md](../README.md).

Measured on 2026-09-13 with `git ls-remote`, `git merge-base` and `git rev-list`.
Nothing below is assumed.

---

## The two repositories, and which is which

There are two, and until 2026-09-13 they held three projects between them across
six branches, with no way to tell which branch was what. That is fixed.

| Repository | Question it answers |
| --- | --- |
| **orbital-robotic-servicing-lab** (this one) | Can a robot service a modular spacecraft rack in zero gravity, which constraint stops it, and do skills that pass on their own survive being chained together? |
| [**constrained-cable-safety**](https://github.com/tryaksh/constrained-cable-safety) | An assembly attempt has failed and the robot must back off and try again. What does a safety check need to know before that retreat, and does it survive being used repeatedly and on hardware it was not tuned for? |

Neither repository now contains the other's subject matter. If you find cable or
peg-insertion work here, or spacecraft racks there, it is a mistake and should be
reported.

---

## Branches here

| Branch | State | What is on it |
| --- | --- | --- |
| `main` | **alive — the only branch** | Everything. The zero-gravity servicing chain, the serviceability qualification work, 245 evidence reports and the full history. |

**One branch, on purpose.** Before this session there were five, three of which
were a different project or already merged, and a newcomer had no way to tell
which was current. The sister repository has exactly one branch and is easier to
understand for it.

The remote is `https://github.com/tryaksh/orbital-robotic-servicing-lab.git`.

Open a branch when you have a reason a name can state. `main` is what the
portfolio and any reader will land on, so it has to be the current work.

---

## What was retired on 2026-09-13, and where it went

Nothing was deleted before it was provably reachable somewhere else. Two of the
four were preserved by a tag; the other two were already contained in a branch
that survives.

| Removed | Why | Where it is now |
| --- | --- | --- |
| `paper/serviceability-qualification` | It *was* the current work, 188 commits ahead of `main` and 0 behind, while `main` sat on a README quoting a result that branch had already retracted. A reader landing on `main` saw a number nobody stood behind any more. | **Fast-forwarded into `main`.** Not a rewrite and not a merge commit: `main` was a strict ancestor, so `main` now points at exactly what that branch pointed at. |
| `research/assembly-recovery-training` | 623 commits of a different project — peg insertion, a reinforcement-learning training stack, and the cable studies. Its 35 unpushed commits were pushed first, so nothing was lost before it moved. | Tagged **`archive/assembly-recovery-training`**. Its evidence and frozen contracts were copied into the cable repository, where the subject matter belongs; the code and the full history stay reachable through the tag. |
| `industrial-relocation` | A fully merged ancestor of `main`: 0 commits ahead, 15 behind. Nothing was on it that was not already on `main`. | Reachable from `main`. Verified with `git merge-base --is-ancestor` before deletion. |
| `agent/zero-g-blade-swap` | 14 commits of an abandoned early experiment — "Autonomous Server Blade Swap in Zero-G" on a UR10e — last touched 2026-08-08, 6 ahead of a shared base and 293 behind. | Tagged **`archive/zero-g-blade-swap`**. Nothing on it existed anywhere else, so the tag is the only thing keeping it alive. |

A local-only `research/assembly-recovery` pointer, an ancestor of the training
branch, was deleted as well.

To read anything that was retired:

```
git fetch --tags
git checkout archive/assembly-recovery-training
git checkout archive/zero-g-blade-swap
```

---

## Layout

The same shape as the sister repository, so an agent knows where to look in
either.

| Path | What is in it |
| --- | --- |
| `README.md` | What this is and what it found, for someone who has never seen it |
| `ROADMAP.md` | What is closed, what is open, and what each open item costs |
| `AGENTS.md` | Operating rules and the command sequence |
| `docs/REPO_MAP.md` | This file |
| `docs/NOW.md` | Verified current state, in detail |
| `docs/NEXT_WORK.md` | The long-form open task list, by task number |
| `docs/handover/` | Session handovers, kept as history. Not maintained: each describes the repository as it was the day it was written. |
| `src/zero_g_blade_swap/` | The importable package |
| `scripts/` | Entry points, one job each, indexed in `scripts/README.md` |
| `configs/` | Frozen contracts |
| `evidence/` | Immutable reports, plus a generated `MANIFEST.json` |
| `tests/` | Fast CPU tests. No simulator, no GPU, no graphics. |
| `artifacts/` | Generated output, git-ignored except the campaign queue scripts |

Two differences from the cable repository, both deliberate:

- The generated evidence index is **`evidence/MANIFEST.json`**, not `INDEX.json`.
  It does more than the other one: it classifies every report as canonical,
  retracted or historical, and the reports reference it by path.
- `CLAUDE.md` sits beside `AGENTS.md` and names the currently promoted checkpoint
  set. That is machine state rather than documentation, which is why it is not in
  `docs/`.

---

## Things that will look wrong and are not

- **The package is called `zero_g_blade_swap`** and the tasks are registered as
  `Isaac-ZeroG-Blade-…`. "Blade" means a server blade; this began as a
  blade-swapping experiment. 35 evidence reports record the hash of a file at
  `src/zero_g_blade_swap/…`, so renaming the package would break every one of
  those provenance links to make a directory listing read better. The name stays.
- **`src/zero_g_blade_swap/tasks/` has no `__init__.py`.** It is a namespace
  package on purpose. The consequence is that `import zero_g_blade_swap.tasks`
  runs no code and registers nothing — import `…tasks.blade_swap`, which is where
  the `gym.register` calls are. One script had this wrong for months.
- **A clone does not carry the trained policies.** They live under `logs/` and
  `checkpoints/`, which are not in git. Every report records the hash of the
  checkpoint behind it, so reports can be read without them but not re-run.
- **About 865 MB of raw run artifacts** — trajectories, per-tick samples, the
  detail the reports were derived from — are untracked in `artifacts/` on the
  workstation that produced them, and in no repository. The reports carry their
  hashes, so they can be checked but not replaced.

---

## A document that lives outside both repositories

`D:/orbital-servicing-paper` on the workstation holds a manuscript draft with no
git remote and no backup. It is not tracked here. It was deliberately not touched
on 2026-09-13, and it is mentioned only so that nobody looking for it concludes it
was lost.
