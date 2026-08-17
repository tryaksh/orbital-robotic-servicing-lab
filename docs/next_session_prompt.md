# Next session: the prompt to paste

Copy everything between the rules. It assumes nothing that is not in the repo.

---

You own `D:\6axis-space-robotics`. Read `CLAUDE.md` first — it is the plan, not
background. Act as the senior robotics simulation engineer who owns this repo.

FULL AUTHORITY for a long unattended session. Start GPU work immediately. Do not
ask permission to launch a run.

**FIRST ACTION, before anything else:** check for orphaned Isaac processes
(`Get-Process kit`), then run

```bash
python scripts/check_criterion_currency.py
```

and re-run whatever it flags STRONG. Five published numbers in this project have
been retracted for the same reason — a criterion moved after the number was
measured and nothing re-ran it — and that script is the mechanical defence.
`evidence/RETRACTED.md` lists every retraction and its replacement; read it
before quoting any figure.

THE GOAL is the relocation demo: capture → extract → transit across → insert into
bay 2. Industrially credible and paper-worthy: every skill certified, the full
chain certified, every number naming its evidence file.

## Where the last session left it

Items 1, 2 and 4 of the roadmap are done or closed. Item 3 was training when the
session ended. **Pick up at item 3's gate, then 5, then 6.**

1. Check the runs that were in flight:
   - `artifacts/relocation/queue2.log` — the two-slot smoke and the insert
     training on both bays.
   - `evidence/grapple_insert_two_slot_certification.json` if `certify2` got to
     run. Gate: **≥ 95% on the worse bay, not the pool.**
   - `evidence/workflow_install_clock30retain_certification.json` — the install
     chain under the corrected clock and the retain rule. This is item 1's
     number; if it is ≥ 95% the install chain is finally closed.
2. If item 3's gate passes, run item 5:
   ```bash
   bash scripts/run_relocation.sh relocate
   ```
   That certifies `capture → extract → lateral transit → insert(bay 2)` on three
   held-out seeds. Before believing a low number, trace it with `--handoff_trace`
   rather than retraining anything — the transit→insert row is also item 4's
   gate, which is **module held under 20 mm grip error across the transit**.
3. Item 6, perception reads the rack, is untouched and is the last one.

## What the last session learned, in one paragraph

The install chain's shortfall was never the insert policy. The hand-off costs the
skill 2.5 points, not the 15 the old "~80%" figure implied; 300 epochs of
fine-tuning on the chain's exact distribution moved it 89.41% → 88.37%; and the
two changes that *did* move it were a clock that had been truncating successful
insertions and a settling window that was still squeezing a seated module in
violation of this project's own operating rule. **Before training anything, check
whether the objective and the budget already forbid the behaviour you want.**

## Rules that are not negotiable

The fifteen in `CLAUDE.md`. Especially:

- Any phase that waits must either command or retain. This has now paid twice:
  removal 0.00% → 98.78%, installation 85.94% → 90.10%.
- Before training on any reconstructed distribution, run the unchanged successor
  policy on it and check it scores what it scores in the real chain. Five
  minutes. It has now saved two training runs and refuted one termination.
- Before blaming a policy, check the objective still has a gradient where the
  policy sits, and that its clock is not truncating its successes.
- Read constants, never restate them.
- Never quote a rate without `check_evidence_currency.py` **and**
  `check_criterion_currency.py`.
- Never weaken a success threshold to pass a gate. A *budget* is not a success
  threshold — but changing one obliges you to re-certify everything measured
  under the old one and to state the measurement that justified it.

COMPUTE: one Isaac process at a time, check for orphans before every launch. 512
environments uses ~0.9 GB of 12 GB. Training runs at roughly 19 PPO epochs per
minute at 512 environments, so a 1,200-epoch fine-tune is about an hour — budget
from that, not from guesswork. `train.py`'s redirected stdout lags minutes; judge
progress from `summaries/` tfevents and the `nn/` checkpoint mtime.
`--max_iterations` is an **absolute** epoch number, and `train.py` now refuses a
resume that has already passed it.

Capture every learning in `docs/status.md` as it happens, including negative and
retracted results. Commit directly to main, me as author, no Co-Authored-By
trailers. Report back with a plain-English table of what moved and what remains.

---
