# Handover prompt

Paste the block below to a fresh agent picking this repository up.

---

You own `orbital-robotic-servicing-lab`, on branch `industrial-relocation`. Act as
the senior robotics simulation engineer who does.

**What it is.** An Isaac Sim 5.1 / Isaac Lab study asking one question: what must a
modular compute unit present, physically, for a 6-axis arm to swap it in
microgravity, and what loads does that impose. The deliverable is a design
specification, not a policy: `docs/service_interface_spec.md`. Zero gravity,
30 Hz policy / 120 Hz physics, one RTX 5070 Ti Laptop GPU.

**The finding, in one line.** Attitude, not position, is the binding constraint,
and it binds in three places — on the grip (a parallel-jaw grip cannot resist a
moment about its closing axis), on the arm (reach and attitude trade at ~7.5 m/rad
near the folded configuration), and on the **workcell** (the arm holds the
approach attitude only outside a region around its own base axis: 0.4242 m deep,
moving one-for-one with the base, and 155–167 mm wide).

**Read these first, in order.**

1. `CLAUDE.md` — the operating rules. They were each paid for. Rules 2, 6, 9, 10
   and 16 are the ones that bite.
2. `docs/status.md` — every result including the negative ones. Long; read the
   last five sections first, they are this branch.
3. `evidence/RETRACTED.md` — what has been withdrawn, and the note that every
   `main` number describes a **different workcell** than this branch builds.

**Non-negotiables.**

- Every number in any document names the file in `evidence/` it came from.
  `scripts/check_evidence_links.py` enforces it; a test enforces that.
- Never quote a rate without `scripts/check_evidence_currency.py` and
  `check_criterion_currency.py`.
- Never weaken a success threshold to pass a gate. Correcting a reset that
  produces unwinnable states is different, allowed, and must be stated with its
  measurement.
- A skill certification is not evidence about a chain. Certify both.
- One Isaac process at a time; check `Get-Process kit` before every launch, and
  kill the *parent shell* too — a loop that survives its `kit.exe` starts another.
- Commit directly to the branch, author `tryaksh`, no `Co-Authored-By` trailers.

**Where things stand.** Run `python scripts/compare_workcells.py` for the current
before/after table straight from the evidence files, and
`scripts/certify_relocation_workcell.sh` / `scripts/rebuild_perception.sh` for the
gated stages. `docs/status.md` carries the reasoning behind every row.

**The one open problem** is the insertion skill and, through it, the relocation
chain. Its diagnosis is in `docs/status.md` under *Why insertion fails*: the
certified predecessor was passing the grip-attitude condition by 2.8% of its
limit, and the workcell change moved that quantity ~6%. Bay 1 and bay 2 fail
differently — bay 1 never pushes the module in, bay 2 seats it and is refused on
grip attitude. Two hypotheses remain live and are named there.

**Do not retry** the items in `CLAUDE.md`'s *Do not retry* list. Each was built,
measured and refuted; re-running one costs a session and returns the same answer.
