# Agent instructions

The instructions for this repository are in [`AGENTS.md`](AGENTS.md), which is
tool-agnostic and applies to any coding agent. **Read it, then
[`docs/NOW.md`](docs/NOW.md).** Nothing is duplicated here, so that the two can
never disagree.

Work to pick up: [`docs/NEXT_WORK.md`](docs/NEXT_WORK.md), prioritised, T1 first.

Three rules are worth having before you open anything, because breaking them is
expensive and quiet:

1. **Never widen a tolerance to make a gate pass.** Replace a wrong criterion with
   one derived from the parts, and re-run the old checkpoint under both.
2. **A criterion change and a policy change are never quoted as one number.**
3. **Never claim a capability whose checkpoint is not reachable.** `logs/` and
   `checkpoints/` are gitignored; a clone has the reports and none of the weights.

Do not read `evidence/*.json` in bulk — 162 files, most superseded. Query
`evidence/MANIFEST.json` instead.
