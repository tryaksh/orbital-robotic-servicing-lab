# Agent instructions

The repository instructions are in [`AGENTS.md`](AGENTS.md). Read them, then
[`docs/CHARTER.md`](docs/CHARTER.md), which carries the research question and
the day-14 decision, then [`docs/NOW.md`](docs/NOW.md) for verified state.
Bounded tasks are in [`docs/NEXT_WORK.md`](docs/NEXT_WORK.md).

Three rules prevent the most expensive silent failures:

1. Never widen a tolerance to make a gate pass; derive it from the parts.
2. Never combine a criterion change and a policy change in one result.
3. Never claim a learned capability whose checkpoint is unreachable.

Do not read `evidence/*.json` in bulk. Query the generated
`evidence/MANIFEST.json`; its counts and status groups are authoritative.
