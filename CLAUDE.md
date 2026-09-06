# Agent instructions

Start with [`docs/HANDOFF.md`](docs/HANDOFF.md) — where the project is and the
single next action. Then [`AGENTS.md`](AGENTS.md) for the working rules,
[`docs/METHOD.md`](docs/METHOD.md) for what the project is building, and
[`docs/NOW.md`](docs/NOW.md) for verified state. Bounded tasks are in
[`docs/NEXT_WORK.md`](docs/NEXT_WORK.md); `docs/CHARTER.md` holds the
investigation that produced the method.

Three rules prevent the most expensive silent failures:

1. Never widen a tolerance to make a gate pass; derive it from the parts.
2. Never combine a criterion change and a policy change in one result.
3. Never claim a learned capability whose checkpoint is unreachable.

Do not read `evidence/*.json` in bulk. Query the generated
`evidence/MANIFEST.json`; its counts and status groups are authoritative.
