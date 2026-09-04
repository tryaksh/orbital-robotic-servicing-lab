# Agent instructions

The repository instructions are in [`AGENTS.md`](AGENTS.md). Read them, then
[`docs/NOW.md`](docs/NOW.md), whose current gates determine priority. Bounded
legacy task detail is in [`docs/NEXT_WORK.md`](docs/NEXT_WORK.md).

Three rules prevent the most expensive silent failures:

1. Never widen a tolerance to make a gate pass; derive it from the parts.
2. Never combine a criterion change and a policy change in one result.
3. Never claim a learned capability whose checkpoint is unreachable.

Do not read `evidence/*.json` in bulk. Query the generated
`evidence/MANIFEST.json`; its counts and status groups are authoritative.
