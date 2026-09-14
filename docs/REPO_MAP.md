# Repository map

`main` contains the servicing simulation, design tools, mission application,
evidence and tests. The separate
[constrained-cable-safety repository](https://github.com/tryaksh/constrained-cable-safety)
contains the cable-constrained assembly recovery work.

## Main paths

| Path | Contents |
| --- | --- |
| `README.md` | Project overview and entry points |
| `docs/NOW.md` | Current results, limits and source provenance |
| `docs/NEXT_WORK.md` | Bounded follow-up tasks |
| `src/zero_g_blade_swap/` | Geometry, sensing, controllers and service code |
| `src/zero_g_blade_swap/service/mission_cli.py` | Terminal mission runner and offline verification |
| `policies/servicing_v2/` | Three frozen mission checkpoints, hashes and original training paths |
| `scripts/` | Research and validation commands; indexed in `scripts/README.md` |
| `evidence/` | Reports and generated `MANIFEST.json` |
| `tests/` | CPU contracts, application tests and optional simulator tests |
| `docs/handover/` | Historical reasoning and archived state/backlog documents |
| `artifacts/` | Local recordings, traces and logs; campaign shell scripts are tracked |
| `logs/`, `checkpoints/`, `datasets/` | Local training history and rendered corpora, outside git |

The package name `zero_g_blade_swap` and `Isaac-ZeroG-Blade-*` task IDs are kept
for checkpoint and report compatibility. A blade is a compute module.
`tasks` is a namespace package: import `zero_g_blade_swap.tasks.blade_swap` to
register the simulator tasks.

## Retired branches

The 2026-09-13 consolidation left one branch, `main`. The retired history remains
reachable:

| Former branch | Where to find it |
| --- | --- |
| `paper/serviceability-qualification` | Fast-forwarded into `main` |
| `industrial-relocation` | Already an ancestor of `main` |
| `research/assembly-recovery-training` | `archive/assembly-recovery-training` tag |
| `agent/zero-g-blade-swap` | `archive/zero-g-blade-swap` tag |
| Local `research/assembly-recovery` | Ancestor of the archived training branch |

```bash
git fetch --tags
git show archive/assembly-recovery-training:README.md
```

## Recovered servicing evidence

The archived assembly-training branch also contained servicing measurements
from September 4-6. Fifty-four reports were restored byte-for-byte:

| Group | Reports |
| --- | ---: |
| Seating skill and chain comparisons | 7 |
| Camera factorial and paired arms | 18 |
| Prediction, mechanism and trace controls | 16 |
| Gravity ladder | 5 |
| Paired n=192 rack/section arms | 5 |
| Rack-requirement and workcell checks | 3 |

Nineteen associated generator files and a design-library correction were restored.
The manifest initially classified the reports as historical. Several have since
been audited and promoted; T21/T22 in [NEXT_WORK.md](NEXT_WORK.md) cover the rest.
A report's presence in a tag is not evidence that its numbers have been reproduced.

The full reorganisation record is in
[the archived handover](handover/v8_two_repo_reorganisation.md). A separate
manuscript working directory at `D:/orbital-servicing-paper` is outside this
repository and was not modified for the mission application.
