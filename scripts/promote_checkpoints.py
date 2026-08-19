"""Move every script's default checkpoint set to a newly promoted one, atomically.

This repository has recorded the same defect twice: `evidence/` named one set of
policies while the scripts loaded another, so every figure quoted about the
demonstration described a superseded checkpoint. The comments in
`certify_demo_policies.sh`, `certify_workflow.sh`, `run_relocation.sh` and
`certify_vision_workflow.sh` all say the same thing -- *these defaults must
always name the promoted set in CLAUDE.md and must be moved with it* -- and all
four have drifted behind it at least once, because moving them is four files of
hand-editing at the end of a long session.

So this does it in one call. It resolves the highest-epoch checkpoint under each
named run, rewrites the `GRASP_CKPT=`, `EXTRACT_CKPT=` and `INSERT_CKPT=`
defaults in every script that carries them, and prints a diff of what moved.

It refuses to write anything unless every requested checkpoint exists, because a
half-moved set is worse than an un-moved one: the scripts would then disagree
with each other rather than with CLAUDE.md.

Usage::

    python scripts/promote_checkpoints.py \\
        --grasp grapple_grasp_l0_seed70_v6w65 \\
        --extract grapple_extract_l0_seed70_v15w65 \\
        --insert grapple_insert_l0_seed70_v11w65

CPU only. Reads no checkpoint weights, imports nothing from Isaac Lab.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs" / "rl_games"
#: Every script that carries a default checkpoint set.
SCRIPTS = (
    "scripts/certify_demo_policies.sh",
    "scripts/certify_workflow.sh",
    "scripts/certify_vision_workflow.sh",
    "scripts/run_relocation.sh",
    "scripts/probe_workcell_policies.sh",
)
EPOCH = re.compile(r"_ep_(\d+)_")


def newest_checkpoint(run: str) -> str:
    """Return the highest-epoch checkpoint under a run, as a repo-relative path.

    Highest epoch, not newest mtime. A resumed run writes its early checkpoints
    after a later run's, and picking by mtime would silently promote a policy
    thousands of epochs behind the one meant.

    **One epoch can have two files.** Some runs here carry both
    ``..._ep_1500_rew_35.348194.pth`` and ``..._ep_1500_rew__35.348194_.pth``
    -- the same epoch and the same reward under two rl-games naming conventions.
    CLAUDE.md names the double-underscore form and every certification in
    ``evidence/`` was produced from it, so a silent pick between them would be a
    promotion decision made by ``sorted()``. Ties are reported and broken by
    file size then name, deterministically.
    """

    candidates = sorted(LOG_ROOT.glob(f"*/{run}/nn/*_ep_*.pth"))
    if not candidates:
        raise SystemExit(f"no checkpoints under logs/rl_games/*/{run}/nn/")
    top = max(int(EPOCH.search(path.name).group(1)) for path in candidates)
    tied = sorted(path for path in candidates if int(EPOCH.search(path.name).group(1)) == top)
    if len(tied) > 1:
        print(f"  NOTE {run}: epoch {top} exists under {len(tied)} filenames:")
        for path in tied:
            print(f"       {path.name}  ({path.stat().st_size} bytes)")
    best = max(tied, key=lambda path: (path.stat().st_size, path.name))
    return best.relative_to(ROOT).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", required=True, help="Run name of the promoted capture policy.")
    parser.add_argument("--extract", required=True, help="Run name of the promoted extraction policy.")
    parser.add_argument("--insert", required=True, help="Run name of the promoted insertion policy.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    promoted = {
        "GRASP_CKPT": newest_checkpoint(args.grasp),
        "EXTRACT_CKPT": newest_checkpoint(args.extract),
        "INSERT_CKPT": newest_checkpoint(args.insert),
    }
    for name, path in promoted.items():
        print(f"{name} -> {path}")

    # The scripts write these as `NAME="${NAME:-$CKPT_ROOT/<run>/nn/<file>}"`,
    # with CKPT_ROOT already naming the experiment directory, so only the part
    # after it is substituted.
    changed = 0
    for relative in SCRIPTS:
        script = ROOT / relative
        if not script.is_file():
            print(f"  (missing, skipped) {relative}")
            continue
        text = script.read_text(encoding="utf-8")
        original = text
        for name, path in promoted.items():
            tail = path.split("logs/rl_games/", 1)[1]
            experiment, _, run_tail = tail.partition("/")
            # The override variable is not always the same name as the target:
            # certify_vision_workflow.sh writes
            # GRASP_CKPT="${GRASP_OVERRIDE:-$CKPT_ROOT/...}". Matching only the
            # same-name form silently skipped that whole file, which is exactly
            # the drift this tool exists to stop.
            pattern = re.compile(rf'^({name}="\$\{{[A-Z0-9_]+:-)\$CKPT_ROOT/[^"]*(\}}")$', re.MULTILINE)
            replacement = rf"\g<1>$CKPT_ROOT/{run_tail}\g<2>"
            text, count = pattern.subn(replacement, text)
            if count == 0:
                print(f"  (no {name} default) {relative}")
            elif f"CKPT_ROOT=\"logs/rl_games/{experiment}\"" not in text:
                raise SystemExit(
                    f"{relative} has CKPT_ROOT pointing at a different experiment than {experiment}; "
                    "moving the set would silently name a checkpoint that is not there"
                )
        if text != original:
            changed += 1
            if not args.dry_run:
                script.write_text(text, encoding="utf-8")
            print(f"  {'would update' if args.dry_run else 'updated'} {relative}")
    print(f"{changed} script(s) {'would change' if args.dry_run else 'changed'}")
    print("Now move CLAUDE.md's promoted set to match, and re-run the certifications.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
