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

If one epoch exists under two filenames it stops and asks, rather than settling
it by a sort order; add ``--resolve <run>=<filename>`` to say which you mean.

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
    # Added 2026-08-25, and the omission is the reason it was added: this is the
    # driver for the chain that carries the headline number, and it was the one
    # script with a default checkpoint set that promotion never moved. It sat
    # two promotions behind, so the documented way to reproduce 97.92% ran the
    # superseded policies instead. A promotion tool that skips the promoted
    # chain is worse than none.
    "scripts/run_robot_carried.sh",
)
EPOCH = re.compile(r"_ep_(\d+)_")


def newest_checkpoint(run: str, chosen: dict[str, str] | None = None) -> str:
    """Return the highest-epoch checkpoint under a run, as a repo-relative path.

    Highest epoch, not newest mtime. A resumed run writes its early checkpoints
    after a later run's, and picking by mtime would silently promote a policy
    thousands of epochs behind the one meant.

    **One epoch can have two files, and this refuses to guess between them.**
    Some runs here carry both ``..._ep_1500_rew_35.348194.pth`` and
    ``..._ep_1500_rew__35.348194_.pth`` — the same epoch and the same reward
    under two rl-games naming conventions. Their weights are byte-identical, so
    nothing about the policy's behaviour depends on which you pick, but a report's
    ``checkpoint_sha256`` is a hash of the *file*, so the two give the same policy
    two different provenance records, and ``check_evidence_currency.py`` can then
    be made to disagree with itself.

    This used to break the tie by ``(file size, name)``, which selects the
    double-underscore file. That is **not** the one the current extraction
    certification was produced from: extract epoch 12600 is certified under
    ``last_..._ep_12600_rew_172.70488.pth`` at 1,341,301 bytes, and the size rule
    picks the 1,341,477-byte twin. A tool whose entire purpose is to stop
    evidence and scripts drifting apart should not settle that quietly.

    So a tie now stops the promotion and prints both candidates. Pass the one you
    mean with ``--resolve <run>=<filename>``.
    """

    candidates = sorted(LOG_ROOT.glob(f"*/{run}/nn/*_ep_*.pth"))
    if not candidates:
        raise SystemExit(f"no checkpoints under logs/rl_games/*/{run}/nn/")
    top = max(int(EPOCH.search(path.name).group(1)) for path in candidates)
    tied = sorted(path for path in candidates if int(EPOCH.search(path.name).group(1)) == top)
    if len(tied) == 1:
        return tied[0].relative_to(ROOT).as_posix()

    wanted = (chosen or {}).get(run)
    if wanted is not None:
        for path in tied:
            if path.name == wanted:
                return path.relative_to(ROOT).as_posix()
        raise SystemExit(
            f"--resolve {run}={wanted} names a file that is not one of the tied "
            f"candidates: " + ", ".join(path.name for path in tied)
        )

    listing = "\n".join(f"       {path.name}  ({path.stat().st_size:,} bytes)" for path in tied)
    raise SystemExit(
        f"{run}: epoch {top} exists under {len(tied)} filenames and this tool will not choose "
        f"between them.\n{listing}\n"
        f"Their weights are usually identical, but a report hashes the file, not the weights, so "
        f"picking the wrong one gives the same policy two provenance records.\n"
        f"Check which file the current certification in evidence/ was produced from, then pass:\n"
        f"       --resolve {run}=<filename>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", required=True, help="Run name of the promoted capture policy.")
    parser.add_argument("--extract", required=True, help="Run name of the promoted extraction policy.")
    parser.add_argument("--insert", required=True, help="Run name of the promoted insertion policy.")
    parser.add_argument(
        "--resolve",
        action="append",
        default=[],
        metavar="RUN=FILENAME",
        help=(
            "Which file to promote when one epoch exists under two filenames. Repeatable. "
            "Without it a tie stops the promotion rather than being settled by a sort order."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    chosen: dict[str, str] = {}
    for pair in args.resolve:
        run, _, filename = pair.partition("=")
        if not run or not filename:
            raise SystemExit(f"--resolve expects RUN=FILENAME, got {pair!r}")
        chosen[run] = filename

    promoted = {
        "GRASP_CKPT": newest_checkpoint(args.grasp, chosen),
        "EXTRACT_CKPT": newest_checkpoint(args.extract, chosen),
        "INSERT_CKPT": newest_checkpoint(args.insert, chosen),
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
