"""The documented way to reproduce a number must load the policies that produced it.

This repository has now recorded the same defect three times: ``evidence/`` named
one set of policies while a script loaded another, so a figure quoted about the
demonstration described a superseded checkpoint. ``promote_checkpoints.py`` was
written after the second occurrence and its docstring says the defaults "must
always name the promoted set ... and must be moved with it".

The third occurrence was the promotion tool's own coverage gap. It did not list
``scripts/run_robot_carried.sh`` -- the driver for the chain that carries the
headline rate -- so that file sat two promotions behind on the superseded w65
set, and ``run_robot_carried.sh certify`` did not reproduce the 97.92% it was
documented as reproducing.

A comment saying "keep these in sync" is what failed three times, so this checks
it instead. The checkpoint paths are read out of the preserved certification
rather than repeated here: repeating them would just move the place where the
two can disagree.

Source-level and CPU-only. No simulator, no GPU, no checkpoint weights -- so it
runs on every commit, which is the only way it catches the next promotion.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_robot_carried.sh"
PROMOTER = ROOT / "scripts" / "promote_checkpoints.py"
#: The single end-to-end run of the certified chain. It records all three
#: checkpoint paths; the pooled report keeps only the combined set hash.
CERTIFIED_RUN = ROOT / "evidence" / "robot_carried_full_chain_pin.json"


def _certified_checkpoints() -> dict[str, str]:
    """The three checkpoint paths the certified chain actually loaded."""
    report = json.loads(CERTIFIED_RUN.read_text(encoding="utf-8"))
    found: dict[str, str] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "checkpoints" in node and isinstance(node["checkpoints"], dict):
                found.update(node["checkpoints"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(report)
    assert found, f"{CERTIFIED_RUN.name} records no checkpoint paths"
    return found


def test_the_chain_runner_defaults_to_the_certified_policy_set() -> None:
    """`run_robot_carried.sh` must default to the policies the rate was measured on."""
    runner = RUNNER.read_text(encoding="utf-8")
    for phase, recorded in _certified_checkpoints().items():
        # The report is written on Windows and the script is POSIX.
        run_and_file = recorded.replace("\\", "/").split("zero_g_blade_insertion_contact/", 1)[1]
        assert run_and_file in runner, (
            f"run_robot_carried.sh does not default the {phase} policy to the checkpoint the "
            f"certification loaded ({run_and_file}). A reproduction path that loads a different "
            "policy reproduces a different number."
        )


def test_promotion_covers_the_script_that_runs_the_certified_chain() -> None:
    """The gap that let the defaults drift was the promoter's coverage, not the defaults."""
    promoter = PROMOTER.read_text(encoding="utf-8")
    assert '"scripts/run_robot_carried.sh"' in promoter, (
        "promote_checkpoints.py does not cover scripts/run_robot_carried.sh, so the next "
        "promotion will leave the certified chain's driver behind again."
    )


def test_the_extract_default_names_the_file_the_certification_hashed() -> None:
    """Epoch 12600 exists twice, byte-identical in weights and not in file hash.

    Both files hold the same policy -- the same 17 tensors, verified equal --
    but they hash differently, and a report's ``checkpoint_sha256`` is a file
    hash. Picking the other one would publish a provenance that disagrees with
    the preserved certification while changing no behaviour at all, which is the
    hardest kind of drift to notice. See docs/NEXT_WORK.md T8.
    """
    runner = RUNNER.read_text(encoding="utf-8")
    assert "last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth" in runner
    assert "last_zero_g_blade_insertion_contact_ep_12600_rew__172.70488_.pth" not in runner
