"""Arms compared as a paired A/B must have run the same policies.

The strongest experiment in this project is a substitution: the same chain, the
same checkpoints, one term changed -- the module pose comes from the simulator or
from the cameras. That claim is only as good as the checkpoints being identical,
and "identical" was asserted in prose rather than checked.

It is checkable. Every certification records the SHA-256 of the policy set it
loaded. If two arms of a paired comparison disagree on that hash, they differ by
more than the flag under test and the comparison is not what it says it is --
which is also the only way a paired reading of them could be wrong for a reason
the data cannot show.

Text only, so CI runs it. It reads the committed evidence and needs no simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"

#: ``(label, baseline report, treatment report)``. Only pairs the documentation
#: reports as one changed flag on a fixed cohort belong here.
PAIRS = [
    (
        "the substitution: module pose from the cameras against from the simulator",
        "workflow_robot_carried_m130pin_vision_datum_pair_certification.json",
        "workflow_robot_carried_m130pin_vision_oracle_control_v2_certification.json",
    ),
    (
        "rack retention against the no-rack control",
        "workflow_robot_carried_release_rack_retention_control_v1_certification.json",
        "workflow_robot_carried_release_rack_retention_v1_certification.json",
    ),
]


def _checkpoint_hash(name: str) -> str:
    path = EVIDENCE / name
    if not path.exists():
        pytest.skip(f"{name} is not present")
    report = json.loads(path.read_text(encoding="utf-8"))
    policy = report.get("policy")
    if isinstance(policy, dict) and isinstance(policy.get("checkpoint_sha256"), str):
        return policy["checkpoint_sha256"]
    for key in ("policy_set_sha256", "checkpoint_sha256"):
        value = report.get(key)
        if isinstance(value, str):
            return value
    pytest.skip(f"{name} records no policy hash")
    raise AssertionError("unreachable")


@pytest.mark.parametrize(("label", "baseline", "treatment"), PAIRS, ids=lambda v: v if isinstance(v, str) else "")
def test_both_arms_loaded_the_same_policies(label: str, baseline: str, treatment: str) -> None:
    first, second = _checkpoint_hash(baseline), _checkpoint_hash(treatment)
    assert first == second, (
        f"{label}: the two arms loaded different policy sets\n"
        f"  {baseline}: {first}\n"
        f"  {treatment}: {second}\n"
        "One changed flag was claimed. A different checkpoint is a second change."
    )
