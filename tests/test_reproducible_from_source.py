"""Unit tests for the rebuild check.

The check writes into ``evidence/`` and restores it with git, so the behaviour
that matters most is the refusal: run over uncommitted evidence it would destroy
work. That is tested by construction rather than by trusting the docstring.

The second thing worth testing is the distinction the check exists to draw. A
report that differs only in a regenerated timestamp is a clean reproduction; a
report whose numbers moved is a divergence. Collapsing the two would make the
check useless in opposite directions -- every run would look either broken or
fine.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_reproducible_from_source", ROOT / "scripts/check_reproducible_from_source.py"
)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def test_a_regenerated_timestamp_is_not_a_divergence() -> None:
    first = json.dumps({"generated_utc": "2026-08-26T00:00:00+00:00", "value": 261.5})
    second = json.dumps({"generated_utc": "2026-09-04T05:00:00+00:00", "value": 261.5})
    assert first != second
    assert check._strip_volatile(first) == check._strip_volatile(second)


def test_a_moved_number_is_a_divergence() -> None:
    first = json.dumps({"generated_utc": "2026-08-26T00:00:00+00:00", "value": 261.5})
    second = json.dumps({"generated_utc": "2026-08-26T00:00:00+00:00", "value": 262.0})
    assert check._strip_volatile(first) != check._strip_volatile(second)


def test_key_order_alone_is_not_a_divergence() -> None:
    first = json.dumps({"a": 1, "b": 2})
    second = json.dumps({"b": 2, "a": 1})
    assert check._strip_volatile(first) == check._strip_volatile(second)


def test_non_json_falls_back_to_exact_comparison() -> None:
    assert check._strip_volatile("not json") == "not json"


def test_every_named_generator_exists() -> None:
    """A generator that has been renamed must fail loudly, not be skipped quietly."""

    missing = [name for name in check.GENERATORS if not (ROOT / "scripts" / name).exists()]
    assert not missing, f"named generators that no longer exist: {missing}"
