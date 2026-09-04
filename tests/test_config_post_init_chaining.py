"""A config class may only chain `__post_init__` through its own ancestry.

`ZeroGBladeGrapplePinInsertForcePlayEnvCfg` borrowed the two-slot play config's
`__post_init__` by calling it unbound with its own instance. That cannot work:
the borrowed body runs a zero-argument `super()`, which binds to the class the
source sits in, and the instance passed in is a sibling rather than a subclass,
so Python raises `TypeError: super(type, obj): obj must be an instance or
subtype of type` before the environment is built.

The registration was valid, so `tests/test_task_registry_resolves.py` passed. The
task failed on construction instead, ten seconds into each of the three skill
runs of the 2026-09-04 force verification, and the aggregation that followed
reported a missing episode file rather than a broken task. The first seating
policy that can feel contact therefore reached its chain arm without ever having
been scored on its own skill.

Same reasoning as the registry test: this needs no simulator. Whether a call can
resolve is a property of the source, so it is checked in the source.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "zero_g_blade_swap"


def _class_bases() -> dict[str, set[str]]:
    """Map every class defined in the package to the base names it is written with.

    Bases are matched by *name* across the whole package rather than resolved
    through imports. That is deliberately loose in the safe direction: an unknown
    name is treated as an ancestor's ancestor and simply ends the walk, so this
    can miss a violation but cannot invent one.
    """

    bases: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            named = {base.id for base in node.bases if isinstance(base, ast.Name)}
            named |= {base.attr for base in node.bases if isinstance(base, ast.Attribute)}
            bases.setdefault(node.name, set()).update(named)
    return bases


def _ancestors(name: str, bases: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    frontier = list(bases.get(name, ()))
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(bases.get(current, ()))
    return seen


def _unbound_post_init_calls() -> list[tuple[Path, int, str, str]]:
    """Every ``Other.__post_init__(...)`` written inside a class body."""

    found: list[tuple[Path, int, str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner in ast.walk(tree):
            if not isinstance(owner, ast.ClassDef):
                continue
            for node in ast.walk(owner):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not isinstance(function, ast.Attribute) or function.attr != "__post_init__":
                    continue
                target = function.value
                if isinstance(target, ast.Call) and getattr(target.func, "id", None) == "super":
                    continue  # `super().__post_init__()` is the correct form
                if not isinstance(target, ast.Name):
                    continue
                found.append((path, node.lineno, owner.name, target.id))
    return found


def test_post_init_is_only_chained_through_a_real_ancestor() -> None:
    bases = _class_bases()
    violations = []
    for path, line, owner, target in _unbound_post_init_calls():
        if target in ("super", owner):
            continue
        if target not in _ancestors(owner, bases):
            violations.append(
                f"{path.relative_to(ROOT).as_posix()}:{line}: {owner} calls "
                f"{target}.__post_init__, and {target} is not one of its bases"
            )
    assert not violations, "unbound __post_init__ borrowed from outside the class's ancestry:\n" + "\n".join(
        violations
    )


def test_the_scan_actually_reaches_the_config_classes() -> None:
    """Guard against the walk above silently matching nothing."""

    bases = _class_bases()
    assert "ZeroGBladeGrapplePinInsertForcePlayEnvCfg" in bases
    assert "ZeroGBladeGrapplePinInsertForceEnvCfg" in _ancestors(
        "ZeroGBladeGrapplePinInsertForcePlayEnvCfg", bases
    )
    assert len(bases) > 100, f"only {len(bases)} classes parsed, which means this test is broken"
