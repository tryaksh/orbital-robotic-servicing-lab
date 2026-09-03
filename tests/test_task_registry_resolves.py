"""Every registered task must name a config class that exists.

A registration that points at a missing or renamed class fails only when the task
is launched, which in this repository means an hour into a queued GPU batch, with
the failure looking like an Isaac problem rather than a typo. Nothing checked it
because checking it looked like it needed Isaac Lab -- and it does not: the
registry names a module and a class, and both are readable as source.

This is the same reasoning as `tests/test_skill_chain_agreement.py`. If it has to
keep being true, it has to be checked without a simulator.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap"
REGISTRY = TASKS / "__init__.py"


def _classes_in(module_name: str) -> set[str]:
    path = TASKS / f"{module_name}.py"
    if not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _entry_point_module(node: ast.AST) -> str | None:
    """Return the module named by an ``f"{__name__}.<module>:{...}"`` entry point."""

    for joined in ast.walk(node):
        if not isinstance(joined, ast.JoinedStr):
            continue
        rendered = "".join(part.value for part in joined.values if isinstance(part, ast.Constant))
        match = re.search(r"\.(\w+):", rendered)
        if match:
            return match.group(1)
    return None


def _registrations() -> list[tuple[str, str]]:
    """Return every ``(module, class)`` pair the registry actually registers.

    Two shapes exist and both are handled: a ``for`` over ``(id, class_name)``
    tuples whose body registers with a templated entry point, and a bare
    ``gym.register`` whose entry point names the class directly.
    """

    tree = ast.parse(REGISTRY.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            module = _entry_point_module(node)
            if module is None:
                continue
            # The loop's iterable is a tuple of (task id, config class name).
            for element in ast.walk(node.iter):
                if not isinstance(element, ast.Tuple) or len(element.elts) != 2:
                    continue
                second = element.elts[1]
                if (
                    isinstance(second, ast.Constant)
                    and isinstance(second.value, str)
                    and second.value.endswith("EnvCfg")
                ):
                    pairs.append((module, second.value))
        elif isinstance(node, ast.Call):
            function = node.func
            name = getattr(function, "attr", None)
            if name != "register":
                continue
            module = _entry_point_module(node)
            if module is None:
                continue
            for joined in ast.walk(node):
                if not isinstance(joined, ast.JoinedStr):
                    continue
                rendered = "".join(p.value for p in joined.values if isinstance(p, ast.Constant))
                match = re.search(r"\.\w+:(\w+EnvCfg)$", rendered)
                if match:
                    pairs.append((module, match.group(1)))
    return sorted(set(pairs))


def test_every_registered_config_class_exists() -> None:
    missing: list[str] = []
    checked = 0
    for module, name in _registrations():
        available = _classes_in(module)
        if not available:
            continue  # the module lives elsewhere in the package; not this test's job
        checked += 1
        if name not in available:
            missing.append(f"{module}:{name}")
    assert checked > 30, f"the registry parse only found {checked} pairs, which means this test is broken"
    assert not missing, f"registered but not defined: {sorted(set(missing))}"


def test_the_new_task_families_are_all_registered() -> None:
    """The ids the queued batches launch, spelled once so a typo cannot hide."""

    registry = REGISTRY.read_text(encoding="utf-8")
    for task in (
        "Isaac-ZeroG-Blade-GrapplePin-GraspNoised-v0",
        "Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-v0",
        "Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-Play-v0",
        "Isaac-ZeroG-Blade-GrapplePin-ExtractPoseNoised-Play-v0",
        "Isaac-ZeroG-Blade-GrapplePin-ExtractVelocityNoised-Play-v0",
        "Isaac-ZeroG-Blade-GrapplePin-InsertWedgeGated-v0",
    ):
        assert f'"{task}"' in registry, f"{task} is launched by a queued batch and is not registered"


def test_no_task_id_is_registered_twice() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    ids = re.findall(r'"(Isaac-ZeroG-Blade-[\w-]+)"', registry)
    duplicates = {task for task in ids if ids.count(task) > 1}
    assert not duplicates, f"registered more than once: {sorted(duplicates)}"
