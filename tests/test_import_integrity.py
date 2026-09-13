"""Every import of this repository's own code must point at something real.

Why this test exists
--------------------
A sibling repository deleted a helper module that two live entry points still
imported. Every unit test passed, because no unit test imports an entry point,
and every simulator entry point failed the moment it was started. Nothing caught
it until someone tried to run the project.

The same trap is wider open here. Most scripts and most of
``src/zero_g_blade_swap/tasks/`` import Isaac Lab, which only exists inside the
simulator's own Python environment, so the ordinary test suite never imports
them and never would. Importing them for real is not an option on a machine
without Isaac Sim.

So this test reads the code instead of running it. It parses every Python file,
finds every import that refers to this repository's own packages — absolute and
relative — and checks that the module exists and that the name being imported out
of it is actually defined there. It runs in well under a second and needs nothing
installed.

It also carries one narrow check for a specific defect, described at
``test_no_class_borrows_another_class_post_init`` below, which cost three
simulator runs and left a policy unscored.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Directories whose Python files are checked.
SOURCE_DIRECTORIES = ("src", "scripts", "tests")

# Top-level names that belong to this repository rather than to a package
# installed from elsewhere. Anything else in an import statement is treated as a
# third-party dependency and left alone.
FIRST_PARTY_ROOTS = ("zero_g_blade_swap", "scripts", "tests")

# Where each first-party root's files live, relative to the repository root.
ROOT_DIRECTORIES = {
    "zero_g_blade_swap": ROOT / "src",
    "scripts": ROOT,
    "tests": ROOT,
}


def python_files() -> list[pathlib.Path]:
    """Every Python file in the repository worth checking, sorted."""
    found: list[pathlib.Path] = []
    for directory in SOURCE_DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.append(path)
    return found


ALL_FILES = python_files()


def module_path(dotted: str) -> pathlib.Path | None:
    """Turn ``zero_g_blade_swap.math_utils`` into the file it lives in.

    Returns ``None`` if the name does not belong to this repository. A first-party
    name with no file on disk comes back as a path that does not exist, which is
    exactly the failure we are looking for.
    """
    root = dotted.split(".", 1)[0]
    if root not in FIRST_PARTY_ROOTS:
        return None
    base = ROOT_DIRECTORIES[root]
    parts = dotted.split(".")
    as_module = base.joinpath(*parts).with_suffix(".py")
    as_package = base.joinpath(*parts, "__init__.py")
    return as_package if as_package.exists() and not as_module.exists() else as_module


def dotted_name_of(path: pathlib.Path) -> str | None:
    """The dotted module name a file is imported as, or ``None`` if it is not on a
    package path we track."""
    for root, base in ROOT_DIRECTORIES.items():
        try:
            relative = path.relative_to(base)
        except ValueError:
            continue
        if relative.parts[0] != root and root != "scripts" and root != "tests":
            continue
        parts = list(relative.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts and parts[0] in FIRST_PARTY_ROOTS:
            return ".".join(parts)
    return None


def resolve_relative(path: pathlib.Path, level: int, module: str | None) -> str | None:
    """Turn ``from ..mdp import x`` inside a file into an absolute dotted name."""
    own = dotted_name_of(path)
    if own is None:
        return None
    parts = own.split(".")
    # A package's ``__init__`` is the package itself, so level 1 means "me";
    # in any other module level 1 means the package the module sits in.
    package = parts if path.name == "__init__.py" else parts[:-1]
    if level - 1 > len(package):
        return None
    base = package[: len(package) - (level - 1)] if level > 1 else package
    return ".".join([*base, module]) if module else ".".join(base)


def names_defined_in(path: pathlib.Path) -> set[str]:
    """Every name a module binds at its top level.

    Covers ``def``, ``class``, plain and annotated assignment, ``import`` and
    ``from ... import``. A star import is reported by returning a set containing
    ``"*"`` so the caller can decline to judge.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    defined: set[str] = set()
    for node in tree.body:
        defined.update(_bound_by(node))
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            # Names bound inside ``try``/``except ImportError`` or
            # ``if TYPE_CHECKING`` still count as defined.
            for inner in ast.walk(node):
                defined.update(_bound_by(inner))
    return defined


def _bound_by(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            names.update(_assigned_names(target))
        return names
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _assigned_names(node.target)
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    return set()


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_assigned_names(element))
        return names
    return set()


def first_party_imports(path: pathlib.Path) -> list[tuple[int, str, str | None]]:
    """Every first-party import in one file, relative ones resolved.

    Each entry is ``(line number, dotted module name, name imported out of it)``
    where the third item is ``None`` for a plain ``import x.y``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    found: list[tuple[int, str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FIRST_PARTY_ROOTS:
                    found.append((node.lineno, alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level:
                module = resolve_relative(path, node.level, node.module)
                if module is None:
                    continue
            if module and module.split(".")[0] in FIRST_PARTY_ROOTS:
                for alias in node.names:
                    found.append((node.lineno, module, alias.name))
    return found


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_first_party_import_resolves(path: pathlib.Path) -> None:
    """Every module this file imports from the repository exists on disk."""
    for lineno, dotted, _name in first_party_imports(path):
        target = module_path(dotted)
        assert target is not None, "first-party filter let a third-party name through"
        where = f"{path.relative_to(ROOT).as_posix()}:{lineno}"
        assert target.exists(), f"{where} imports {dotted}, which has no file at {target}"


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_imported_name_is_defined(path: pathlib.Path) -> None:
    """Every name pulled out of one of this repository's modules is defined there."""
    for lineno, dotted, name in first_party_imports(path):
        if name is None or name == "*":
            continue
        target = module_path(dotted)
        if target is None or not target.exists():
            continue  # the test above already reports this
        # ``from .tasks import blade_swap`` imports a submodule, not a name inside
        # ``__init__.py``.
        submodule = module_path(f"{dotted}.{name}")
        if submodule is not None and submodule.exists():
            continue
        defined = names_defined_in(target)
        if "*" in defined:
            continue  # a star import; we cannot tell without running it
        where = f"{path.relative_to(ROOT).as_posix()}:{lineno}"
        assert name in defined, f"{where} imports {name} from {dotted}, which does not define it"


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_class_borrows_another_class_post_init(path: pathlib.Path) -> None:
    """A class may not call an unrelated class's ``__post_init__`` with its own self.

    ``ZeroGBladeGrapplePinInsertForcePlayEnvCfg`` did this to reuse presentation
    settings from a class it does not inherit from. The borrowed method's
    zero-argument ``super()`` then resolves against a type ``self`` is not an
    instance of, and Python raises ``TypeError: super(type, obj)`` at
    construction — long after the test suite has passed, and only on a machine
    with the simulator installed.

    It cost the three runs meant to score the first seating policy able to feel
    contact: each exited in about ten seconds with no episodes, and the policy is
    still unscored on its own. The correct form, used everywhere else in this
    package, is to call the shared ``configure_*`` helper directly.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    for name, node in classes.items():
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "__post_init__" or not isinstance(call.func.value, ast.Name):
                continue
            borrowed = call.func.value.id
            if borrowed in bases:
                continue  # an explicit base-class call is fine
            where = f"{path.relative_to(ROOT).as_posix()}:{call.lineno}"
            raise AssertionError(
                f"{where}: {name} calls {borrowed}.__post_init__ but does not inherit from "
                f"{borrowed}; this raises TypeError at construction. Call the shared "
                f"configure_* helper instead."
            )


def test_the_check_covers_the_repository() -> None:
    """Guard against the walk silently finding nothing."""
    scripts = [p for p in ALL_FILES if p.parts[len(ROOT.parts)] == "scripts"]
    sources = [p for p in ALL_FILES if p.parts[len(ROOT.parts)] == "src"]
    assert len(scripts) > 50, f"only {len(scripts)} scripts found; the walk is wrong"
    assert len(sources) > 50, f"only {len(sources)} source files found; the walk is wrong"
