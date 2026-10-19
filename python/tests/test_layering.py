"""Which module may import what, read off the source.

A route is a module of `zrth.lean.magic` that defines a `TA2Magic`: what two
routes share lives under `zrth.lean.smt` or in `smt_synth`, never in one of
them. And a name with a leading underscore is its module's own: no module of
`zrth.lean` imports one from another, nor reads one off an imported module.
A name another module needs is part of its module's interface, and is
spelled without the underscore.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

LEAN = Path(__file__).parent.parent / "zrth" / "lean"
MAGIC = LEAN / "magic"

# Every module of the package: the SMT layer the routes are built from, and
# the Lean code generators.
PACKAGE = sorted(LEAN.rglob("*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _package(path: Path) -> list[str]:
    """`path`'s package as a dotted list: `["zrth", "lean", "magic"]`."""
    return ["zrth", *path.parent.relative_to(LEAN.parent).parts]


def _imports(path: Path):
    """`(module, name, bound, line)` for every import in `path`, the module
    absolute.

    `name` is what a `from` import takes out of `module`, and `None` for a
    plain `import`; `bound` is the name the import binds in `path`."""
    pkg = _package(path)
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, None, a.asname or a.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            for a in node.names:
                yield module, a.name, a.asname or a.name, node.lineno


def _is_route(path: Path) -> bool:
    return any(
        isinstance(node, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id == "TA2Magic" for b in node.bases)
        for node in ast.walk(_tree(path))
    )


ROUTES = {f"zrth.lean.magic.{p.stem}" for p in MAGIC.glob("*.py") if _is_route(p)}


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _name(path: Path) -> str:
    return str(path.relative_to(LEAN))


def test_the_routes_are_found():
    assert {r.rsplit(".", 1)[1] for r in ROUTES} >= {
        "ai", "cegis", "houdini", "learn", "linear", "sygus", "vampire"}


@pytest.mark.parametrize("path", sorted(MAGIC.glob("*.py")), ids=_name)
def test_no_route_imports_another(path):
    me = f"zrth.lean.magic.{path.stem}"
    bad = []
    for module, name, _, line in _imports(path):
        target = f"{module}.{name}" if name and f"{module}.{name}" in ROUTES else module
        if target in ROUTES and target != me:
            bad.append(f"{_name(path)}:{line} imports {target}")
    assert not bad, "\n".join(bad)


ROW_DIR = LEAN / "infer_route"
ROW_MODULES = sorted(
    p for p in ROW_DIR.glob("*.py") if p.stem not in {"__init__", "api", "shared"}
)


@pytest.mark.parametrize("path", ROW_MODULES, ids=_name)
def test_no_route_row_imports_another(path):
    """A row names its engine and the shared types, never a sibling row."""
    rows = {f"zrth.lean.infer_route.{p.stem}" for p in ROW_MODULES}
    bad = []
    for module, name, _, line in _imports(path):
        target = f"{module}.{name}" if name and f"{module}.{name}" in rows else module
        if target in rows and target != f"zrth.lean.infer_route.{path.stem}":
            bad.append(f"{_name(path)}:{line} imports {target}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("path", PACKAGE, ids=_name)
def test_no_private_name_crosses_a_module(path):
    bad = []
    imported = set()
    for module, name, bound, line in _imports(path):
        if not module.startswith("zrth"):
            continue
        imported.add(bound)
        if name and _private(name):
            bad.append(f"{_name(path)}:{line} imports {module}.{name}")
    # `mod._name`, where `mod` is a module or a name this one imported.
    for node in ast.walk(_tree(path)):
        if (isinstance(node, ast.Attribute) and _private(node.attr)
                and isinstance(node.value, ast.Name)
                and node.value.id in imported):
            bad.append(f"{_name(path)}:{node.lineno} reads "
                       f"{node.value.id}.{node.attr}")
    assert not bad, "\n".join(bad)


def test_the_project_builder_does_not_know_the_routes():
    """`create_project` is told what to remove and what to add to the lakefile;
    it does not read the route table to find out."""
    bad = [
        f"project.py:{line} imports {module}"
        for module, _, _, line in _imports(LEAN / "project.py")
        if module.startswith("zrth.lean.infer_route")
    ]
    assert not bad, "\n".join(bad)
