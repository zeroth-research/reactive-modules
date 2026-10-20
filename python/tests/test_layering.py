"""Which module may import what, read off the source.

The package is layered. `LAYERS` names the modules of each layer and the
layers it may import besides itself, and a module imports nothing else of
`zrth.lean`. An import under `if TYPE_CHECKING:` is exempt: it names a type
for the reader and loads nothing.

A route is a package of `zrth.lean.infer`: its row in `__init__.py` and its
engine beside it. No route imports another; what two share lives in
`infer.common` or under `zrth.lean.smt`. A route's row imports its engine
inside the function that runs it, so `verith --help` loads no engine.

And a name with a leading underscore is its module's own: no module of
`zrth.lean` imports one from another, nor reads one off an imported module.
A name another module needs is part of its module's interface, and is
spelled without the underscore.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

LEAN = Path(__file__).parent.parent / "zrth" / "lean"
INFER = LEAN / "infer"

# Every module of the package: the SMT layer the routes are built from, and
# the Lean code generators.
PACKAGE = sorted(LEAN.rglob("*.py"))

# Each layer: the modules it is made of (a package names everything under
# it), and the layers it may import besides itself. A module no layer names
# is not checked, but a checked module may import it only through a layer.
LAYERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "core": (frozenset({"zrth.lean.core"}), frozenset()),
    "lowering": (frozenset({"zrth.lean.lowering"}), frozenset({"core"})),
    "artifacts": (frozenset({"zrth.lean.artifacts"}), frozenset({"core"})),
    "codegen": (frozenset({"zrth.lean.codegen"}),
                frozenset({"core", "lowering"})),
    "smt": (frozenset({"zrth.lean.smt"}),
            frozenset({"core", "lowering", "artifacts"})),
    # The project's vocabulary, templates, lakefile, build and `Main.lean`;
    # `scaffold`, which writes the certificate's files, sits above `proof`.
    "project": (frozenset({
        "zrth.lean.project.names", "zrth.lean.project.templates",
        "zrth.lean.project.lakefile", "zrth.lean.project.build",
        "zrth.lean.project.main_lean",
    }), frozenset({"core", "lowering", "codegen"})),
    "proof": (frozenset({"zrth.lean.proof"}),
              frozenset({"core", "lowering", "codegen", "smt", "project"})),
    "scaffold": (frozenset({"zrth.lean.project.scaffold"}), frozenset({
        "core", "lowering", "codegen", "project", "proof"})),
    # The input formats; one encodes a module through `smt` as it reads it.
    "inputs": (frozenset({"zrth.lean.inputs"}), frozenset({"core", "smt"})),
    # The routes, their engines and the registry over them.
    "infer": (frozenset({"zrth.lean.infer"}), frozenset({
        "core", "lowering", "artifacts", "codegen", "smt", "project",
        "proof", "scaffold"})),
    # The command line: read off the input formats' and the routes' rows.
    "cli": (frozenset({"zrth.lean.cli"}), frozenset({
        "core", "artifacts", "smt", "inputs", "infer"})),
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _module(path: Path) -> str:
    """`path` as a dotted module name: `zrth.lean.infer.houdini.engine`."""
    parts = path.relative_to(LEAN.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["zrth", *parts])


MODULES = {_module(p) for p in PACKAGE}


def _package(path: Path) -> list[str]:
    """`path`'s package as a dotted list: `["zrth", "lean", "infer"]`."""
    return ["zrth", *path.parent.relative_to(LEAN.parent).parts]


class Import(NamedTuple):
    module: str          # absolute
    name: str | None     # what a `from` import takes; `None` for `import`
    bound: str           # the name the import binds
    line: int
    top: bool            # runs when the module is imported
    typing: bool         # under `if TYPE_CHECKING:`

    @property
    def target(self) -> str:
        """The module this import loads: `from zrth.lean.core import errors`
        loads `zrth.lean.core.errors`, and so does `from zrth.lean.core.errors
        import Refused`."""
        if self.name and f"{self.module}.{self.name}" in MODULES:
            return f"{self.module}.{self.name}"
        return self.module


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            or isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _import_nodes(body, *, top: bool, typing: bool):
    """`(node, top, typing)` for every import statement under `body`."""
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, top, typing
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            yield from _import_nodes(node.body, top=top, typing=True)
            yield from _import_nodes(node.orelse, top=top, typing=typing)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _import_nodes(node.body, top=False, typing=typing)
        else:
            for field in ("body", "orelse", "finalbody", "handlers"):
                yield from _import_nodes(getattr(node, field, ()), top=top,
                                         typing=typing)


def _imports(path: Path):
    """Every import in `path`, the module absolute."""
    pkg = _package(path)
    for node, top, typing in _import_nodes(_tree(path).body, top=True,
                                           typing=False):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield Import(a.name, None, a.asname or a.name.split(".")[0],
                             node.lineno, top, typing)
        else:
            if node.level:
                base = pkg[:len(pkg) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            for a in node.names:
                yield Import(module, a.name, a.asname or a.name, node.lineno,
                             top, typing)


def _layer(module: str) -> str | None:
    for layer, (members, _) in LAYERS.items():
        if any(module == m or module.startswith(m + ".") for m in members):
            return layer
    return None


# Every route: a package of `infer` other than `common`.
ROUTE_DIRS = sorted(d for d in INFER.iterdir()
                    if (d / "__init__.py").exists() and d.name != "common")
ROUTES = {d.name for d in ROUTE_DIRS}


def _route(module: str) -> str | None:
    """The route `module` belongs to: `houdini` for
    `zrth.lean.infer.houdini.engine`."""
    parts = module.split(".")
    if parts[:3] == ["zrth", "lean", "infer"] and len(parts) > 3:
        return parts[3] if parts[3] in ROUTES else None
    return None


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _name(path: Path) -> str:
    return str(path.relative_to(LEAN))


LAYERED = [p for p in PACKAGE if _layer(_module(p))]


@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_every_module_a_layer_names_exists(layer):
    """A module moved or renamed takes its layer entry with it; a stale
    entry would check nothing."""
    members, below = LAYERS[layer]
    assert members and all(m in MODULES for m in members), sorted(
        m for m in members if m not in MODULES)
    assert below <= set(LAYERS), sorted(below - set(LAYERS))


@pytest.mark.parametrize("path", LAYERED, ids=_name)
def test_a_layer_imports_only_the_layers_below_it(path):
    me = _layer(_module(path))
    allowed = {me} | LAYERS[me][1]
    bad = []
    for i in _imports(path):
        if i.typing or not i.target.startswith("zrth.lean."):
            continue
        layer = _layer(i.target)
        if layer not in allowed:
            bad.append(f"{_name(path)}:{i.line} ({me}) imports {i.target} "
                       f"({layer or 'in no layer'})")
    assert not bad, "\n".join(bad)


def test_the_routes_are_found():
    assert ROUTES >= {
        "ai", "ai_cegis", "fbk_proveit", "houdini", "nuterm", "smt_linear",
        "sygus", "vampire"}


@pytest.mark.parametrize("route", ROUTE_DIRS, ids=lambda d: d.name)
def test_every_route_has_an_engine(route):
    """The row is `__init__.py` and what it runs is `engine.py`, so the rule
    below names one module."""
    assert (route / "engine.py").exists()


INFER_MODULES = sorted(p for p in INFER.rglob("*.py")
                       if p != INFER / "__init__.py")


@pytest.mark.parametrize("path", INFER_MODULES, ids=_name)
def test_no_route_imports_another(path):
    """A route imports its own modules and what no route owns; a module
    that belongs to no route imports none. The registry, which lists the
    rows, is the one module that knows them all."""
    me = _route(_module(path))
    bad = [f"{_name(path)}:{i.line} imports {i.target}"
           for i in _imports(path)
           if _route(i.target) not in (None, me)]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("route", ROUTE_DIRS, ids=lambda d: d.name)
def test_a_route_row_imports_its_engine_only_when_it_runs(route):
    """The CLI imports every row to print `--help`; an engine imported with
    its row would be loaded by every run, whichever route it asks for."""
    path = route / "__init__.py"
    engine = f"zrth.lean.infer.{route.name}.engine"
    bad = [f"{_name(path)}:{i.line} imports {i.target} at module level"
           for i in _imports(path)
           if i.top and not i.typing and i.target == engine]
    assert not bad, "\n".join(bad)


def test_the_registry_loads_no_engine():
    """What the rule above is for, observed: importing every row, as
    `verith --help` does, imports no engine and no module an engine
    imports at its top."""
    probe = ("import sys, zrth.lean.infer; print(sorted(m for m in "
             "sys.modules if m.endswith('.engine') or m == 'cvc5'))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, check=True)
    assert r.stdout.strip() == "[]", r.stdout


@pytest.mark.parametrize("path", PACKAGE, ids=_name)
def test_no_private_name_crosses_a_module(path):
    bad = []
    imported = set()
    for i in _imports(path):
        if not i.module.startswith("zrth"):
            continue
        imported.add(i.bound)
        if i.name and _private(i.name):
            bad.append(f"{_name(path)}:{i.line} imports {i.module}.{i.name}")
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
        f"{_name(p)}:{i.line} imports {i.module}"
        for p in sorted((LEAN / "project").glob("*.py"))
        for i in _imports(p)
        if i.module.startswith("zrth.lean.infer")
    ]
    assert not bad, "\n".join(bad)
