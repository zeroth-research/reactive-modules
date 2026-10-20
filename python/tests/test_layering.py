"""Which module may import what, read off the source.

The package is layered. `LAYERS` names the modules of each layer and the
layers it may import besides itself, and a module imports nothing else of
`zrth.lean`. An import under `if TYPE_CHECKING:` is exempt: it names a type
for the reader and loads nothing.

A route is a module of `zrth.lean.magic` that defines a `TA2Magic`: what two
routes share lives under `zrth.lean.smt` or in a module of `magic` that is
not a route (`synth`, `llm`), never in one of them. A route's row imports its engine inside the function that runs it, so
`verith --help` loads no engine. And a name with a leading underscore is its
module's own: no module of `zrth.lean` imports one from another, nor reads
one off an imported module. A name another module needs is part of its
module's interface, and is spelled without the underscore.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest

LEAN = Path(__file__).parent.parent / "zrth" / "lean"
MAGIC = LEAN / "magic"

# Every module of the package: the SMT layer the routes are built from, and
# the Lean code generators.
PACKAGE = sorted(LEAN.rglob("*.py"))

# Each layer: the modules it is made of (a package names everything under
# it), and the layers it may import besides itself. A module no layer names
# is not checked, but a checked module may import it only through a layer.
LAYERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "core": (frozenset({"zrth.lean.core"}), frozenset()),
    "ir": (frozenset({
        "zrth.lean.sorts", "zrth.lean.layout", "zrth.lean.literals",
        "zrth.lean.ops",
    }), frozenset({"core"})),
    "artifacts": (frozenset({"zrth.lean.artifacts"}), frozenset({"core"})),
    "smt": (frozenset({"zrth.lean.smt"}), frozenset({"core", "ir", "artifacts"})),
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _module(path: Path) -> str:
    """`path` as a dotted module name: `zrth.lean.magic.houdini`."""
    parts = path.relative_to(LEAN.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["zrth", *parts])


MODULES = {_module(p) for p in PACKAGE}


def _package(path: Path) -> list[str]:
    """`path`'s package as a dotted list: `["zrth", "lean", "magic"]`."""
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
    assert {r.rsplit(".", 1)[1] for r in ROUTES} >= {
        "ai", "cegis", "houdini", "learn", "linear", "sygus", "vampire"}


@pytest.mark.parametrize("path", sorted(MAGIC.glob("*.py")), ids=_name)
def test_no_route_imports_another(path):
    me = f"zrth.lean.magic.{path.stem}"
    bad = [f"{_name(path)}:{i.line} imports {i.target}"
           for i in _imports(path) if i.target in ROUTES and i.target != me]
    assert not bad, "\n".join(bad)


ROW_DIR = LEAN / "infer_route"
ROW_MODULES = sorted(
    p for p in ROW_DIR.glob("*.py") if p.stem not in {"__init__", "api", "shared"}
)


def test_the_route_rows_are_found():
    assert {p.stem for p in ROW_MODULES} >= {
        "ai", "ai_cegis", "fbk_proveit", "houdini", "nuterm", "smt_linear",
        "sygus", "vampire"}


@pytest.mark.parametrize("path", ROW_MODULES, ids=_name)
def test_no_route_row_imports_another(path):
    """A row names its engine and the shared types, never a sibling row."""
    rows = {f"zrth.lean.infer_route.{p.stem}" for p in ROW_MODULES}
    me = f"zrth.lean.infer_route.{path.stem}"
    bad = [f"{_name(path)}:{i.line} imports {i.target}"
           for i in _imports(path) if i.target in rows and i.target != me]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("path", ROW_MODULES, ids=_name)
def test_a_route_row_imports_its_engine_only_when_it_runs(path):
    """The CLI imports every row to print `--help`; an engine imported with
    its row would be loaded by every run, whichever route it asks for."""
    bad = [f"{_name(path)}:{i.line} imports {i.target} at module level"
           for i in _imports(path)
           if i.top and not i.typing and i.target in ROUTES]
    assert not bad, "\n".join(bad)


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
        f"project.py:{i.line} imports {i.module}"
        for i in _imports(LEAN / "project.py")
        if i.module.startswith("zrth.lean.infer_route")
    ]
    assert not bad, "\n".join(bad)
