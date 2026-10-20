"""Every way verith can be handed a certificate, as one row each.

``--infer <name>`` selects a row.  The row says what the route needs of the
property, what it does with predicates supplied alongside it, what it reads
and writes of the generated project, and what it hands back.  The CLI
(:mod:`zrth.lean.cli`) reads the rows; it knows nothing about any particular
route, which is what makes adding one a module and a line here.

Two things are deliberate about the shape.

**The check is generic and the reason is not.**  Whether a route accepts a
safety property is a set membership; *why* the ``ai`` route does not is prose
about ``rule_globally``, and why ``nuterm`` discards a precondition is prose
about Houdini.  So the row carries both: the set the CLI tests, and the
sentence it prints when the test fails.

**The project is generated before inference runs, and that is the point.**
Some routes read the emitted Lean (the LLM ones prompt on the functional
encoding), and generating it twice is a second chance to generate it
differently.  Some routes write more of the project than the predicates --
``fbk-proveit`` writes an NA model, installs a certificate and emits a
bridge.  So a route is handed a :class:`ProjectHandle` rather than a path,
and ``reads``/``owns`` on its row are the whole of its access: what a run
wrote is what the row says it may have.

What a route cannot do after generation is change the *lakefile* -- it is
written while the project is created.  That is why the row declares its
lakefile needs as data (``lakefile``) and resolves what those needs name at
parse time (``resolve``), rather than discovering it later.

One package per route.  ``infer/<name>/__init__.py`` is the route's row,
and the modules beside it are its engine: ``engine.py``, which the row's
``run`` builds and calls, and whatever only that engine uses.  The types a
row is written in are :mod:`.route` (the row) and :mod:`.io` (what a route
is given and hands back).

Lazy imports.  A row imports the types, constants and the options in
:mod:`.common.options` at module level; its ``run`` and hooks import the
engine when they are called.  A run that did not ask for a route should not
need its packages present -- ``anthropic``/``openai`` for the LLM routes,
``z3`` and the learner for ``nuterm``, a ``lean-ltl-certifying`` checkout
for ``fbk-proveit`` -- and ``verith --help``, which reads every row, imports
no engine.

Adding a route: write ``infer/<name>/__init__.py`` exporting ``ROUTE`` and
its engine beside it, and add the package to the tuple below.  No route
imports another; what two share goes in :mod:`.common` or under
:mod:`zrth.lean.smt`.
"""

from __future__ import annotations

from ..core.errors import Refused
from ..core.options import Opt
from . import (
    ai,
    ai_cegis,
    fbk_proveit,
    houdini,
    nuterm,
    smt_linear,
    sygus,
    vampire,
)
from .io import FIXABLE, InferInput, InferResult, ProjectHandle
from .route import Alias, InferRoute

__all__ = [
    "FIXABLE", "Alias", "InferInput", "InferResult", "InferRoute", "Opt",
    "ProjectHandle", "ROUTES", "DEFAULT_ROUTE", "index_routes",
    "route_by_name", "route_names", "all_route_options",
    "shared_route_options", "all_route_aliases", "owned_files",
]

# The order is the order `--infer` lists them in.
ROUTES: tuple[InferRoute, ...] = tuple(
    m.ROUTE
    for m in (
        ai,
        ai_cegis,
        nuterm,
        sygus,
        smt_linear,
        houdini,
        vampire,
        fbk_proveit,
    )
)

DEFAULT_ROUTE = "ai-cegis"


def index_routes(routes: "tuple[InferRoute, ...]") -> dict[str, InferRoute]:
    """The rows by name, refusing a duplicate.

    `lowering/ops.py` guards its op table the same way: a shadowed row would
    also make `--infer` offer the same name twice.
    """
    by_name: dict[str, InferRoute] = {}
    for r in routes:
        if r.name in by_name:
            raise RuntimeError(
                f"two route-table rows named {r.name!r} in "
                f"infer.ROUTES: the second would shadow the first and "
                f"`--infer` would offer the name twice"
            )
        by_name[r.name] = r
    return by_name


_BY_NAME = index_routes(ROUTES)


def route_by_name(name: str) -> InferRoute:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"no inference route named {name!r}; add a row to "
            f"infer.ROUTES (have: {', '.join(_BY_NAME)})"
        ) from None


def route_names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def all_route_options() -> tuple[tuple[InferRoute, Opt], ...]:
    """Every route-scoped option, with the route that owns it."""
    return tuple((r, o) for r in ROUTES for o in r.options)


def shared_route_options() -> tuple[tuple[Opt, tuple[InferRoute, ...]], ...]:
    """Every distinct option, with every route that declares it.

    An option can belong to more than one route: `--vampire` names the same
    binary whether `--infer houdini` is putting candidates to it or `--infer
    vampire` is asking it to derive one. Declared on both rows, because the
    row is what `--help` groups by and what the "only meaningful with" check
    reads -- and registered once here, because argparse takes a flag once.

    Keyed by `dest` rather than by the flag tuple, so two rows spelling the
    same option differently would be caught here as the mistake it is.
    """
    order: list[str] = []
    seen: dict[str, tuple[Opt, list[InferRoute]]] = {}
    for route, opt in all_route_options():
        if opt.dest not in seen:
            order.append(opt.dest)
            seen[opt.dest] = (opt, [])
        first, owners = seen[opt.dest]
        if first.flags != opt.flags:
            raise Refused(
                f"--infer {route.name} declares {opt.flags[-1]} for the same "
                f"option `{opt.dest}` that --infer {owners[0].name} spells "
                f"{first.flags[-1]}: one option, one spelling"
            )
        owners.append(route)
    return tuple((seen[d][0], tuple(seen[d][1])) for d in order)


def all_route_aliases() -> tuple[tuple[InferRoute, Alias], ...]:
    """Every deprecated spelling, with the route it selects."""
    return tuple((r, a) for r in ROUTES for a in r.aliases)



def owned_files() -> tuple[str, ...]:
    """Every file, not directory, any route declares it owns, sorted.

    What an earlier run's route may have left in the project: `create_project`
    is told to remove these before this run's route writes its own.
    """
    return tuple(sorted(
        {o for r in ROUTES for o in r.owns if not o.endswith("/")}
    ))
