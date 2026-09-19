"""Shared scaffolding for the SV-COMP DSL encodings.

Each encoding module in this package exposes one ``BENCH`` (a :class:`Bench`).
This module provides the common ``INT`` sort, a small wire-pair helper, the
``Bench`` record, and a ``discover`` helper that imports every sibling
encoding module and collects their ``BENCH`` objects.

See ``_template.py`` for the encoding conventions (copy it to add a benchmark).
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable
import torch

from zrth import Sort, Wire

# Scalar integer sort: every program variable is a 1x1 integer matrix.
INT = Sort.Int([1, 1])

# (module, ctrl_by_name, extl_by_name), the return of a Bench.build()
BuildResult = tuple[object, dict[str, tuple], dict[str, tuple]]


def pair() -> tuple[Wire, Wire]:
    """A fresh ``(latched, next)`` integer wire pair."""
    return (Wire(INT), Wire(INT))


@dataclass(frozen=True)
class Bench:
    """One encoded SV-COMP benchmark.

    - ``name``   : the benchmark's canonical name (matches the .c stem).
    - ``source`` : filename of the original C file in this package's ``c/``
                   directory (ground truth for the faithfulness checker).
    - ``state``  : ctrl variable names, in C declaration order.
    - ``inputs`` : extl (nondeterministic input) names.
    - ``build``  : ``() -> (module, ctrl_by_name, extl_by_name)``, which builds
                   the module with fresh wires and returns name->pair maps so a
                   runner can seed Z3 symbols on the latched wires. The loop
                   guard is not declared here: it is the ``update`` block's own
                   ``ite(guard, body, self)``, its single source of truth.
    - ``precondition`` : optional ``(state: dict[str, T]) -> list`` giving the
                   conjuncts of the outer ``if (P)`` entry gate, the entry-gate
                   counterpart of the loop guard. Written with plain comparison
                   ops so it works either way: on ints it yields Python bools,
                   which the equivalence checker and the trainer use to keep
                   only inputs that enter the loop; on Z3 terms it yields
                   ``BoolRef``s, asserted at entry and seeded as invariant
                   candidates.
    """

    name: str
    source: str
    state: tuple[str, ...]
    inputs: tuple[str, ...]
    build: Callable[[], BuildResult]
    precondition: Callable[[dict], list] | None = None


def discover() -> list[Bench]:
    """Import every sibling encoding module and return their ``BENCH`` objects.

    Skips private modules (leading underscore), so ``_bench`` / ``_template``
    are not collected.
    """
    pkg_name = __name__.rsplit(".", 1)[0] + ".dsl"   # the encodings live in <svcomp>.dsl
    pkg = importlib.import_module(pkg_name)
    out: list[Bench] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):                # skip _template, etc.
            continue
        mod = importlib.import_module(f"{pkg_name}.{info.name}")
        bench = getattr(mod, "BENCH", None)
        if isinstance(bench, Bench):
            out.append(bench)
    return out
