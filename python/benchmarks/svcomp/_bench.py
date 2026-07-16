"""Shared scaffolding for the SV-COMP DSL encodings.

Each encoding module in this package exposes one ``BENCH`` (a :class:`Bench`).
This module provides the common ``INT`` sort, a small wire-pair helper, the
``Bench`` record, and a ``discover`` helper that imports every sibling
encoding module and collects their ``BENCH`` objects.

See ``CONVENTIONS.md`` for the encoding rules.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

from zrth import Sort, Wire

# Scalar integer sort: every program variable is a 1x1 integer matrix.
INT = Sort.Int([1, 1])

# (module, ctrl_by_name, extl_by_name) — the return of a Bench.build()
BuildResult = tuple[object, dict[str, tuple], dict[str, tuple]]


def pair() -> tuple[Wire, Wire]:
    """A fresh ``(latched, next)`` integer wire pair."""
    return (Wire(INT), Wire(INT))


@dataclass(frozen=True)
class Bench:
    """One encoded SV-COMP benchmark.

    - ``name``   : the benchmark's canonical name (matches the .c stem).
    - ``source`` : path to the original C file (ground truth for faithfulness).
    - ``state``  : ctrl variable names, in C declaration order.
    - ``inputs`` : extl (nondeterministic input) names.
    - ``build``  : ``() -> (module, ctrl_by_name, extl_by_name)`` — builds the
                   module with fresh wires and returns name->pair maps so a
                   runner can seed Z3 symbols on the latched wires.
    - ``domain`` : ``(latched_symbols: dict[str, z3.ArithRef]) -> z3.BoolRef`` —
                   the loop condition as a Z3 predicate (the verification
                   domain). Mirrors the guard used inside ``update``.
    """

    name: str
    source: str
    state: tuple[str, ...]
    inputs: tuple[str, ...]
    build: Callable[[], BuildResult]
    domain: Callable[[dict], object]


def discover() -> list[Bench]:
    """Import every sibling encoding module and return their ``BENCH`` objects.

    Skips private modules (leading underscore), so ``_bench`` / ``_template``
    are not collected.
    """
    pkg_name = __name__.rsplit(".", 1)[0]          # this file's package
    pkg = importlib.import_module(pkg_name)
    out: list[Bench] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{pkg_name}.{info.name}")
        bench = getattr(mod, "BENCH", None)
        if isinstance(bench, Bench):
            out.append(bench)
    return out
