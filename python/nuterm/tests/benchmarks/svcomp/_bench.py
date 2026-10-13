"""Shared scaffolding for the SV-COMP DSL encodings.

Each encoding module in this package exposes one ``BENCH`` (a :class:`Bench`).
This module provides the common ``INT`` sort, a fresh-variable helper, the
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

from nuterm._bench import Bench, BuildResult, INT, var  # noqa: F401  (re-exported for the encodings)

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
