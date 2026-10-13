"""The benchmark record and the scalar-state vocabulary the procedure works on.

A :class:`Bench` describes one program as a module with named state and input
variables; the trainer and the termination check consume it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from zrth import Int, Var

# Scalar integer sort: every program variable is a 1x1 integer matrix.
INT = Int([1, 1])

# (module, ctrl_by_name, extl_by_name), the return of a Bench.build()
BuildResult = tuple[object, dict[str, Var], dict[str, Var]]


def var() -> Var:
    """A fresh scalar integer state variable.

    A :class:`zrth.Var` stands for its latched wire and carries its next wire,
    reached by ``X(v)`` — so one variable is both ends of a round."""
    return Var(INT)


@dataclass(frozen=True)
class Bench:
    """One encoded SV-COMP benchmark.

    - ``name``   : the benchmark's canonical name (matches the .c stem).
    - ``source`` : filename of the original C program (informational; the
                   faithfulness checker takes the C file as an argument).
    - ``state``  : ctrl variable names, in C declaration order.
    - ``inputs`` : extl (nondeterministic input) names.
    - ``build``  : ``() -> (module, ctrl_by_name, extl_by_name)``, which builds
                   the module with fresh variables and returns name->``Var`` maps so a
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


def bench_of_module(module, name: str, source: str = "") -> Bench:
    """A :class:`Bench` for a ready-made module: its ctrl variables are named
    ``s0..``, its extl variables ``e0..``, in declaration order."""
    ctrl = {f"s{i}": v for i, v in enumerate(module.ctrl)}
    extl = {f"e{i}": v for i, v in enumerate(module.extl)}
    return Bench(name=name, source=source, state=tuple(ctrl), inputs=tuple(extl),
                 build=lambda: (module, ctrl, extl))
