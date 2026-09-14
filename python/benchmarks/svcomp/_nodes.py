"""The verifier's view of a composed module, indexed by its non-affine nodes.

A module is already a graph: each term carries an ``itype`` and its
read/write wires. This takes it from wire space into solver space and keeping the structure the cell engine
needs.

:func:`node_view` walks the terms once and does two things. It classifies
which terms are affine and which are piecewise-linear, so it may need a case split or a
relaxation. And it partially evaluates: affine terms are folded into z3 expressions, while each
piecewise-linear term keeps a symbol standing for its output. 

So a wire's value comes back affine in the state and the node symbols, and the
nodes say what each symbol means. ``ReLU`` and ``Ite`` are the kinds the engine
splits today (later to be extended).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import z3

from zrth import z3 as zz3

@dataclass(frozen=True)
class ModeKind:
    """A piecewise-linear kind's cell rule: ``at`` reads which mode a witness lies
    in, ``region`` gives the constraints that mode imposes. The modes must be
    complementary, so cells partition rather than overlap."""
    at: object
    region: object


@dataclass(frozen=True)
class Op:
    """What a procedure knows about one of the theory's operations.

    ``kind`` is ``None`` for an operation the walk evaluates straight through —
    affine arithmetic, or boolean structure z3 handles directly. Otherwise it names
    a piecewise-linear kind, and exactly one of:

      * ``mode`` — a :class:`ModeKind`, so the kind can be *pinned* into a cell;
      * ``split`` — the kind is case-split away before the LP is built.

    A kind with neither is recognised but unusable, so a module containing it is
    refused rather than reasoned over. The theory publishes no semantics for its
    operations, so this classification is the procedure's own declaration of what
    it understands."""
    kind: str | None = None
    mode: object = None
    split: bool = False


@dataclass(frozen=True)
class Node:
    """One piecewise-linear node: its ``kind``, the symbol standing for its output,
    and its inputs as z3 expressions."""
    kind: str
    sym: object
    args: tuple


@dataclass(frozen=True)
class NodeView:
    """One traversal, two readings of the same wires.

    ``values`` is each wire evaluated all the way down — the flattened form, with a
    ReLU as ``If(z > 0, z, 0)`` — which is what the obligation's ``s_syms``,
    ``sp_syms`` and ``V`` terms are. ``opaque`` is the same wires with each
    non-affine node's output left as a symbol, so a value downstream of one comes
    out *affine in those symbols*, which is how ``V``'s structure is read off
    without touching the weights.

    The two agree everywhere upstream of a node, so only terms below one are
    evaluated twice.

    ``entry`` is the ``init`` block evaluated the same way — the state's values at
    tick 0 — empty when the walk was not asked for it. A module's two blocks are
    two relations, so each is walked once and both are read from here."""
    nodes: tuple
    values: dict
    opaque: dict
    entry: dict = field(default_factory=dict)

    def feeding(self, value, kind: str | None = None) -> tuple:
        """The nodes whose symbol occurs in ``value`` (an ``opaque`` expression),
        optionally of one kind."""
        names = {c.decl().name() for c in _consts(value)}
        return tuple(n for n in self.nodes
                     if n.sym.decl().name() in names
                     and (kind is None or n.kind == kind))


class Unsupported(Exception):
    """The module contains something the procedure has no rule for.

    Raised at the door rather than degraded quietly: a procedure that states its
    preconditions can assume them, and one that silently proceeds with less
    information reports a proof that will not close instead of the reason."""


def _eval(term, reads, ops):
    """``term`` evaluated to z3, as ``(kind, outputs)``.

    An itype outside ``ops``, or one the Z3 backend cannot translate, is refused by
    name — the procedure says what it understands, and anything else stops here
    rather than being evaluated away or surfacing as a backend error.

    :mod:`zrth.z3` speaks in 2-D arrays, one per wire, because a wire carries a
    tensor. Every wire this procedure reads is a scalar (:func:`._farkas.read_system`
    refuses the rest), so a value is wrapped into its 1x1 array on the way in and
    read back out of one on the way out, and the maps here hold plain expressions."""
    name = type(term.itype).__name__
    op = ops.get(name)
    if op is None:
        raise Unsupported(f"itype {name!r} is not in this procedure's vocabulary")
    try:
        out = zz3.eval(term.itype, [np.array([[r[0]]], dtype=object) for r in reads])
    except Exception as e:
        raise Unsupported(f"itype {name!r} has no Z3 translation: {e}")
    return op.kind, [[a.reshape(-1)[0]] for a in out]


def _entry_values(module, seed, ops) -> dict:
    """``module``'s init block evaluated once, from ``seed`` (its awaited inputs)."""
    values = {w: list(v) for w, v in seed.items()}
    for atom in module.atoms:
        for term in atom.init:
            _, out = _eval(term, [values[w] for w in term.read], ops)
            values.update(zip(term.write, out))
    return values


def node_view(module, seed, ops, entry_seed=None, atoms=None) -> NodeView:
    """Walk ``module``'s update block once, reading it both ways.

    ``seed`` is ``{wire: [value]}`` for the wires the walk starts from — a module's
    latched state, or the wires it awaits. Each term is evaluated to z3 for
    ``values``; a non-affine term additionally gets a symbol for its output in
    ``opaque`` and is recorded as a :class:`Node` with its inputs. Only wires below
    a node differ between the two maps, so the second evaluation is done just for
    those — everything upstream is shared.

    ``ops`` maps an itype's class name to the :class:`Op` describing it. An itype
    absent from it raises :class:`Unsupported` rather than being evaluated away —
    the procedure says what it understands, and anything else is refused. So is a
    kind the table names but has no rule for: Z3 may well read it, and then its
    output would be a symbol the engine can neither pin nor split, which is a
    proof that does not close rather than the reason it could not.

    ``entry_seed``, when given, is the seed for a second walk over the ``init``
    block, whose values land in ``entry``. ``atoms`` restricts the walk to some of
    the module's atoms — how a reading's structure is read as a function, by
    walking its atom alone with the wires it awaits seeded as symbols."""
    values = {w: list(v) for w, v in seed.items()}
    opaque = dict(values)
    diverged, nodes = set(), []
    for atom in (module.atoms if atoms is None else atoms):
        for t_ in atom.update:
            reads = [values[w] for w in t_.read]
            kind, out = _eval(t_, reads, ops)
            values.update(zip(t_.write, out))
            if kind is not None:
                op = ops[type(t_.itype).__name__]
                if not op.mode and not op.split:
                    raise Unsupported(
                        f"itype {type(t_.itype).__name__!r} has no rule in this "
                        f"procedure: its kind {kind!r} has neither a cell rule "
                        f"nor a case split")
                sym = z3.Int(f"_n{len(nodes)}")
                nodes.append(Node(kind, sym, tuple(r[0] for r in reads)))
                opaque.update(zip(t_.write, [[sym]]))
                diverged.update(t_.write)
            elif any(w in diverged for w in t_.read):
                opaque.update(zip(t_.write,
                                  _eval(t_, [opaque[w] for w in t_.read], ops)[1]))
                diverged.update(t_.write)
            else:
                opaque.update(zip(t_.write, out))     # identical upstream; reuse
    entry = ({} if entry_seed is None
             else _entry_values(module, entry_seed, ops))
    return NodeView(tuple(nodes), values, opaque, entry)


def free_symbols(e) -> set:
    """The names of the uninterpreted constants in ``e``."""
    return {c.decl().name() for c in _consts(e)}


def _consts(e):
    """Every uninterpreted constant in ``e``."""
    if z3.is_const(e):
        return [e] if e.decl().kind() == z3.Z3_OP_UNINTERPRETED else []
    out = []
    for c in e.children():
        out += _consts(c)
    return out
