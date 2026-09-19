"""The verifier's view of a composed module, indexed by its non-affine nodes.

A module is already a graph: each term carries an ``itype`` and its read and
write wires. This takes that graph from wire space into solver space, keeping the
structure the cell engine needs.

:func:`node_view` walks the terms once and does two things. It classifies each
term as affine or piecewise-linear, since a piecewise-linear one needs a case
split or a relaxation. And it partially evaluates: an affine term is folded into
a z3 expression, while a piecewise-linear one keeps a symbol standing for its
output.

A wire's value therefore comes back affine in the columns and the node symbols,
and the nodes say what each symbol means. ``ReLU`` and ``Ite`` are the kinds the
engine splits; :data:`._farkas.OPS` is where a kind is declared.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

    ``kind`` is ``None`` for an operation the walk evaluates straight through, and
    affine arithmetic, or boolean structure z3 handles directly. Otherwise it names
    a piecewise-linear kind, and exactly one of:

      * ``mode``, a :class:`ModeKind`, so the kind can be *pinned* into a cell;
      * ``split``, so the kind is case-split away before the LP is built.

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

    ``values`` is each wire evaluated all the way down: the flattened form, with a
    ReLU as ``If(z > 0, z, 0)``, which is what the obligation's ``s_syms``,
    ``sp_syms`` and ``V`` terms are. ``opaque`` is the same wires with each
    non-affine node's output left as a symbol, so a value downstream of one comes
    out *affine in those symbols*, which is how ``V``'s structure is read off
    without touching the weights.

    The two agree everywhere upstream of a node, so only terms below one are
    evaluated twice.

    ``entry`` is the ``init`` block evaluated the same way, giving the state's
    values at tick 0, and is empty when the walk was not asked for it."""
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
    name: the procedure says what it understands, and anything else stops here
    rather than being evaluated away or surfacing as a backend error."""
    name = type(term.itype).__name__
    op = ops.get(name)
    if op is None:
        raise Unsupported(f"itype {name!r} is not in this procedure's vocabulary")
    try:
        return op.kind, zz3.eval(term.itype, reads)
    except Exception as e:
        raise Unsupported(f"itype {name!r} has no Z3 translation: {e}")


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

    ``seed`` is ``{wire: [value]}`` for the wires the walk starts from: a module's
    latched state, or the wires it awaits. Each term is evaluated to z3 for
    ``values``; a non-affine term additionally gets a symbol for its output in
    ``opaque`` and is recorded as a :class:`Node` with its inputs. Only wires below
    a node differ between the two maps, so the second evaluation is done just for
    those.

    ``ops`` maps an itype's class name to the :class:`Op` describing it. An itype
    absent from it raises :class:`Unsupported` rather than being evaluated away.

    ``entry_seed``, when given, is the seed for a second walk over the ``init``
    block, whose values land in ``entry``. ``atoms`` restricts the walk to some of the module's atoms (see
    :func:`._farkas.reading`)."""
    values = {w: list(v) for w, v in seed.items()}
    opaque = dict(values)
    diverged, nodes = set(), []
    for atom in (module.atoms if atoms is None else atoms):
        for t_ in atom.update:
            reads = [values[w] for w in t_.read]
            kind, out = _eval(t_, reads, ops)
            values.update(zip(t_.write, out))
            if kind is not None:
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
