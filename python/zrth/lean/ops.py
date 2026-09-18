"""One row per op variant, one column per backend: the op table.

`itype_name` turns a theory op into a variant name (`LIA.Ne()` -> `"Ne"`),
and four backends turn that name into something they can emit:

| column   | backend                          | a cell holds                 |
|----------|----------------------------------|------------------------------|
| `mat`    | `native._translate_terms`        | args -> Lean expr (on `Mat`) |
| `scalar` | `native._translate_terms_scalar` | args -> Lean expr (bare)     |
| `box`    | `circ._translate_terms_circ`     | a `Box` combinator name      |
| `smt`    | `smt_encode.translate_terms`     | `SmtOp` -> `cvc5.Term`       |

These were six dicts and an if/elif chain — one per backend, doubled for the
BV forms — with nothing relating them. A test can see a *dead* key (one no
theory defines) but not a *missing* one, so a variant the SMT encoder
handled and no Lean emitter did went unnoticed: every unsigned and signed BV
comparison passed `--pre-check` and `--infer` and then died at project
generation in `No Lean expression mapping for: ULt`.

A row is one variant of one or more theories, and it must say something
about every column. A cell is either an emitter or one of

* `ViaMat(...)` — scalar column only: the matrix emitter, with scalar reads
  lifted to 1x1 and `0 0` projected back off the result;
* `Inline(...)` — the backend emits this op from a branch of its own,
  because the argument strings are not enough (constants need the registry,
  `Argmax` and `Linear` need shapes and the baked tensors);
* `Unsupported(...)` — no emitter, and the reason it is missing rather than
  forgotten. That reason is what the caller is told.

`python -m zrth.lean.ops` prints the matrix, gaps and all.
`tests/test_lean_ops.py` checks it against `LIA`/`LRA`/`BV` in both
directions, so a variant added to a theory fails the suite until it has a
row, and a row no theory backs fails it too.

This module never imports cvc5, on purpose: `native.py` and `circ.py`
import *it*, and the Lean path has to keep working where cvc5 is absent
(see `magic/cegar.py`'s guarded import). The SMT emitters reach cvc5 only
through the `SmtOp` context `smt_encode` hands them — `op.K` is
`cvc5.Kind`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Union

from zrth import Bool, Int, Real, BitVec

from .common import Refused, itype_name

if TYPE_CHECKING:  # pragma: no cover - typing only, cvc5 is not imported here
    import cvc5

    from .smt_encode import SmtOp


# ══════════════════════════════════════════════════════════════════════
#  Cells
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Unsupported:
    """No emitter for this (variant, backend), and why not.

    The reason reaches the user: it is appended to the error the backend
    raises, so "no mapping" stops being the whole message.
    """

    reason: str


@dataclass(frozen=True)
class Inline:
    """The backend emits this op from a dedicated branch, not from a table.

    `site` names that branch. These ops need more than the argument
    strings — the constant registry, the read wire's shape, the tensors
    baked into the op — so a table cell could not hold them.
    """

    site: str


@dataclass(frozen=True)
class ViaMat:
    """Scalar column only: there is no scalar form, the matrix one serves.

    `_translate_terms_scalar` lifts scalar-bound reads to `fun _ _ => x`
    and projects `0 0` off the result, which is right whenever the matrix
    form is elementwise or folds to a 1x1.
    """

    note: str


MatCell = Union[Callable[[list[str]], str], Inline, Unsupported]
ScalarCell = Union[Callable[[list[str]], str], ViaMat, Inline, Unsupported]
BoxCell = Union[str, Inline, Unsupported]
SmtCell = Union[Callable[["SmtOp"], "cvc5.Term"], Unsupported]


# ══════════════════════════════════════════════════════════════════════
#  Rows
# ══════════════════════════════════════════════════════════════════════

THEORIES = ("LIA", "LRA", "BV")

ALL = THEORIES
LIA_LRA = ("LIA", "LRA")
BV_ONLY = ("BV",)
# No theory defines these any more; the emitters are kept because removal is
# the open half of KNOWN_ISSUES #23. Dispatch goes through `itype_name`, so
# nothing can reach them.
NONE: tuple[str, ...] = ()


@dataclass(frozen=True)
class Op:
    """One variant, as every backend sees it."""

    name: str
    theories: tuple[str, ...]
    mat: MatCell
    scalar: ScalarCell
    box: BoxCell
    smt: SmtCell


# ══════════════════════════════════════════════════════════════════════
#  SMT emitters
#
#  Everything cvc5 arrives through `op` (see `smt_encode.SmtOp`), including
#  the `Kind` enum as `op.K`.
# ══════════════════════════════════════════════════════════════════════


def _smt_id(op: SmtOp) -> cvc5.Term:
    return op.args[0]


def _smt_const(op: SmtOp) -> cvc5.Term:
    """LIA.Int/Bool, LRA.Real/Bool, BV.Const — scalar or per element."""
    payload = op.term.itype._0
    numel = payload.numel() if hasattr(payload, "numel") else 1
    if numel > 1:
        # matrix constant: materialize per element (payload matches shape)
        return op.tensor(op.write, payload)
    # BV.Const's element type isn't encoded in the name (unlike
    # LIA.Bool/LIA.Int/LRA.Real) — read it off the write wire.
    elem = (
        op.write.dtype
        if op.name == "Const"
        else {"Bool": Bool([1, 1]), "Int": Int([1, 1]), "Real": Real([1, 1])}[op.name]
    )
    v = op.const(elem, payload)
    return op.pack([v] * op.out_shape.total)


def _smt_tensor(op: SmtOp) -> cvc5.Term:
    return op.tensor(op.write, op.term.itype._0)


def _smt_not(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_NOT if op.is_bv else op.K.NOT
    return op.elementwise(op.unop(kind), op.args[0])


def _smt_and(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_AND if op.is_bv else op.K.AND
    return op.elementwise(op.binop(kind), *op.args)


def _smt_or(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_OR if op.is_bv else op.K.OR
    return op.elementwise(op.binop(kind), *op.args)


def _smt_xor(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_XOR if op.is_bv else op.K.XOR
    return op.elementwise(op.binop(kind), *op.args)


def _smt_ite(op: SmtOp) -> cvc5.Term:
    # cond is Mat Bool 1 1 (LIA/LRA) or Mat BV<1> 1 1 (BV); extract its
    # scalar, coercing a BV<1> condition to a genuine Bool via `!= 0`.
    cond = op.scalar(0)
    cond_dtype = op.term.read[0].dtype
    if isinstance(cond_dtype, BitVec):
        cond = op.mk(op.K.DISTINCT, cond, op.bitvector(cond_dtype._0, 0))
    return op.mk(op.K.ITE, cond, op.args[1], op.args[2])


def _smt_add(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_ADD if op.is_bv else op.K.ADD
    return op.elementwise(op.binop(kind), *op.args)


def _smt_sub(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_SUB if op.is_bv else op.K.SUB
    return op.elementwise(op.binop(kind), *op.args)


def _smt_mul(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_MULT if op.is_bv else op.K.MULT
    return op.elementwise(op.binop(kind), *op.args)


def _smt_neg(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_NEG if op.is_bv else op.K.NEG
    return op.elementwise(op.unop(kind), op.args[0])


def _smt_bv_divmod(op: SmtOp) -> cvc5.Term:
    """BV is the only theory with division and modulo, and it signs both."""
    kind = {
        "UDiv": op.K.BITVECTOR_UDIV,
        "SDiv": op.K.BITVECTOR_SDIV,
        "UMod": op.K.BITVECTOR_UREM,
        "SMod": op.K.BITVECTOR_SMOD,
    }[op.name]
    return op.elementwise(op.binop(kind), *op.args)


def _smt_bv_to_bool(op: SmtOp) -> cvc5.Term:
    # Despite the name, output is BV<1> (non-zero test), not Bool.
    a = op.scalar(0)
    bw = op.term.read[0].dtype._0
    pred = op.mk(op.K.DISTINCT, a, op.bitvector(bw, 0))
    return op.pack([op.bool_or_bv1(pred, True)])


def _smt_compare(op: SmtOp) -> cvc5.Term:
    kind = {
        "Lt": op.K.LT,
        "Le": op.K.LEQ,
        "Gt": op.K.GT,
        "Ge": op.K.GEQ,
        "Eq": op.K.EQUAL,
        "Ne": op.K.DISTINCT,
    }[op.name]
    # LIA/LRA Eq/Ne write Bool; BV.Eq/BV.Ne write BV<1> (bv.rs).
    pred = op.mk(kind, op.scalar(0), op.scalar(1))
    return op.pack([op.bool_or_bv1(pred, op.is_bv)])


def _smt_bv_compare(op: SmtOp) -> cvc5.Term:
    kind = {
        "ULe": op.K.BITVECTOR_ULE,
        "ULt": op.K.BITVECTOR_ULT,
        "UGe": op.K.BITVECTOR_UGE,
        "UGt": op.K.BITVECTOR_UGT,
        "SLe": op.K.BITVECTOR_SLE,
        "SLt": op.K.BITVECTOR_SLT,
        "SGe": op.K.BITVECTOR_SGE,
        "SGt": op.K.BITVECTOR_SGT,
    }[op.name]
    # These are BV-only ops; output is BV<1>, not the native Bool cvc5
    # gives back from BITVECTOR_U*/S* comparison Kinds.
    pred = op.mk(kind, op.scalar(0), op.scalar(1))
    return op.pack([op.bool_or_bv1(pred, True)])


def _smt_min_max(op: SmtOp) -> cvc5.Term:
    return op.pack([op.reduce_flat(0, take_min=op.name == "Min")])


def _smt_matmul(op: SmtOp) -> cvc5.Term:
    return op.matmul(op.args[0], op.args[1], op.in_shapes[0], op.in_shapes[1])


def _smt_linear(op: SmtOp) -> cvc5.Term:
    # Convention Y = A·X + B: A ([out,in]) and B ([out,1] or empty) are
    # baked into the op; args[0] is the single read wire X ([in,batch]).
    A_tensor = op.term.itype._0
    B_tensor = op.term.itype._1
    a_rows, a_cols = int(A_tensor.shape[0]), int(A_tensor.shape[1])
    A_shape = op.shape(a_rows, a_cols)
    A_data = A_tensor.reshape(a_rows, a_cols)
    A_term = op.pack_at(
        A_shape,
        [
            op.const(op.write.dtype, A_data[i, j].item())
            for i in range(a_rows)
            for j in range(a_cols)
        ],
    )
    ax = op.matmul(A_term, op.args[0], A_shape, op.in_shapes[0])
    if B_tensor.numel() == 0:
        return ax
    out_shape = op.out_shape
    B_data = B_tensor.reshape(int(B_tensor.shape[0]), int(B_tensor.shape[1]))
    B_term = op.pack(
        [
            op.const(op.write.dtype, B_data[i, 0].item())
            for i in range(out_shape.m)
            for _ in range(out_shape.n)
        ]
    )
    return op.elementwise(op.binop(op.K.ADD), ax, B_term)


def _smt_relu(op: SmtOp) -> cvc5.Term:
    zero = op.const(op.write.dtype, 0)
    return op.elementwise(
        lambda a: op.mk(op.K.ITE, op.mk(op.K.GEQ, a, zero), a, zero), op.args[0]
    )


def _smt_argmax(op: SmtOp) -> cvc5.Term:
    idx = op.argmax_flat(0)
    # `argmax_flat` builds an Int index; LRA modules carry it on a Real wire
    if isinstance(op.write.dtype, Real):
        idx = op.mk(op.K.TO_REAL, idx)
    return op.pack([idx])


def _smt_tensor_get(op: SmtOp) -> cvc5.Term:
    return op.pack([op.scalar(0)])


def _smt_to_unsigned(op: SmtOp) -> cvc5.Term:
    a = op.scalar(0)
    zero = op.integer(0)
    return op.pack([op.mk(op.K.ITE, op.mk(op.K.GEQ, a, zero), a, zero)])


# ══════════════════════════════════════════════════════════════════════
#  Reasons shared by several rows
# ══════════════════════════════════════════════════════════════════════

# The three Lean columns share one reason whenever a variant is missing from
# all of them; only the matrix cell spells the reason out.
_SEE_MAT = Unsupported("see the matrix column")

_NONDET = Unsupported(
    "a nondeterministic value (nullary, one written wire — `check_havoc` in "
    "theory/src/lib.rs). Every encoding here is a function of the read "
    "wires, so there is nowhere to put the choice."
)

_UNINTERPRETED = Unsupported(
    "an uninterpreted symbol has no obvious translation — emit an `opaque` "
    "nothing can prove anything about, or reject it at the front end. "
    "KNOWN_ISSUES #27 holds the open decision; `--fbk-proveit` already "
    "takes the second option."
)

_ZEROGRAD = Unsupported(
    "the zero of the `Differential` trait (`theory/src/lra.rs`, marked "
    "unstable), used for gradients rather than for a transition. No "
    "encoding has needed it."
)


def _bv_only_in_smt(lean_form: str) -> Unsupported:
    """A BV op cvc5 encodes and no Lean backend emits.

    This is the divergence that made the table necessary: `--pre-check` and
    `--infer` run on the SMT encoding and pass, and generation then fails.
    """
    return Unsupported(
        f"encoded for SMT, never for Lean. {lean_form} A module using it "
        "passes --pre-check and fails at project generation."
    )


# ══════════════════════════════════════════════════════════════════════
#  The table
# ══════════════════════════════════════════════════════════════════════

OPS: tuple[Op, ...] = (
    # ── structural ────────────────────────────────────────────────────
    Op("Id", ALL, mat=lambda a: a[0], scalar=lambda a: a[0], box="Box.id", smt=_smt_id),
    # ── constants ─────────────────────────────────────────────────────
    # The element type is in the variant name for LIA/LRA and on the write
    # wire for BV; scalar-vs-matrix is decided by the wire's shape.
    *(
        Op(
            name,
            theories,
            mat=Inline("common._constant_expr"),
            scalar=Inline("native._constant_expr_scalar"),
            box=Inline("circ: @Box.const"),
            smt=_smt_const,
        )
        for name, theories in (
            ("Bool", LIA_LRA),
            ("Int", ("LIA",)),
            ("Real", ("LRA",)),
            ("Const", BV_ONLY),
        )
    ),
    # ── Boolean forms (LIA/LRA) ───────────────────────────────────────
    # Operands and results are all `Mat _ 1 1`, so the matrix forms extract
    # position `0 0`. FIXME: element-wise ops should be defined over the
    # whole matrix, to match Rust.
    Op(
        "Not",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => !({a[0]} 0 0))",
        scalar=lambda a: f"(!{a[0]})",
        box="Box.not",
        smt=_smt_not,
    ),
    Op(
        "And",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => ({a[0]} 0 0 && {a[1]} 0 0))",
        scalar=lambda a: f"({a[0]} && {a[1]})",
        box="Box.and",
        smt=_smt_and,
    ),
    Op(
        "Or",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => ({a[0]} 0 0 || {a[1]} 0 0))",
        scalar=lambda a: f"({a[0]} || {a[1]})",
        box="Box.or",
        smt=_smt_or,
    ),
    Op(
        "Xor",
        LIA_LRA,
        mat=Unsupported(
            "no Lean form; nothing has asked for one. The NA route does not "
            "need it — it prints the `smt_encode` term through "
            "`smt_to_lean_bool` as `!(a == b)` (tests/test_lean_fbk.py)."
        ),
        scalar=_SEE_MAT,
        box=_SEE_MAT,
        smt=_smt_xor,
    ),
    Op(
        "Ite",
        LIA_LRA,
        mat=lambda a: f"(if {a[0]} 0 0 then {a[1]} else {a[2]})",
        scalar=lambda a: f"(if {a[0]} then {a[1]} else {a[2]})",
        box="Box.ite",
        smt=_smt_ite,
    ),
    # ── BitVec forms ──────────────────────────────────────────────────
    # BV has no Bool wires at all: an `Ite` condition and an `Eq` result are
    # both `BitVec 1`. The Boolean forms above do not apply (`!`, `&&`,
    # `||` and a bare `if c then` were emitted against `BitVec 1` values,
    # which does not elaborate). These use BitVec's own operators and
    # compare a 1-bit value against 1 directly, rather than converting
    # through `BV.BVToBool`. `Box.and`/`not`/`or` are likewise fixed to
    # `Mat Bool 1 1` — see the note beside the `bv*` boxes in Core/Box.lean.
    Op(
        "Not",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ~~~({a[0]} i j))",
        scalar=lambda a: f"(~~~{a[0]})",
        box="Box.bvNot",
        smt=_smt_not,
    ),
    Op(
        "And",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ({a[0]} i j) &&& ({a[1]} i j))",
        scalar=lambda a: f"({a[0]} &&& {a[1]})",
        box="Box.bvAnd",
        smt=_smt_and,
    ),
    Op(
        "Or",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ({a[0]} i j) ||| ({a[1]} i j))",
        scalar=lambda a: f"({a[0]} ||| {a[1]})",
        box="Box.bvOr",
        smt=_smt_or,
    ),
    Op(
        "Xor",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ({a[0]} i j) ^^^ ({a[1]} i j))",
        scalar=lambda a: f"({a[0]} ^^^ {a[1]})",
        box="Box.bvXor",
        smt=_smt_xor,
    ),
    Op(
        "Ite",
        BV_ONLY,
        mat=lambda a: f"(if {a[0]} 0 0 = 1 then {a[1]} else {a[2]})",
        scalar=lambda a: f"(if {a[0]} = 1 then {a[1]} else {a[2]})",
        box="Box.bvIte",
        smt=_smt_ite,
    ),
    # ── arithmetic ────────────────────────────────────────────────────
    Op(
        "Add",
        ALL,
        mat=lambda a: f"({a[0]} + {a[1]})",
        scalar=lambda a: f"({a[0]} + {a[1]})",
        box="Box.add",
        smt=_smt_add,
    ),
    Op(
        "Sub",
        ALL,
        mat=lambda a: f"({a[0]} - {a[1]})",
        scalar=lambda a: f"({a[0]} - {a[1]})",
        box="Box.sub",
        smt=_smt_sub,
    ),
    Op(
        "Mul",
        BV_ONLY,
        mat=lambda a: f"({a[0]} * {a[1]})",
        scalar=lambda a: f"({a[0]} * {a[1]})",
        box="Box.mul",
        smt=_smt_mul,
    ),
    Op(
        "Neg",
        BV_ONLY,
        mat=lambda a: f"(-{a[0]})",
        scalar=lambda a: f"(-{a[0]})",
        box="Box.neg",
        smt=_smt_neg,
    ),
    Op(
        "UDiv",
        BV_ONLY,
        mat=_bv_only_in_smt(
            "The matrix form would be `BitVec.udiv`, the Box a new "
            "`Box.bvUDiv` in Core/Box.lean."
        ),
        scalar=_SEE_MAT,
        box=_SEE_MAT,
        smt=_smt_bv_divmod,
    ),
    Op(
        "SDiv",
        BV_ONLY,
        mat=_bv_only_in_smt(
            "The matrix form would be `BitVec.sdiv`, the Box a new "
            "`Box.bvSDiv` in Core/Box.lean."
        ),
        scalar=_SEE_MAT,
        box=_SEE_MAT,
        smt=_smt_bv_divmod,
    ),
    Op(
        "UMod",
        BV_ONLY,
        mat=lambda a: f"(fun i j => BitVec.umod ({a[0]} i j) ({a[1]} i j))",
        scalar=lambda a: f"(BitVec.umod {a[0]} {a[1]})",
        box="Box.bvUMod",
        smt=_smt_bv_divmod,
    ),
    Op(
        "SMod",
        BV_ONLY,
        mat=lambda a: f"(fun i j => BitVec.smod ({a[0]} i j) ({a[1]} i j))",
        scalar=lambda a: f"(BitVec.smod {a[0]} {a[1]})",
        box="Box.bvSMod",
        smt=_smt_bv_divmod,
    ),
    # ── comparisons ───────────────────────────────────────────────────
    Op(
        "Lt",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 < {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} < {a[1]}))",
        box="Box.lt",
        smt=_smt_compare,
    ),
    Op(
        "Le",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 ≤ {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} ≤ {a[1]}))",
        box="Box.le",
        smt=_smt_compare,
    ),
    Op(
        "Gt",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[1]} 0 0 < {a[0]} 0 0))",
        scalar=lambda a: f"(decide ({a[1]} < {a[0]}))",
        box="Box.gt",
        smt=_smt_compare,
    ),
    Op(
        "Ge",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[1]} 0 0 ≤ {a[0]} 0 0))",
        scalar=lambda a: f"(decide ({a[1]} ≤ {a[0]}))",
        box="Box.ge",
        smt=_smt_compare,
    ),
    Op(
        "Eq",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 = {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} = {a[1]}))",
        box="Box.eq",
        smt=_smt_compare,
    ),
    Op(
        "Ne",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 ≠ {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} ≠ {a[1]}))",
        box="Box.neq",
        smt=_smt_compare,
    ),
    Op(
        "Eq",
        BV_ONLY,
        mat=lambda a: f"(fun _ _ => if {a[0]} 0 0 = {a[1]} 0 0 then 1 else 0)",
        scalar=lambda a: f"(if {a[0]} = {a[1]} then 1 else 0)",
        box="Box.bvEq",
        smt=_smt_compare,
    ),
    Op(
        "Ne",
        BV_ONLY,
        mat=lambda a: f"(fun _ _ => if {a[0]} 0 0 = {a[1]} 0 0 then 0 else 1)",
        scalar=lambda a: f"(if {a[0]} = {a[1]} then 0 else 1)",
        box="Box.bvNe",
        smt=_smt_compare,
    ),
    # BV compares unsigned and signed separately, and writes the answer to a
    # `BV<1>` wire rather than a Bool one.
    *(
        Op(
            name,
            BV_ONLY,
            mat=_bv_only_in_smt(
                "The matrix form would compare with `BitVec`'s own "
                "ordering and reify the answer as a 1-bit value, the Box a "
                "new `Box.bv*` in Core/Box.lean."
            ),
            scalar=_SEE_MAT,
            box=_SEE_MAT,
            smt=_smt_bv_compare,
        )
        for name in ("ULt", "ULe", "UGt", "UGe", "SLt", "SLe", "SGt", "SGe")
    ),
    # ── reductions and matrix ops ─────────────────────────────────────
    # `Min`/`Max` are unary reductions — one read wire of any shape, a
    # `Mat t 1 1` written — folded row-major from element (0,0).
    Op(
        "Min",
        LIA_LRA,
        mat=lambda a: f"(matMin {a[0]})",
        scalar=ViaMat("`matMin` folds a matrix; there is no bare-scalar fold"),
        box="Box.min",
        smt=_smt_min_max,
    ),
    Op(
        "Max",
        LIA_LRA,
        mat=lambda a: f"(matMax {a[0]})",
        scalar=ViaMat("`matMax` folds a matrix; there is no bare-scalar fold"),
        box="Box.max",
        smt=_smt_min_max,
    ),
    Op(
        "MatMul",
        BV_ONLY,
        mat=lambda a: f"MatMul {a[0]} {a[1]}",
        scalar=lambda a: f"({a[0]} * {a[1]})",
        box="Box.mul",
        smt=_smt_matmul,
    ),
    Op(
        "Transpose",
        LIA_LRA,
        mat=lambda a: f"(MatTranspose {a[0]})",
        scalar=ViaMat("`MatTranspose` is a matrix rearrangement"),
        box="Box.transpose",
        smt=Unsupported(
            "no cvc5 term. A transpose is a permutation of the flat tuple "
            "and could be encoded; nothing has needed it, and "
            "`check_na_supported` reports the refusal by name."
        ),
    ),
    Op(
        "ReLU",
        LIA_LRA,
        mat=lambda a: f"ReLu {a[0]}",
        # `ReLu` is a matrix op; on a scalar-bound wire it has to be the
        # bare `max 0 x`. Lifting the operand and projecting the result
        # instead leaves `ReLu (fun _ _ => x) 0 0` with unconstrained
        # dimensions, so this is not a `ViaMat` cell.
        scalar=lambda a: f"(Max.max 0 {a[0]})",
        box="Box.relu",
        smt=_smt_relu,
    ),
    # `Linear`'s A and B are baked into the op rather than read from wires,
    # and `Argmax` picks its form from the read wire's shape — neither fits
    # a cell taking argument strings.
    Op(
        "Linear",
        LIA_LRA,
        mat=Inline("native._linear_expr"),
        scalar=Inline("native._linear_expr, lifted and projected"),
        box=Inline("circ._linear_box"),
        smt=_smt_linear,
    ),
    Op(
        "Argmax",
        LIA_LRA,
        mat=Inline("native._argmax_expr"),
        scalar=Inline("native: the `argmax1d_scalar_*` axiom, else _argmax_expr"),
        box=Inline("circ._argmax_box"),
        smt=_smt_argmax,
    ),
    # ── bit-width and conversion ──────────────────────────────────────
    Op(
        "BVToBool",
        BV_ONLY,
        mat=_bv_only_in_smt(
            "Despite the name it writes a `BV<1>` (a non-zero test), which "
            "the Lean BV forms get by comparing against 1 inline."
        ),
        scalar=_SEE_MAT,
        box=_SEE_MAT,
        smt=_smt_bv_to_bool,
    ),
    *(
        Op(
            name,
            BV_ONLY,
            mat=Unsupported(reason),
            scalar=_SEE_MAT,
            box=_SEE_MAT,
            smt=Unsupported(reason),
        )
        for name, reason in (
            (
                "BitSelect",
                "width-changing: `BitSelect[high:low]` writes a narrower "
                "wire than it reads (theory/src/bv.rs). No backend emits it.",
            ),
            (
                "Extend",
                "width-changing: `Extend(+extra)` writes a wider wire than "
                "it reads (theory/src/bv.rs). No backend emits it.",
            ),
            (
                "Abs",
                "modelled as `Ite(x < 0, Neg x, x)` in theory/src/bv.rs, "
                "which leaves `Abs(BV_MIN)` open there (XXX). No backend "
                "emits it.",
            ),
        )
    ),
    # ── nondeterminism and symbols ────────────────────────────────────
    *(
        Op(
            name,
            theories,
            mat=reason,
            scalar=_SEE_MAT,
            box=_SEE_MAT,
            smt=reason,
        )
        for name, theories, reason in (
            ("AnyBool", LIA_LRA, _NONDET),
            ("AnyInt", ("LIA",), _NONDET),
            ("AnyReal", ("LRA",), _NONDET),
            ("Havoc", BV_ONLY, _NONDET),
            ("Uninterpreted", ALL, _UNINTERPRETED),
            ("BoolZerograd", ("LRA",), _ZEROGRAD),
            ("RealZerograd", ("LRA",), _ZEROGRAD),
        )
    ),
    # ── dead ──────────────────────────────────────────────────────────
    # `IType`-era names no theory defines any more. Kept because removing
    # them is the decision KNOWN_ISSUES #23 asks for; unreachable either
    # way, since dispatch goes through `itype_name`.
    Op(
        "Tensor",
        NONE,
        mat=Inline("common._constant_expr"),
        scalar=Inline("native._constant_expr_scalar"),
        box=Inline("circ: @Box.const"),
        smt=_smt_tensor,
    ),
    Op(
        "TensorGet",
        NONE,
        mat=lambda a: f"({a[0]} 0 0)",
        scalar=lambda a: a[0],
        box=Unsupported("never had a Box"),
        smt=_smt_tensor_get,
    ),
    Op(
        "ToUnsigned",
        NONE,
        mat=lambda a: f"(fun _ _ => Int.toNat ({a[0]} 0 0))",
        scalar=lambda a: f"(Int.toNat {a[0]})",
        box=Unsupported("never had a Box"),
        smt=_smt_to_unsigned,
    ),
)


# Rows are keyed by (theory, variant); a dead row answers to no theory and
# is keyed by ("", name), which is what `theory_of` returns for an op
# carrying no theory prefix.
_BY_KEY: dict[tuple[str, str], Op] = {}
for _op in OPS:
    for _theory in _op.theories or ("",):
        _key = (_theory, _op.name)
        if _key in _BY_KEY:
            raise RuntimeError(f"two op-table rows for {_theory}.{_op.name}")
        _BY_KEY[_key] = _op
del _op, _theory, _key


# ══════════════════════════════════════════════════════════════════════
#  Lookup
# ══════════════════════════════════════════════════════════════════════


def theory_of(itype) -> str:
    """The theory an op came from: "LIA", "LRA" or "BV".

    PyO3 exports the variants as `LIA_Add`, `LRA_Real`, `BV_MatMul`; an op
    with no such prefix belongs to no theory and gets "".
    """
    name = type(itype).__name__
    for theory in THEORIES:
        if name.startswith(theory + "_"):
            return theory
    return ""


def op_for(itype) -> Op:
    """The table row for this op, or a `ValueError` naming what is missing."""
    theory = theory_of(itype)
    name = itype_name(itype)
    row = _BY_KEY.get((theory, name))
    if row is None:
        raise ValueError(
            f"no op-table row for {theory or '<no theory>'}.{name}; "
            "add one in zrth/lean/ops.py"
        )
    return row


def _reason(row: Op, cell) -> str:
    """What a non-emitter cell has to say for itself.

    `_SEE_MAT` carries no reason of its own: the matrix cell holds the one
    that covers all three Lean columns.
    """
    if cell is _SEE_MAT:
        cell = row.mat
    if isinstance(cell, Unsupported):
        return cell.reason
    if isinstance(cell, Inline):
        return f"handled outside the table, by {cell.site}"
    return f"the matrix form serves it ({cell.note})"


def _lean_emitter(itype, row: Op, cell, column: str):
    """Unwrap a Lean cell, or raise carrying the reason the table records."""
    if isinstance(cell, (Unsupported, Inline, ViaMat)):
        raise Refused(
            f"No Lean expression mapping for: {itype_name(itype)} "
            f"({column}) — {_reason(row, cell)}"
        )
    return cell


def mat_emitter(itype) -> Callable[[list[str]], str]:
    """The `Mat`-form Lean emitter, or a `ValueError` carrying the reason."""
    row = op_for(itype)
    return _lean_emitter(itype, row, row.mat, "matrix form")


def scalar_emitter(itype) -> "Callable[[list[str]], str] | None":
    """The bare-scalar Lean emitter, or None when the matrix form serves.

    None covers `ViaMat` (deliberate), `Inline` (the caller has its own
    branch) and `Unsupported` alike: in every case the scalar printer falls
    through to the matrix column, which raises there if it has to.
    """
    cell = op_for(itype).scalar
    if isinstance(cell, (Unsupported, Inline, ViaMat)):
        return None
    return cell


def box_for(itype) -> str:
    """The `Box` combinator, or a `ValueError` carrying the reason."""
    row = op_for(itype)
    return _lean_emitter(itype, row, row.box, "Box")


def smt_emitter(itype) -> Callable[[SmtOp], cvc5.Term]:
    """The cvc5 emitter, or a `ValueError` carrying the reason.

    The SMT column holds no `Inline` or `ViaMat` cells: `translate_terms`
    dispatches every variant through the table, constants included.
    """
    cell = op_for(itype).smt
    if isinstance(cell, Unsupported):
        raise ValueError(
            f"SMT translator: unsupported IType {itype_name(itype)} — {cell.reason}"
        )
    return cell


# ══════════════════════════════════════════════════════════════════════
#  The matrix, printed
# ══════════════════════════════════════════════════════════════════════

COLUMNS = ("mat", "scalar", "box", "smt")


def cell_of(op: Op, column: str):
    return getattr(op, column)


def cell_mark(cell) -> str:
    """One-word rendering of a cell, for the printed matrix."""
    if isinstance(cell, Unsupported):
        return "-"
    if isinstance(cell, Inline):
        return "inline"
    if isinstance(cell, ViaMat):
        return "via mat"
    return "yes"


def rows() -> list[tuple[str, Op]]:
    """(theory, row) for every variant, theory-major then alphabetical."""
    order = {t: i for i, t in enumerate(THEORIES)}
    order[""] = len(order)
    return sorted(
        ((theory, op) for (theory, _), op in _BY_KEY.items()),
        key=lambda pair: (order[pair[0]], pair[1].name),
    )


def format_matrix() -> str:
    """The whole table as text: one line per variant, then every gap.

    Gaps are grouped by reason — the same one covers several variants, and
    reading it once per group is the point of having it written down.
    """
    head = f"{'theory':7} {'op':14} " + " ".join(f"{c:8}" for c in COLUMNS)
    lines = [head, "-" * len(head)]
    gaps: dict[str, list[str]] = {}
    for theory, op in rows():
        marks = " ".join(f"{cell_mark(cell_of(op, c)):8}" for c in COLUMNS)
        lines.append(f"{theory or '(dead)':7} {op.name:14} {marks}")
        for column in COLUMNS:
            cell = cell_of(op, column)
            if isinstance(cell, Unsupported) and cell is not _SEE_MAT:
                gaps.setdefault(cell.reason, []).append(
                    f"{theory or '(dead)'}.{op.name} [{column}]"
                )
    lines.append("\ngaps:")
    for reason, sites in gaps.items():
        lines.append(f"  * {', '.join(sites)}")
        lines.append(f"      {reason}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(format_matrix())
