"""Torch interpreter for theory_pyo3 terms.

Dispatch is by `match` on the op (per-theory classes), not op-name strings.
LIA/LRA ops with identical torch semantics share a case via OR-patterns; BV
diverges (mod-2^width wraparound, bitwise ops, 1-bit-BV comparisons), so it
needs the output Sort for its width — threaded in as `out_sort`.
"""

import torch
from .zrth import LRA, LIA, BV
from .sort import BitVec, Sort


def _bv_width(sort: Sort):
    """Get bitwidth of the given `Sort` if the sort is `BV`
    (return `None` otherwise).
    """
    match sort:
        case BitVec(bw, _):
            return bw
    return None


def _wrap(t, sort):
    """Mask a BV result to its bit-width (arithmetic modulo 2^width)."""
    bw = _bv_width(sort)
    if bw is None or bw >= 64:
        return t
    return t & ((1 << bw) - 1)


def _signed(t, sort: Sort):
    """`t`, the unsigned bits of a `sort` bit-vector, read as two's
    complement. Values are stored unsigned (`_wrap`), so the signed ops have
    to read them this way: at width 4, `9` is `-7`."""
    bw = _bv_width(sort)
    if bw is None or bw >= 64:
        return t
    return torch.where(t >= (1 << (bw - 1)), t - (1 << bw), t)


_SIGNED_CMP = {BV.SLt: torch.lt, BV.SLe: torch.le,
               BV.SGt: torch.gt, BV.SGe: torch.ge}


def _nonzero(t):
    """`t` with each 0 replaced by 1: a divisor torch can divide by, for an
    op whose result at 0 is chosen separately."""
    return torch.where(t == 0, torch.ones_like(t), t)


def _shaped(t, sort: Sort):
    """`t` in `sort`'s shape. A reduction gives a 0-d tensor, and the wire it
    writes is `[1, 1]`: left 0-d, the next `Linear` reading it fails with
    "both arguments to matmul need to be at least 1D"."""
    if sort is None:
        return t
    match sort:
        case BitVec(_, shape):
            return t.reshape(*shape)
    return t.reshape(*sort._0)


def _linear(weight, bias, x):
    """A·x (+ b). An empty bias tensor means no bias (matmul-as-Linear).

    Operands are promoted to a common dtype rather than coerced to `x`'s.
    Baked Int weights meet float runtime tensors (what the promotion was
    added for), but the reverse also occurs -- `LIA.Linear` accepts float
    weights on Int wires -- and casting those down to `x` floored every
    coefficient below 1 to zero with no error.
    """
    dtype = torch.promote_types(weight.dtype, x.dtype)
    if bias is not None and bias.numel() != 0:
        dtype = torch.promote_types(dtype, bias.dtype)
        return weight.to(dtype) @ x.to(dtype) + bias.to(dtype)
    return weight.to(dtype) @ x.to(dtype)


def eval_itype(itype, read, out_sort=None, in_sorts=None):
    """Evaluate a single op with the given input tensors.

    `out_sort` is the write wire's Sort; needed for BV width-masking.
    `in_sorts` are the read wires' Sorts, which a signed BV comparison needs
    to read its operands as two's complement: its own result is one bit
    wide whatever they are.

    BV division and modulo by zero are what SMT-LIB defines them as, and
    the signed ops read their operands as two's complement, as `bvslt`,
    `bvsdiv` and `bvsmod` do and Lean's `BitVec.smod` does.
    """
    r = read
    match itype:
        # --- identity / control flow ---
        case LRA.Id() | LIA.Id() | BV.Id():
            return [r[0].clone()]
        case LRA.Ite() | LIA.Ite() | BV.Ite():
            return [torch.where(r[0].bool(), r[1], r[2])]

        # --- constants (payload bound from the op) ---
        case LRA.Bool(t) | LRA.Real(t) | LIA.Bool(t) | LIA.Int(t) | BV.Const(t):
            return [t.clone()]

        # --- arithmetic: LIA/LRA (exact, no wraparound) ---
        case LRA.Add() | LIA.Add():
            return [r[0] + r[1]]
        case LRA.Sub() | LIA.Sub():
            return [r[0] - r[1]]

        # --- arithmetic: BV (modulo 2^width) ---
        case BV.Add():
            return [_wrap(r[0] + r[1], out_sort)]
        case BV.Sub():
            return [_wrap(r[0] - r[1], out_sort)]
        case BV.Mul():
            return [_wrap(r[0] * r[1], out_sort)]
        case BV.Neg():
            return [_wrap(-r[0], out_sort)]
        case BV.Abs():
            return [_wrap(_signed(r[0], out_sort).abs(), out_sort)]
        case BV.UDiv():
            # `bvudiv s 0` is all ones.
            q = r[0].div(_nonzero(r[1]), rounding_mode="floor")
            ones = torch.full_like(q, (1 << _bv_width(out_sort)) - 1)
            return [torch.where(r[1] == 0, ones, q)]
        case BV.SDiv():
            # Truncating, on the signed values; `bvsdiv s 0` is -1 for a
            # non-negative `s` and 1 for a negative one.
            a, b = _signed(r[0], out_sort), _signed(r[1], out_sort)
            q = a.div(_nonzero(b), rounding_mode="trunc")
            by0 = torch.where(a >= 0, -torch.ones_like(a), torch.ones_like(a))
            return [_wrap(torch.where(b == 0, by0, q), out_sort)]
        case BV.UMod():
            # `bvurem s 0` is `s`.
            return [torch.where(r[1] == 0, r[0], r[0].remainder(_nonzero(r[1])))]
        case BV.SMod():
            # `bvsmod`: the sign of the divisor, as Python's `%` has it, on
            # the signed values; `bvsmod s 0` is `s`.
            a, b = _signed(r[0], out_sort), _signed(r[1], out_sort)
            m = torch.where(b == 0, a, a.remainder(_nonzero(b)))
            return [_wrap(m, out_sort)]
        case BV.MatMul():
            return [_wrap(r[0] @ r[1], out_sort)]

        # --- Linear (LIA/LRA): A, B constants carried in the op ---
        case LRA.Linear(weight, bias) | LIA.Linear(weight, bias):
            return [_linear(weight, bias, r[0])]

        # --- comparisons: LIA/LRA produce a boolean tensor ---
        case LRA.Eq() | LIA.Eq():
            return [r[0].eq(r[1])]
        case LRA.Ne() | LIA.Ne():
            return [r[0].ne(r[1])]
        case LRA.Lt() | LIA.Lt():
            return [r[0].lt(r[1])]
        case LRA.Le() | LIA.Le():
            return [r[0].le(r[1])]
        case LRA.Gt() | LIA.Gt():
            return [r[0].gt(r[1])]
        case LRA.Ge() | LIA.Ge():
            return [r[0].ge(r[1])]

        # --- comparisons: BV produce a 1-bit BV (0/1 int) ---
        # Values are stored unsigned, so the unsigned ones compare them as
        # they are and the signed ones read them as two's complement first.
        case BV.ULt():
            return [(r[0] < r[1]).to(r[0].dtype)]
        case BV.ULe():
            return [(r[0] <= r[1]).to(r[0].dtype)]
        case BV.UGt():
            return [(r[0] > r[1]).to(r[0].dtype)]
        case BV.UGe():
            return [(r[0] >= r[1]).to(r[0].dtype)]
        case BV.SLt() | BV.SLe() | BV.SGt() | BV.SGe():
            if in_sorts is None:
                raise RuntimeError(
                    f"{type(itype).__name__} compares signed values, and "
                    f"without `in_sorts` the operands' width is unknown")
            a, b = _signed(r[0], in_sorts[0]), _signed(r[1], in_sorts[1])
            return [_SIGNED_CMP[type(itype)](a, b).to(r[0].dtype)]
        case BV.Eq():
            return [(r[0] == r[1]).to(r[0].dtype)]
        case BV.Ne():
            return [(r[0] != r[1]).to(r[0].dtype)]

        # --- logical: LIA/LRA boolean ops ---
        case LRA.And() | LIA.And():
            return [r[0].logical_and(r[1])]
        case LRA.Or() | LIA.Or():
            return [r[0].logical_or(r[1])]
        case LRA.Not() | LIA.Not():
            return [r[0].logical_not()]
        case LRA.Xor() | LIA.Xor():
            return [r[0].logical_xor(r[1])]

        # --- bitwise: BV (width-preserving) ---
        case BV.And():
            return [r[0] & r[1]]
        case BV.Or():
            return [r[0] | r[1]]
        case BV.Xor():
            return [r[0] ^ r[1]]
        case BV.Not():
            return [_wrap(~r[0], out_sort)]
        case BV.BVToBool():
            return [(r[0] != 0).to(r[0].dtype)]
        case BV.BitSelect(high=high, low=low):
            return [(r[0] >> low) & ((1 << (high - low + 1)) - 1)]
        case BV.Extend(extra=_):
            return [r[0]]  # zero-extend: value unchanged, width grows

        # --- neural / aggregate (LIA/LRA) ---
        case LRA.ReLU() | LIA.ReLU():
            return [r[0].relu()]
        case LRA.Argmax() | LIA.Argmax():
            return [_shaped(r[0].argmax(), out_sort)]
        # Reductions, not elementwise binaries: the theory checks Min/Max as
        # one read to a single-element write (`check_mat_ops`, and its own
        # `min_ok` asserts int(4,1) -> int(1,1)), the same shape as Argmax.
        case LRA.Min() | LIA.Min():
            return [_shaped(r[0].min(), out_sort)]
        case LRA.Max() | LIA.Max():
            return [_shaped(r[0].max(), out_sort)]

        # --- matrix ---
        case LRA.Transpose() | LIA.Transpose():
            return [r[0].transpose(0, 1)]

        # --- uninterpreted ---
        case LRA.Uninterpreted(name) | LIA.Uninterpreted(name) | BV.Uninterpreted(name):
            raise RuntimeError(f"cannot evaluate uninterpreted function '{name}'")

    raise RuntimeError(f"cannot evaluate op: {itype}")


# ============================================================================
# Interpreter helpers (shared by zrth.gym.Wrapper, zrth.gym.Env)
# ============================================================================


def eval_term(term, state):
    """Evaluate `term` on the wires in `state`, writing its results back.

    Only a signed comparison is handed its operands' sorts: reading them is
    a call into the bindings per wire, and paid on every term it is a
    quarter of a step."""
    itype = term.itype
    read = [state[w] for w in term.read]
    out_sort = term.write[0].dtype if len(term.write) else None
    in_sorts = ([w.dtype for w in term.read]
                if type(itype) in _SIGNED_CMP else None)
    for w, val in zip(term.write, eval_itype(itype, read, out_sort, in_sorts)):
        state[w] = val


def _execute_block(state, atoms, get_block):
    """Evaluate a block from each atom."""
    for atom in atoms:
        for term in get_block(atom):
            eval_term(term, state)


def execute_init(state, atoms):
    """Evaluate the init block of all atoms."""
    _execute_block(state, atoms, lambda a: a.init)


def execute_update(state, atoms):
    """Evaluate the update block of all atoms."""
    _execute_block(state, atoms, lambda a: a.update)


def read_wire(state, wire):
    """Read a wire value from state."""
    return state[wire].detach().clone()


def getattr_wire(self, name):
    """__getattr__ helper for named wire access."""
    wire_names = object.__getattribute__(self, "_wire_names")
    if name in wire_names:
        state = object.__getattribute__(self, "_state")
        wire = wire_names[name]  # read from latched wire
        if wire in state:
            val = state[wire]
            return val.item() if val.numel() == 1 else val.detach().clone()
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
