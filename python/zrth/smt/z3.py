"""Translate theory_pyo3 terms to Z3 expressions.

Each wire value is a list of Z3 expressions (the flattened tensor). Dispatch is
by `match` on the op; LIA/LRA share cases. BV -> Z3 (BitVec) is not implemented
yet (BV ops fall through to a clear error).
"""

import z3
import torch
from ..zrth import LRA, LIA, BV


def _tensor_to_z3(tensor):
    """Convert a torch.Tensor to a list of Z3 constants (sort chosen by dtype)."""
    flat = tensor.detach().flatten().tolist()
    if tensor.dtype == torch.bool:
        return [z3.BoolVal(bool(v)) for v in flat]
    elif tensor.dtype in (torch.long, torch.int, torch.int32, torch.int64):
        return [z3.IntVal(int(v)) for v in flat]
    else:
        return [z3.RealVal(str(v)) for v in flat]


def _to_arith(vals):
    """Coerce Z3 expressions to arithmetic (Bool -> If(b, 1, 0))."""
    return [z3.If(v, z3.RealVal(1), z3.RealVal(0)) if z3.is_bool(v) else v for v in vals]


def _to_bool(vals):
    """Coerce Z3 expressions to Bool (arith -> val != 0)."""
    return [v if z3.is_bool(v) else v != z3.RealVal(0) for v in vals]


def _z3_linear(weight, bias, read):
    """A·x (+ b) for constant tensors `weight` [out, in] and `bias`."""
    x = _to_arith(read[0])
    w = _tensor_to_z3(weight)  # flattened row-major
    n_out, n_in = int(weight.shape[0]), int(weight.shape[1])
    has_bias = bias is not None and bias.numel() > 0
    b = _to_arith(_tensor_to_z3(bias)) if has_bias else None
    result = []
    for i in range(n_out):
        s = z3.Sum([w[i * n_in + j] * x[j] for j in range(n_in)]) if n_in else z3.RealVal(0)
        if has_bias:
            s = s + b[i]
        result.append(s)
    return [result]


def _z3_ite(read):
    cond, then_branch, else_branch = read[0], read[1], read[2]
    c = _to_bool(cond)[0]
    if any(z3.is_bool(v) for v in then_branch) or any(z3.is_bool(v) for v in else_branch):
        if all(z3.is_bool(v) for v in then_branch) and all(z3.is_bool(v) for v in else_branch):
            return [z3.If(c, t, f) for t, f in zip(then_branch, else_branch)]
        then_branch = _to_arith(then_branch)
        else_branch = _to_arith(else_branch)
    return [z3.If(c, t, f) for t, f in zip(then_branch, else_branch)]


def eval(itype, read):
    """Translate a single op to Z3 expressions."""
    r = read
    match itype:
        case LRA.Id() | LIA.Id():
            return [r[0]]

        case LRA.Const(t) | LIA.Const(t):
            return [_tensor_to_z3(t)]

        # arithmetic
        case LRA.Add() | LIA.Add():
            return [[a + b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
        case LRA.Sub() | LIA.Sub():
            return [[a - b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
        case LRA.Linear(weight, bias) | LIA.Linear(weight, bias):
            return _z3_linear(weight, bias, r)

        # comparisons (produce Bool)
        case LRA.Eq() | LIA.Eq():
            return [[a == b for a, b in zip(r[0], r[1])]]
        case LRA.Ne() | LIA.Ne():
            return [[a != b for a, b in zip(r[0], r[1])]]
        case LRA.Lt() | LIA.Lt():
            return [[a < b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
        case LRA.Le() | LIA.Le():
            return [[a <= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
        case LRA.Gt() | LIA.Gt():
            return [[a > b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
        case LRA.Ge() | LIA.Ge():
            return [[a >= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]

        # logical
        case LRA.And() | LIA.And():
            return [[z3.And(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]
        case LRA.Or() | LIA.Or():
            return [[z3.Or(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]
        case LRA.Not() | LIA.Not():
            return [[z3.Not(a) for a in _to_bool(r[0])]]
        case LRA.Xor() | LIA.Xor():
            return [[z3.Xor(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]

        # control flow
        case LRA.Ite() | LIA.Ite():
            return [_z3_ite(r)]

        # neural-ish / aggregate
        case LRA.ReLU() | LIA.ReLU():
            return [[z3.If(x > 0, x, z3.RealVal(0)) for x in _to_arith(r[0])]]
        case LRA.Min() | LIA.Min():
            return [[z3.If(a <= b, a, b) for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
        case LRA.Max() | LIA.Max():
            return [[z3.If(a >= b, a, b) for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]

        # matrix: wire values are flat expression lists, so for the vector shapes
        # the Linear path transposes (1xn <-> nx1) this is identity on the flat form.
        case LRA.Transpose() | LIA.Transpose():
            return [r[0]]

        case LRA.Uninterpreted(name) | LIA.Uninterpreted(name) | BV.Uninterpreted(name):
            raise RuntimeError(f"cannot translate uninterpreted '{name}' to Z3")

    raise RuntimeError(f"cannot translate op to Z3: {itype} (BV -> Z3 BitVec is TODO)")
