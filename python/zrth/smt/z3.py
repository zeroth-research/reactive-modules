"""Translate theory_pyo3 terms to Z3 expressions.

Each wire value is a list of Z3 expressions (the flattened tensor). Dispatch is
by `match` on the op; LIA/LRA share cases. BV -> Z3 (BitVec) is not implemented
yet (BV ops fall through to a clear error).
"""

import z3 as _z3
import torch
from ..zrth import LRA, LIA, BV, Combinatorial
from ..sort import Real, Int, Bool
import numpy as _np
import operator


def fresh(dtype, prefix):
    match dtype:
        case Real([n, m]):
            return _np.array([[_z3.FreshReal(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case Bool([n, m]):
            return _np.array([[_z3.FreshBool(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case Int([n, m]):
            return _np.array([[_z3.FreshInt(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case _:
            raise NotImplementedError(f"fresh not implemented for dtype {dtype}")


def interpret(itype):
    match itype:
        case LRA.Id() | LIA.Id():
            return lambda x: x
        case LRA.Bool(t) | LIA.Bool(t):
            return lambda: _np.vectorize(_z3.BoolVal, otypes=[object])(t.detach().numpy())
        case LRA.Real(t):
            return lambda: _np.vectorize(_z3.RealVal, otypes=[object])(t.detach().numpy())
        case LIA.Int(t):
            return lambda: _np.vectorize(_z3.IntVal, otypes=[object])(t.detach().numpy())
        case Combinatorial.HAVOC(dtype):
            return lambda: fresh(dtype, f"HAVOC")

        # arithmetic
        case LRA.Sub() | LIA.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case LRA.Add() | LIA.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case LRA.Linear(weight, bias) | LIA.Linear(weight, bias):
            return lambda x: weight.detach().numpy() @ x + bias.detach().numpy()

        # comparisons (produce Bool)
        case LRA.Eq() | LIA.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case LRA.Ne() | LIA.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case LRA.Lt() | LIA.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case LRA.Le() | LIA.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case LRA.Gt() | LIA.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case LRA.Ge() | LIA.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # logical
        case LRA.And() | LIA.And():
            return _np.frompyfunc(lambda x, y: _z3.And(x, y), 2, 1)
        case LRA.Or() | LIA.Or():
            return _np.frompyfunc(lambda x, y: _z3.Or(x, y), 2, 1)
        case LRA.Not() | LIA.Not():
            return _np.frompyfunc(lambda x: _z3.Not(x), 1, 1)
        case LRA.Xor() | LIA.Xor():
            return _np.frompyfunc(lambda x, y: _z3.Xor(x, y), 2, 1)

        # control flow
        case LRA.Ite() | LIA.Ite():
            return _np.frompyfunc(lambda c, tt, ff: _z3.If(c, tt, ff), 3, 1)

        # neural-ish / aggregate
        case LRA.ReLU() | LIA.ReLU():
            return _np.frompyfunc(lambda x: _z3.If(x > 0., x, 0.), 1, 1)
        case LRA.Min() | LIA.Min():
            return _np.frompyfunc(lambda x, y: _z3.If(x <= y, x, y), 2, 1)
        case LRA.Max() | LIA.Max():
            return _np.frompyfunc(lambda x, y: _z3.If(x >= y, x, y), 2, 1)

        case LRA.Transpose() | LIA.Transpose():
            return _np.transpose
        case LRA.Uninterpreted(name) | LIA.Uninterpreted(name) | BV.Uninterpreted(name):
            raise RuntimeError(f"cannot translate uninterpreted '{name}' to Z3")
        case _:
            raise NotImplementedError(f"itype: {itype}")


def eval(itype, read):
    """Translate a single op to Z3 expressions."""
    if not isinstance(read, list) or not all(isinstance(r, _np.ndarray) for r in read) or \
            not all(isinstance(a, _z3.AstRef) for r in read for a in r.flat):
        raise RuntimeError(f"read={read} must be a list of numpy arrays of z3 expressions")
    if not all(r.ndim == 2 for r in read):
        raise RuntimeError(
            f"read shapes {[r.shape for r in read]} must all be 2-dimensional to match "
            "their [n,m] sorts; a 1-D operand broadcasts incorrectly in matrix ops"
        )

    result = interpret(itype)(*read)
    return [result]
