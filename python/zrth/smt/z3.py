import z3
import torch
from ..zrth import IType, DType


def eval(itype, read):
    """Evaluate a single instruction type, producing Z3 expressions."""
    if isinstance(itype, IType):
        if itype.is_const:
            data = itype.const_data
            flat = [x for row in data for x in row]
            return [_const_to_z3(flat)]
        fn = _Z3_EVAL.get(itype)
        if fn is None:
            raise RuntimeError(f"cannot translate op '{itype!r}' to Z3")
        return fn(itype, read)
    # Python-only op types
    fn = _Z3_EVAL_PYTHON.get(type(itype))
    if fn is None:
        raise RuntimeError(f"cannot translate instruction type '{type(itype).__name__}' to Z3")
    return fn(itype, read)


def _const_to_z3(flat):
    """Convert a flat list of Python scalars to Z3 constants."""
    if flat and isinstance(flat[0], bool):
        return [z3.BoolVal(v) for v in flat]
    elif flat and isinstance(flat[0], int):
        return [z3.IntVal(v) for v in flat]
    else:
        return [z3.RealVal(str(v)) for v in flat]


def _tensor_to_z3(tensor):
    """Convert a torch.Tensor to a list of Z3 constants."""
    flat = tensor.detach().flatten().tolist()
    if tensor.dtype == torch.bool:
        return [z3.BoolVal(bool(v)) for v in flat]
    elif tensor.dtype in (torch.long, torch.int, torch.int32, torch.int64):
        return [z3.IntVal(int(v)) for v in flat]
    else:
        return [z3.RealVal(str(v)) for v in flat]


def _to_arith(vals):
    """Coerce a list of Z3 expressions to ArithRef (Bool -> If(b, 1, 0))."""
    return [z3.If(v, z3.RealVal(1), z3.RealVal(0)) if z3.is_bool(v) else v for v in vals]


def _to_bool(vals):
    """Coerce a list of Z3 expressions to BoolRef (ArithRef -> val != 0)."""
    return [v if z3.is_bool(v) else v != z3.RealVal(0) for v in vals]


def _z3_ite(itype, read):
    cond = read[0]
    then_branch = read[1]
    else_branch = read[2]

    cond_vals = _to_bool(cond)
    c = cond_vals[0]

    if any(z3.is_bool(v) for v in then_branch) or any(z3.is_bool(v) for v in else_branch):
        then_all_bool = all(z3.is_bool(v) for v in then_branch)
        else_all_bool = all(z3.is_bool(v) for v in else_branch)
        if then_all_bool and else_all_bool:
            return [z3.If(c, t, f) for t, f in zip(then_branch, else_branch)]
        then_branch = _to_arith(then_branch)
        else_branch = _to_arith(else_branch)

    return [z3.If(c, t, f) for t, f in zip(then_branch, else_branch)]


def _z3_linear(itype, read):
    """Linear: input @ weight.T + bias"""
    input_vec = read[0]
    weight_flat = read[1]
    bias = read[2]

    in_features = len(input_vec)
    out_features = len(bias)

    result = []
    for o in range(out_features):
        dot = z3.Sum([weight_flat[o * in_features + i] * input_vec[i]
                      for i in range(in_features)])
        result.append(dot + bias[o])
    return result


# ============================================================================
# Dispatch table — keyed by Ops value
# ============================================================================

_Z3_EVAL = {}

_Z3_EVAL[IType.Id]  = lambda it, r: [r[0]]
_Z3_EVAL[IType.Ite] = lambda it, r: [_z3_ite(it, r)]

# Polymorphic arithmetic
for _name, _impl in [
    ("Add", lambda r: [a + b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]),
    ("Sub", lambda r: [a - b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]),
    ("Mul", lambda r: [a * b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]),
    ("Div", lambda r: [a / b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]),
    ("Neg", lambda r: [-a for a in _to_arith(r[0])]),
    ("Abs", lambda r: [z3.If(a >= 0, a, -a) for a in _to_arith(r[0])]),
]:
    for _ns in (IType.Int, IType.Float, IType.Real):
        _Z3_EVAL[getattr(_ns, _name)] = (lambda fn: lambda it, r: [fn(r)])(_impl)

# Comparisons
_Z3_EVAL[IType.Cmp.Eq] = lambda it, r: [[a == b for a, b in zip(r[0], r[1])]]
_Z3_EVAL[IType.Cmp.Ne] = lambda it, r: [[a != b for a, b in zip(r[0], r[1])]]
_Z3_EVAL[IType.Cmp.Lt] = lambda it, r: [[a < b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
_Z3_EVAL[IType.Cmp.Le] = lambda it, r: [[a <= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
_Z3_EVAL[IType.Cmp.Gt] = lambda it, r: [[a > b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]
_Z3_EVAL[IType.Cmp.Ge] = lambda it, r: [[a >= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]]

# Bool
_Z3_EVAL[IType.Bool.And]     = lambda it, r: [[z3.And(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]
_Z3_EVAL[IType.Bool.Or]      = lambda it, r: [[z3.Or(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]
_Z3_EVAL[IType.Bool.Not]     = lambda it, r: [[z3.Not(a) for a in _to_bool(r[0])]]
_Z3_EVAL[IType.Bool.Xor]     = lambda it, r: [[z3.Xor(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]
_Z3_EVAL[IType.Bool.Xnor]    = lambda it, r: [[z3.Not(z3.Xor(a, b)) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]
_Z3_EVAL[IType.Bool.Implies] = lambda it, r: [[z3.Implies(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]]

# Neural network
_Z3_EVAL[IType.NN.ReLU]   = lambda it, r: [[z3.If(x > 0, x, z3.RealVal(0)) for x in _to_arith(r[0])]]
_Z3_EVAL[IType.NN.Linear] = lambda it, r: [_z3_linear(it, r)]


# ============================================================================
# Dispatch table for Python-only op types
# ============================================================================

_Z3_EVAL_PYTHON = {}
