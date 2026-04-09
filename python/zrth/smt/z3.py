import z3
import torch
from ..zrth import IType, DType


def z3_eval_itype(itype, read):
    """Evaluate a single instruction type, producing Z3 expressions."""
    fn = _Z3_EVAL.get(type(itype))
    if fn is None:
        raise RuntimeError(f"cannot translate instruction type '{type(itype).__name__}' to Z3")
    return fn(itype, read)



def make_z3_vars(name, dtype):
    """Create fresh Z3 variables from a DType, returning a list."""
    shape = dtype._0 if hasattr(dtype, '_0') else [1]
    n = 1
    for d in shape:
        n *= d

    if isinstance(dtype, (DType.Float, DType.Real)):
        if n == 1:
            return [z3.Real(name)]
        return [z3.Real(f"{name}_{i}") for i in range(n)]
    elif isinstance(dtype, DType.Bool):
        if n == 1:
            return [z3.Bool(name)]
        return [z3.Bool(f"{name}_{i}") for i in range(n)]
    elif isinstance(dtype, DType.Int):
        if n == 1:
            return [z3.Int(name)]
        return [z3.Int(f"{name}_{i}") for i in range(n)]
    else:
        raise NotImplementedError(f"make_z3_vars: unsupported dtype {dtype}")


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

    # Condition must be boolean; if it's arithmetic, coerce
    cond_vals = _to_bool(cond)
    # Broadcast scalar condition
    c = cond_vals[0]

    # Align branch types
    if any(z3.is_bool(v) for v in then_branch) or any(z3.is_bool(v) for v in else_branch):
        # If either branch is bool, check if the other is too
        then_all_bool = all(z3.is_bool(v) for v in then_branch)
        else_all_bool = all(z3.is_bool(v) for v in else_branch)
        if then_all_bool and else_all_bool:
            return [z3.If(c, t, f) for t, f in zip(then_branch, else_branch)]
        # Mixed: coerce everything to arithmetic
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
# Dispatch table
# ============================================================================

_Z3_EVAL = {
    type(IType.Tensor(torch.zeros(1))): lambda it, r: [_tensor_to_z3(it._0)],
    type(IType.Id()): lambda it, r: [r[0]],
    # Arithmetic (element-wise)
    type(IType.Add()): lambda it, r: [[a + b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    type(IType.Sub()): lambda it, r: [[a - b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    type(IType.Mul()): lambda it, r: [[a * b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    type(IType.Div()): lambda it, r: [[a / b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    type(IType.Neg()): lambda it, r: [[-a for a in _to_arith(r[0])]],
    type(IType.Abs()): lambda it, r: [[z3.If(a >= 0, a, -a) for a in _to_arith(r[0])]],
    # Comparisons (element-wise, produce Bool)
    type(IType.Eq()): lambda it, r: [[a == b for a, b in zip(r[0], r[1])]],
    type(IType.Neq()): lambda it, r: [[a != b for a, b in zip(r[0], r[1])]],
    type(IType.Lt()): lambda it, r: [[a < b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    type(IType.Le()): lambda it, r: [[a <= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    type(IType.Gt()): lambda it, r: [[a > b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    type(IType.Ge()): lambda it, r: [[a >= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    # Logical (element-wise)
    type(IType.And()): lambda it, r: [[z3.And(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]],
    type(IType.Or()): lambda it, r: [[z3.Or(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]],
    type(IType.Not()): lambda it, r: [[z3.Not(a) for a in _to_bool(r[0])]],
    # Control flow
    type(IType.Ite()): lambda it, r: [_z3_ite(it, r)],
    # Activation / neural
    type(IType.ReLU()): lambda it, r: [[z3.If(x > 0, x, z3.RealVal(0)) for x in _to_arith(r[0])]],
    type(IType.Linear()): lambda it, r: [_z3_linear(it, r)],
    # Constants
    type(IType.ConstBool(False)): lambda it, r: [[z3.BoolVal(it._0)]],
    type(IType.ConstInt(0)): lambda it, r: [[z3.IntVal(it._0)]],
}
