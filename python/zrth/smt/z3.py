import z3
import torch
from ..zrth import DType


def eval(itype, read):
    """Evaluate a single instruction type, producing Z3 expressions."""
    fn = _Z3_EVAL.get(itype.op_name)
    if fn is None:
        raise RuntimeError(
            f"cannot translate instruction '{itype.theory_name}.{itype.op_name}' to Z3"
        )
    return fn(itype, read)


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


# ============================================================================
# Dispatch keyed by `IType.op_name`. Same op across LIA/LRA/BV shares a key.
# ============================================================================

_Z3_EVAL = {
    "Id": lambda it, r: [r[0]],
    # Constants — IType.<theory>.Const*(t) exposes payload via `it.const_data`.
    "ConstBool": lambda it, r: [_tensor_to_z3(it.const_data)],
    "ConstInt": lambda it, r: [_tensor_to_z3(it.const_data)],
    "ConstReal": lambda it, r: [_tensor_to_z3(it.const_data)],
    "Const": lambda it, r: [_tensor_to_z3(it.const_data)],
    # Arithmetic (element-wise)
    "Add": lambda it, r: [[a + b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    "Mul": lambda it, r: [[a * b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    # Comparisons (element-wise, produce Bool)
    "Eq": lambda it, r: [[a == b for a, b in zip(r[0], r[1])]],
    "Ne": lambda it, r: [[a != b for a, b in zip(r[0], r[1])]],
    "Lt": lambda it, r: [[a < b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    "Le": lambda it, r: [[a <= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    "Gt": lambda it, r: [[a > b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    "Ge": lambda it, r: [[a >= b for a, b in zip(_to_arith(r[0]), _to_arith(r[1]))]],
    # Logical (element-wise)
    "And": lambda it, r: [[z3.And(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]],
    "Or": lambda it, r: [[z3.Or(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]],
    "Not": lambda it, r: [[z3.Not(a) for a in _to_bool(r[0])]],
    "Xor": lambda it, r: [[z3.Xor(a, b) for a, b in zip(_to_bool(r[0]), _to_bool(r[1]))]],
    # Control flow
    "Ite": lambda it, r: [_z3_ite(it, r)],
    # Neural-ish
    "ReLU": lambda it, r: [[z3.If(x > 0, x, z3.RealVal(0)) for x in _to_arith(r[0])]],
}
