import torch
from .zrth import IType, DType


def zero_tensor(dtype):
    """Create a zero tensor matching the given DType."""
    shape = list(dtype.shape)
    if isinstance(dtype, DType.Bool):
        return torch.zeros(shape, dtype=torch.bool)
    elif isinstance(dtype, DType.Int):
        return torch.zeros(shape, dtype=torch.long)
    elif isinstance(dtype, DType.Float):
        return torch.zeros(shape, dtype=torch.float32)
    elif isinstance(dtype, DType.Real):
        return torch.zeros(shape, dtype=torch.float64)
    elif isinstance(dtype, (DType.UWord, DType.SWord)):
        return torch.zeros([1], dtype=torch.long)
    else:
        raise RuntimeError(f"unknown dtype kind: {dtype}")


def eval_itype(itype, read):
    """Evaluate a single instruction type with the given input tensors."""
    fn = _EVAL.get(type(itype))
    if fn is None:
        raise RuntimeError(f"cannot evaluate instruction type '{type(itype).__name__}'")
    return fn(itype, read)


def _tensor_set(tensor, index, value):
    result = tensor.clone()
    flat = result.view(-1)
    flat[int(index.item())] = value
    return [result]


def _uninterpreted(itype):
    raise RuntimeError(f"cannot evaluate uninterpreted function '{itype._0}'")


_EVAL = {
    type(IType.Tensor(torch.zeros(1))): lambda it, r: [it._0.clone()],
    type(IType.Id()): lambda it, r: [r[0].clone()],
    # Arithmetic
    type(IType.Add()): lambda it, r: [r[0] + r[1]],
    type(IType.Sub()): lambda it, r: [r[0] - r[1]],
    type(IType.Mul()): lambda it, r: [r[0] * r[1]],
    type(IType.Div()): lambda it, r: [r[0] / r[1]],
    type(IType.Mod()): lambda it, r: [r[0] % r[1]],
    type(IType.Neg()): lambda it, r: [-r[0]],
    type(IType.Abs()): lambda it, r: [r[0].abs()],
    type(IType.MatMul()): lambda it, r: [r[0] @ r[1]],
    # Comparisons
    type(IType.Eq()): lambda it, r: [r[0].eq(r[1])],
    type(IType.Neq()): lambda it, r: [r[0].ne(r[1])],
    type(IType.Lt()): lambda it, r: [r[0].lt(r[1])],
    type(IType.Le()): lambda it, r: [r[0].le(r[1])],
    type(IType.Gt()): lambda it, r: [r[0].gt(r[1])],
    type(IType.Ge()): lambda it, r: [r[0].ge(r[1])],
    # Logical
    type(IType.And()): lambda it, r: [r[0].logical_and(r[1])],
    type(IType.Or()): lambda it, r: [r[0].logical_or(r[1])],
    type(IType.Not()): lambda it, r: [r[0].logical_not()],
    type(IType.Xor()): lambda it, r: [r[0].logical_xor(r[1])],
    type(IType.Xnor()): lambda it, r: [r[0].logical_xor(r[1]).logical_not()],
    type(IType.Implies()): lambda it, r: [r[0].logical_not().logical_or(r[1])],
    # Control flow
    type(IType.Ite()): lambda it, r: [torch.where(r[0], r[1], r[2])],
    # Transcendental
    type(IType.Sin()): lambda it, r: [r[0].sin()],
    type(IType.Cos()): lambda it, r: [r[0].cos()],
    # Activation / neural
    type(IType.ReLU()): lambda it, r: [r[0].relu()],
    type(IType.Linear()): lambda it, r: [r[0] @ r[1] + r[2]],
    type(IType.Argmax()): lambda it, r: [r[0].argmax()],
    # Tensor operations
    type(IType.TensorSum()): lambda it, r: [r[0].sum()],
    type(IType.TensorMean()): lambda it, r: [r[0].mean()],
    type(IType.TensorMax()): lambda it, r: [r[0].max()],
    type(IType.TensorGet()): lambda it, r: [r[0].view(-1)[int(r[1].item())]],
    type(IType.TensorSet()): lambda it, r: _tensor_set(r[0], r[1], r[2]),
    # Constants
    type(IType.ConstBool(False)): lambda it, r: [torch.tensor([it._0], dtype=torch.bool)],
    type(IType.ConstInt(0)): lambda it, r: [torch.tensor([it._0], dtype=torch.long)],
    # Word-level operations
    type(IType.BitSelect(0, 0)): lambda it, r: [((r[0] >> it._1) & ((1 << (it._0 - it._1 + 1)) - 1))],
    type(IType.Extend(0)): lambda it, r: [r[0] & ((1 << it._0) - 1)],
    type(IType.ToBool()): lambda it, r: [r[0].bool()],
    type(IType.ToWord1()): lambda it, r: [r[0].long() & 1],
    type(IType.ToUnsigned()): lambda it, r: [r[0].abs()],
    type(IType.ToSigned()): lambda it, r: [r[0]],
    # Uninterpreted
    type(IType.Uninterpreted("")): lambda it, r: _uninterpreted(it),
}
