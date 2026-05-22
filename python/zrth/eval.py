import torch
from .zrth import IType


def eval_itype(itype, read):
    """Evaluate a single instruction type with the given input tensors."""
    fn = _EVAL.get(type(itype))
    if fn is None:
        raise RuntimeError(f"cannot evaluate instruction type '{type(itype).__name__}'")
    return fn(itype, read)


# ============================================================================
# Interpreter helpers (used by zrth.gym.Env)
# ============================================================================


def _execute_block(state, atoms, get_block):
    """Evaluate a block from each atom."""
    for atom in atoms:
        for term in get_block(atom):
            read = [state[w] for w in term.read]
            results = eval_itype(term.itype, read)
            for w, val in zip(term.write, results):
                state[w] = val


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
        wire = wire_names[name][0]  # read from latched wire
        if wire in state:
            val = state[wire]
            return val.item() if val.numel() == 1 else val.detach().clone()
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


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
    type(IType.Tanh()): lambda it, r: [r[0].tanh()],
    type(IType.Linear()): lambda it, r: [r[0] @ r[1].T + r[2]],
    type(IType.Argmax()): lambda it, r: [r[0].argmax()],
    # Tensor operations
    type(IType.TensorSum()): lambda it, r: [r[0].sum()],
    type(IType.TensorMean()): lambda it, r: [r[0].mean()],
    type(IType.TensorMax()): lambda it, r: [r[0].max()],
    type(IType.TensorGet()): lambda it, r: [r[0].view(-1)[int(r[1].item())]],
    type(IType.TensorSet()): lambda it, r: _tensor_set(r[0], r[1], r[2]),
    type(IType.Stack()): lambda it, r: [torch.cat([x.flatten() for x in r])],
    # Constants
    type(IType.ConstBool(False)): lambda it, r: [
        torch.tensor([it._0], dtype=torch.bool)
    ],
    type(IType.ConstInt(0)): lambda it, r: [torch.tensor([it._0], dtype=torch.long)],
    # Word-level operations
    type(IType.BitSelect(0, 0)): lambda it, r: [
        ((r[0] >> it._1) & ((1 << (it._0 - it._1 + 1)) - 1))
    ],
    type(IType.Extend(0)): lambda it, r: [r[0] & ((1 << it._0) - 1)],
    type(IType.BVToBool()): lambda it, r: [r[0].bool()],
    type(IType.BoolToBV()): lambda it, r: [r[0].long() & 1],
    # Uninterpreted
    type(IType.Uninterpreted("")): lambda it, r: _uninterpreted(it),
}
