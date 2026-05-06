import torch
from .zrth import IType


def eval_itype(itype, read):
    """Evaluate a single instruction type with the given input tensors."""
    if isinstance(itype, IType):
        if itype.is_const:
            return [torch.tensor(itype.const_data)]
        fn = _EVAL.get(itype)
        if fn is None:
            raise RuntimeError(f"cannot evaluate op '{itype!r}'")
        return fn(itype, read)
    # Python-only op types (BitSelect, Extend, etc.)
    fn = _EVAL_PYTHON.get(type(itype))
    if fn is None:
        raise RuntimeError(f"cannot evaluate instruction type '{type(itype).__name__}'")
    return fn(itype, read)


# ============================================================================
# Interpreter helpers (shared by zrth.gym.Wrapper, zrth.gym.Env)
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


# ============================================================================
# Dispatch table — keyed by Ops value (uses __hash__ / __eq__)
# ============================================================================

def _multi(*ops):
    """Register the same handler under multiple Ops keys."""
    def decorator(fn):
        for op in ops:
            _EVAL[op] = fn
        return fn
    return decorator


_EVAL = {}

# Identity / flow
_EVAL[IType.Id] = lambda it, r: [r[0].clone()]
_EVAL[IType.Ite] = lambda it, r: [torch.where(r[0], r[1], r[2])]
_EVAL[IType.BVToBool]    = lambda it, r: [r[0].bool()]
_EVAL[IType.BVToWord1]   = lambda it, r: [r[0].long() & 1]
_EVAL[IType.ToUnsigned]  = lambda it, r: [r[0].abs()]
_EVAL[IType.ToSigned]    = lambda it, r: [r[0]]

# Polymorphic arithmetic — same semantics across Int / Float / Real
for _name, _impl in [
    ("Add",    lambda r: r[0] + r[1]),
    ("Sub",    lambda r: r[0] - r[1]),
    ("Mul",    lambda r: r[0] * r[1]),
    ("Div",    lambda r: r[0] / r[1]),
    ("Mod",    lambda r: r[0] % r[1]),
    ("Neg",    lambda r: -r[0]),
    ("Abs",    lambda r: r[0].abs()),
    ("MatMul", lambda r: r[0] @ r[1]),
]:
    for _ns in (IType.Int, IType.Float, IType.Real):
        _EVAL[getattr(_ns, _name)] = (lambda fn: lambda it, r: [fn(r)])(_impl)

# Comparisons (dtype-agnostic Cmp ops)
_EVAL[IType.Cmp.Eq] = lambda it, r: [r[0].eq(r[1])]
_EVAL[IType.Cmp.Ne] = lambda it, r: [r[0].ne(r[1])]
_EVAL[IType.Cmp.Lt] = lambda it, r: [r[0].lt(r[1])]
_EVAL[IType.Cmp.Le] = lambda it, r: [r[0].le(r[1])]
_EVAL[IType.Cmp.Gt] = lambda it, r: [r[0].gt(r[1])]
_EVAL[IType.Cmp.Ge] = lambda it, r: [r[0].ge(r[1])]

# Bool ops
_EVAL[IType.Bool.And]     = lambda it, r: [r[0].logical_and(r[1])]
_EVAL[IType.Bool.Or]      = lambda it, r: [r[0].logical_or(r[1])]
_EVAL[IType.Bool.Not]     = lambda it, r: [r[0].logical_not()]
_EVAL[IType.Bool.Xor]     = lambda it, r: [r[0].logical_xor(r[1])]
_EVAL[IType.Bool.Xnor]    = lambda it, r: [r[0].logical_xor(r[1]).logical_not()]
_EVAL[IType.Bool.Implies] = lambda it, r: [r[0].logical_not().logical_or(r[1])]

# Transcendental (Real only)
_EVAL[IType.Real.Sin] = lambda it, r: [r[0].sin()]
_EVAL[IType.Real.Cos] = lambda it, r: [r[0].cos()]

# Neural network
_EVAL[IType.NN.ReLU]   = lambda it, r: [r[0].relu()]
_EVAL[IType.NN.Tanh]   = lambda it, r: [r[0].tanh()]
_EVAL[IType.NN.Linear] = lambda it, r: [r[0] @ r[1].T + r[2]]

# Tensor / reduction ops
_EVAL[IType.Tensor.Argmax] = lambda it, r: [r[0].argmax()]
_EVAL[IType.Tensor.Sum]    = lambda it, r: [r[0].sum()]
_EVAL[IType.Tensor.Mean]   = lambda it, r: [r[0].mean()]
_EVAL[IType.Tensor.Max]    = lambda it, r: [r[0].max()]
_EVAL[IType.Tensor.Get]    = lambda it, r: [r[0].view(-1)[int(r[1].item())]]
_EVAL[IType.Tensor.Set]    = lambda it, r: _tensor_set(r[0], r[1], r[2])


# ============================================================================
# Dispatch table for Python-only op types (BitSelect, Extend, etc.)
# ============================================================================

_EVAL_PYTHON = {}
