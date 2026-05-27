import torch


def eval_itype(itype, read):
    """Evaluate a single instruction type with the given input tensors."""
    fn = _EVAL.get(itype.op_name)
    if fn is None:
        raise RuntimeError(
            f"cannot evaluate instruction '{itype.theory_name}.{itype.op_name}'"
        )
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


def _uninterpreted(itype, _read):
    raise RuntimeError(f"cannot evaluate uninterpreted function '{itype.name}'")


# Dispatch keyed by `IType.op_name`. Operations with identical semantics across
# LIA/LRA/BV share a single entry; constants pull their data via `it.const_data`.
_EVAL = {
    # Identity / flow
    "Id": lambda it, r: [r[0].clone()],
    "Ite": lambda it, r: [torch.where(r[0], r[1], r[2])],
    # Arithmetic (only Add exists in LIA/LRA; BV adds Mul/UDiv/SDiv/MatMul)
    "Add": lambda it, r: [r[0] + r[1]],
    "Mul": lambda it, r: [r[0] * r[1]],
    "UDiv": lambda it, r: [r[0].div(r[1], rounding_mode="floor")],
    "SDiv": lambda it, r: [r[0].div(r[1], rounding_mode="trunc")],
    "MatMul": lambda it, r: [r[0] @ r[1]],
    # Comparisons
    "Eq": lambda it, r: [r[0].eq(r[1])],
    "Ne": lambda it, r: [r[0].ne(r[1])],
    "Lt": lambda it, r: [r[0].lt(r[1])],
    "Le": lambda it, r: [r[0].le(r[1])],
    "Gt": lambda it, r: [r[0].gt(r[1])],
    "Ge": lambda it, r: [r[0].ge(r[1])],
    # Logical
    "And": lambda it, r: [r[0].logical_and(r[1])],
    "Or": lambda it, r: [r[0].logical_or(r[1])],
    "Not": lambda it, r: [r[0].logical_not()],
    "Xor": lambda it, r: [r[0].logical_xor(r[1])],
    # Neural-ish / aggregate
    "ReLU": lambda it, r: [r[0].relu()],
    "Argmax": lambda it, r: [r[0].argmax()],
    "Min": lambda it, r: [torch.minimum(r[0], r[1])],
    "Max": lambda it, r: [torch.maximum(r[0], r[1])],
    # Constants (tensor payload accessed via `it.const_data`)
    "ConstBool": lambda it, r: [it.const_data.clone()],
    "ConstInt": lambda it, r: [it.const_data.clone()],
    "ConstReal": lambda it, r: [it.const_data.clone()],
    "Const": lambda it, r: [it.const_data.clone()],
    # Uninterpreted
    "Uninterpreted": _uninterpreted,
}
