import torch
from zrth import IType


def _zero_tensor(dtype):
    """Create a zero tensor matching the given DType."""
    kind = dtype.kind()
    shape = list(dtype.shape)
    if kind == "Bool":
        return torch.zeros(shape, dtype=torch.bool)
    elif kind == "Int":
        return torch.zeros(shape, dtype=torch.long)
    elif kind == "Float":
        return torch.zeros(shape, dtype=torch.float32)
    elif kind == "Real":
        return torch.zeros(shape, dtype=torch.float64)
    elif kind in ("UWord", "SWord"):
        return torch.zeros([1], dtype=torch.long)
    else:
        raise RuntimeError(f"unknown dtype kind: {kind}")


def _eval(itype, read):
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
    type(IType.Add()): lambda it, r: [r[0] + r[1]],
    type(IType.Sub()): lambda it, r: [r[0] - r[1]],
    type(IType.Mul()): lambda it, r: [r[0] * r[1]],
    type(IType.Div()): lambda it, r: [r[0] / r[1]],
    type(IType.MatMul()): lambda it, r: [r[0] @ r[1]],
    type(IType.Eq()): lambda it, r: [r[0].eq(r[1])],
    type(IType.Neq()): lambda it, r: [r[0].ne(r[1])],
    type(IType.Lt()): lambda it, r: [r[0].lt(r[1])],
    type(IType.Le()): lambda it, r: [r[0].le(r[1])],
    type(IType.Gt()): lambda it, r: [r[0].gt(r[1])],
    type(IType.Ge()): lambda it, r: [r[0].ge(r[1])],
    type(IType.And()): lambda it, r: [r[0].logical_and(r[1])],
    type(IType.Or()): lambda it, r: [r[0].logical_or(r[1])],
    type(IType.Not()): lambda it, r: [r[0].logical_not()],
    type(IType.Ite()): lambda it, r: [torch.where(r[0], r[1], r[2])],
    type(IType.Argmax()): lambda it, r: [r[0].argmax()],
    type(IType.ReLU()): lambda it, r: [r[0].relu()],
    type(IType.TensorSum()): lambda it, r: [r[0].sum()],
    type(IType.TensorMean()): lambda it, r: [r[0].mean()],
    type(IType.TensorMax()): lambda it, r: [r[0].max()],
    type(IType.TensorGet()): lambda it, r: [r[0].view(-1)[int(r[1].item())]],
    type(IType.TensorSet()): lambda it, r: _tensor_set(r[0], r[1], r[2]),
    type(IType.Linear()): lambda it, r: [r[0] @ r[1] + r[2]],
    type(IType.Uninterpreted("")): lambda it, r: _uninterpreted(it),
    # Arithmetic extensions
    type(IType.Mod()): lambda it, r: [r[0] % r[1]],
    type(IType.Neg()): lambda it, r: [-r[0]],
    type(IType.Abs()): lambda it, r: [r[0].abs()],
    # Logical extensions
    type(IType.Xor()): lambda it, r: [r[0].logical_xor(r[1])],
    type(IType.Xnor()): lambda it, r: [r[0].logical_xor(r[1]).logical_not()],
    type(IType.Implies()): lambda it, r: [r[0].logical_not().logical_or(r[1])],
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
}


class Interpreter:
    def __init__(self, module):
        self.module = module
        self.state = {}
        self.initialized = False

    def initialize(self, env_inputs=None):
        """Run the init block and latch.

        External wires not provided in *env_inputs* are automatically
        initialized to zero tensors (matching their declared dtype).
        """
        self._load_env_inputs(env_inputs)
        self._execute("init")
        self._latch()
        self.initialized = True

    def step(self, env_inputs=None):
        if not self.initialized:
            raise RuntimeError("interpreter not initialized; call initialize() first")
        self._load_env_inputs(env_inputs)
        self._execute("update")
        self._latch()

    def get(self, wire_id):
        if wire_id not in self.state:
            raise RuntimeError(f"wire w{wire_id} not in state")
        return self.state[wire_id].clone()

    def state_dict(self):
        return dict(self.state)

    def _load_env_inputs(self, env_inputs):
        if env_inputs is not None:
            for wire_id, tensor in env_inputs.items():
                self.state[wire_id] = tensor
        # Auto-initialize any external wires not yet in state
        extl = self.module.extl
        for i in range(len(extl)):
            ltc, nxt = extl[i]
            for w in (ltc, nxt):
                if w.id not in self.state:
                    self.state[w.id] = _zero_tensor(w.dtype)
        # Auto-initialize parameter wires not yet in state
        param = self.module.param
        for i in range(len(param)):
            w = param[i]
            if w.id not in self.state:
                self.state[w.id] = _zero_tensor(w.dtype)

    def _execute(self, block_type):
        atoms = self.module.atoms
        for atom_idx in range(len(atoms)):
            atom = atoms[atom_idx]
            block = atom.init if block_type == "init" else atom.update
            for i in range(len(block)):
                term = block[i]
                read = [self.state[term.read[j].id] for j in range(len(term.read))]
                results = _eval(term.itype, read)
                for j in range(len(term.write)):
                    self.state[term.write[j].id] = results[j]

    def _latch(self):
        ctrl = self.module.ctrl
        for i in range(len(ctrl)):
            ltc, nxt = ctrl[i]
            nxt_id = nxt.id
            if nxt_id in self.state:
                self.state[ltc.id] = self.state[nxt_id].clone()
