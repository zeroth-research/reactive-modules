import torch


def _zero_tensor(dtype):
    """Create a zero tensor matching the given DType."""
    kind = dtype.kind()
    shape = list(dtype.shape)
    if kind == "TensorBool":
        return torch.zeros(shape, dtype=torch.bool)
    elif kind == "TensorInt":
        return torch.zeros(shape, dtype=torch.long)
    elif kind == "TensorFloat":
        return torch.zeros(shape, dtype=torch.float32)
    elif kind == "TensorReal":
        return torch.zeros(shape, dtype=torch.float64)
    else:
        raise RuntimeError(f"unknown dtype kind: {kind}")


def _eval(itype, read):
    name = type(itype).__name__
    fn = _EVAL.get(name)
    if fn is None:
        raise RuntimeError(f"cannot evaluate instruction type '{name}'")
    return fn(itype, read)


def _tensor_set(tensor, index, value):
    result = tensor.clone()
    flat = result.view(-1)
    flat[int(index.item())] = value
    return [result]


def _uninterpreted(itype):
    raise RuntimeError(f"cannot evaluate uninterpreted function '{itype._0}'")


_EVAL = {
    "IType_Tensor":     lambda it, r: [it._0.clone()],
    "IType_Id":         lambda it, r: [r[0].clone()],
    "IType_Add":        lambda it, r: [r[0] + r[1]],
    "IType_Sub":        lambda it, r: [r[0] - r[1]],
    "IType_Mul":        lambda it, r: [r[0] * r[1]],
    "IType_Div":        lambda it, r: [r[0] / r[1]],
    "IType_MatMul":     lambda it, r: [r[0] @ r[1]],
    "IType_Eq":         lambda it, r: [r[0].eq(r[1])],
    "IType_Neq":        lambda it, r: [r[0].ne(r[1])],
    "IType_Lt":         lambda it, r: [r[0].lt(r[1])],
    "IType_Le":         lambda it, r: [r[0].le(r[1])],
    "IType_Gt":         lambda it, r: [r[0].gt(r[1])],
    "IType_Ge":         lambda it, r: [r[0].ge(r[1])],
    "IType_And":        lambda it, r: [r[0].logical_and(r[1])],
    "IType_Or":         lambda it, r: [r[0].logical_or(r[1])],
    "IType_Not":        lambda it, r: [r[0].logical_not()],
    "IType_Ite":        lambda it, r: [torch.where(r[0], r[1], r[2])],
    "IType_Argmax":     lambda it, r: [r[0].argmax()],
    "IType_ReLU":       lambda it, r: [r[0].relu()],
    "IType_TensorSum":  lambda it, r: [r[0].sum()],
    "IType_TensorMean": lambda it, r: [r[0].mean()],
    "IType_TensorMax":  lambda it, r: [r[0].max()],
    "IType_TensorGet":  lambda it, r: [r[0].view(-1)[int(r[1].item())]],
    "IType_TensorSet":  lambda it, r: _tensor_set(r[0], r[1], r[2]),
    "IType_Linear":     lambda it, r: [r[0] @ r[1] + r[2]],
    "IType_Uninterpreted": lambda it, r: _uninterpreted(it),
}


class Interpreter:
    def __init__(self, module):
        self.module = module
        self.state = {}
        self.initialized = False

    def initialize(self, env_inputs=None):
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
        return {wid: t.clone() for wid, t in self.state.items()}

    def _load_env_inputs(self, env_inputs):
        if env_inputs is not None:
            for wire_id, tensor in env_inputs.items():
                self.state[wire_id] = tensor
        # Auto-initialize any external wires not yet in state
        extl = self.module.extl()
        for i in range(len(extl)):
            ltc, nxt = extl[i]
            for w in (ltc, nxt):
                if w.id() not in self.state:
                    self.state[w.id()] = _zero_tensor(w.dtype())

    def _execute(self, block_type):
        atoms = self.module.atoms()
        for atom_idx in range(len(atoms)):
            atom = atoms[atom_idx]
            block = atom.init() if block_type == "init" else atom.update()
            for i in range(len(block)):
                term = block[i]
                read = [self.state[term.read[j].id()] for j in range(len(term.read))]
                results = _eval(term.itype, read)
                for j in range(len(term.write)):
                    self.state[term.write[j].id()] = results[j]

    def _latch(self):
        ctrl = self.module.ctrl()
        for i in range(len(ctrl)):
            ltc, nxt = ctrl[i]
            nxt_id = nxt.id()
            if nxt_id in self.state:
                self.state[ltc.id()] = self.state[nxt_id].clone()
