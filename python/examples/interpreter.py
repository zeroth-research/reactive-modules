from zrth.eval import zero_tensor as _zero_tensor, eval_itype as _eval


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
