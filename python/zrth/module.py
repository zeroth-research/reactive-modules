import torch
from zrth import Module, Wire, DType, IType, Term
from zrth.analyzer import (
    convert_method, classify_attrs, infer_dtype, wire_pair, resolve_wire,
    AbstractValue,
)
from zrth.eval import eval_itype, zero_tensor
import gymnasium as gym
import torch.nn as nn
import inspect


# ============================================================================
# Helpers
# ============================================================================

def _space_to_dtype(space, is_action=False):
    """Convert a gym.spaces.Space object directly to a DType."""
    if isinstance(space, gym.spaces.Discrete):
        return DType.Float([space.n]) if is_action else DType.Int([1])
    elif isinstance(space, gym.spaces.Box):
        return DType.Float(list(space.shape))
    elif isinstance(space, gym.spaces.MultiBinary):
        return DType.Bool([space.n])
    else:
        raise ValueError(f"Unsupported gym space type: {type(space).__name__}")


def _value_to_const_term(value, wire):
    """Create a constant Term that writes a Python value to a wire."""
    if isinstance(value, bool):
        return Term(IType.ConstBool(value), [wire], [])
    elif isinstance(value, int):
        return Term(IType.ConstInt(value), [wire], [])
    elif isinstance(value, float):
        return Term(IType.Tensor(torch.tensor([value], dtype=torch.float32)), [wire], [])
    elif isinstance(value, torch.Tensor):
        return Term(IType.Tensor(value.clone()), [wire], [])
    else:
        raise ValueError(f"Cannot create constant term for {type(value).__name__}: {value}")


def _instance_to_init_attrs(instance):
    """Reconstruct init_attrs dict from a live instance's __dict__."""
    attrs = {}
    for name, value in instance.__dict__.items():
        if isinstance(value, (int, float, bool)):
            attrs[name] = AbstractValue.const(value)
        elif isinstance(value, torch.Tensor):
            attrs[name] = AbstractValue.const(value)
        elif isinstance(value, gym.spaces.Space):
            attrs[name] = _space_to_abstract(name, value)
    return attrs


def _space_to_abstract(name, space):
    """Convert a gym.spaces.Space to an AbstractValue with call_repr."""
    if isinstance(space, gym.spaces.Discrete):
        return AbstractValue.call_result(f"spaces.Discrete({space.n})")
    elif isinstance(space, gym.spaces.Box):
        shape = tuple(space.shape)
        return AbstractValue.call_result(
            f"spaces.Box(low={space.low.flat[0]}, high={space.high.flat[0]}, shape={shape})"
        )
    elif isinstance(space, gym.spaces.MultiBinary):
        return AbstractValue.call_result(f"spaces.MultiBinary({space.n})")
    else:
        raise ValueError(f"Cannot convert space '{name}' of type {type(space).__name__}")


# ============================================================================
# Module extraction
# ============================================================================

def _extract_env_module(env_instance, **kwargs):
    """Analyze a gym.Env instance and extract a symbolic Module."""
    env_cls = type(env_instance)

    user_wires = {
        "action":      kwargs.pop("action",      None),
        "observation": kwargs.pop("observation", None),
        "reward":      kwargs.pop("reward",      None),
        "terminated":  kwargs.pop("terminated",  None),
        "truncated":   kwargs.pop("truncated",   None),
    }

    action_param = next(
        p for p in inspect.signature(env_cls.step).parameters if p != "self"
    )

    action_dtype = _space_to_dtype(env_instance.action_space, is_action=True)
    observation_dtype = _space_to_dtype(env_instance.observation_space, is_action=False)

    init_attrs = _instance_to_init_attrs(env_instance)

    prvt, params, attr_vals = classify_attrs(
        env_cls, ['reset', 'step'], init_attrs=init_attrs, base_cls=gym.Env
    )

    prvt_wires = {name: wire_pair(infer_dtype(name, attr_vals.get(name))) for name in prvt}

    # Bake parameters as constant terms (live values from instance)
    param_wires = {}
    param_const_terms = []
    for name in params:
        value = getattr(env_instance, name)
        dtype = infer_dtype(name, AbstractValue.const(value))
        wire = Wire(dtype)
        param_wires[name] = wire
        param_const_terms.append(_value_to_const_term(value, wire))

    action      = resolve_wire("action",      action_dtype,      user_wires["action"])
    observation = resolve_wire("observation", observation_dtype, user_wires["observation"])
    reward      = resolve_wire("reward",      DType.Float([1]), user_wires["reward"])
    terminated  = resolve_wire("terminated",  DType.Bool([1]),  user_wires["terminated"])
    truncated   = resolve_wire("truncated",   DType.Bool([1]),  user_wires["truncated"])

    wires = {action_param: action, **prvt_wires}

    # gym reset returns (obs, info) → 1 result wire; step returns 4
    reset_result = [observation[1]]
    step_result  = [observation[1], reward[1], terminated[1], truncated[1]]

    reset_terms = convert_method(env_cls.reset, wires, reset_result, cls=env_cls, params=param_wires)
    step_terms  = convert_method(env_cls.step,  wires, step_result,  cls=env_cls, params=param_wires)

    # Add defaults for reward/terminated/truncated in init block
    reset_terms += [
        Term(IType.Tensor(torch.tensor([0.0])), [reward[1]], []),
        Term(IType.ConstBool(False), [terminated[1]], []),
        Term(IType.ConstBool(False), [truncated[1]], []),
    ]

    # Prepend constant terms so param wires have values before they're read
    reset_terms = param_const_terms + reset_terms
    step_terms  = [_value_to_const_term(getattr(env_instance, n), param_wires[n])
                   for n in params] + step_terms

    obs = [action, observation, reward, terminated, truncated]

    # Build name → latched wire mapping for Env.__getattr__
    wire_names = {}
    for name, pair in prvt_wires.items():
        wire_names[name] = pair[0]  # latched wire
    for name, wire in param_wires.items():
        wire_names[name] = wire

    return dict(init=reset_terms, update=step_terms, obs=obs, prvt=list(prvt_wires.values()),
                _wire_names=wire_names)


def _extract_nn_module(nn_instance, **kwargs):
    """Analyze an nn.Module instance and extract a symbolic Module.

    Uses live tensor references for weight/bias so that training updates
    flow through to the symbolic module automatically.
    """
    nn_cls = type(nn_instance)

    user_extl = kwargs.pop("extl", None)
    user_intf = kwargs.pop("intf", None)

    obs_param = next(
        p for p in inspect.signature(nn_cls.forward).parameters if p != "self"
    )

    # Infer layer structure from the actual nn.Module instance
    layers = {}
    live_layers = {}
    for name, child in nn_instance.named_modules():
        if isinstance(child, nn.Linear):
            layers[name] = (child.in_features, child.out_features)
            live_layers[name] = child

    if not layers:
        raise ValueError("No nn.Linear layers found in the module")

    layer_list = list(layers.values())
    obs_size  = layer_list[0][0]
    qval_size = layer_list[-1][1]

    extl = resolve_wire("extl", DType.Float([obs_size]),  user_extl)
    intf = resolve_wire("intf", DType.Float([qval_size]), user_intf)

    # Combinatorial: input wire is index 1 (next), swap the pair
    wires  = {obs_param: [extl[1], extl[0]]}
    result = [intf[1]]

    layer_out_features = {name: out for name, (_, out) in layers.items()}
    forward = convert_method(
        nn_cls.forward, wires, result, cls=nn_cls,
        layers=layer_out_features, live_layers=live_layers,
    )

    obs = [extl, intf]
    return dict(assign=forward, obs=obs)


# ============================================================================
# Shared symbolic interpreter mixin
# ============================================================================

class _SymbolicInterpreter:
    """Mixin providing symbolic state management and term evaluation."""

    def _setup_wire_pairs(self):
        """Set up wire pair references from obs. Call after Module is constructed."""
        n_obs = len(self.obs)
        if n_obs >= 5:
            self._action_pair = self.obs[0]
            self._observation_pair = self.obs[1]
            self._reward_pair = self.obs[2]
            self._terminated_pair = self.obs[3]
            self._truncated_pair = self.obs[4]
        elif n_obs >= 2:
            self._action_pair = self.obs[0]
            self._observation_pair = self.obs[1]
            self._reward_pair = None
            self._terminated_pair = None
            self._truncated_pair = None
        else:
            raise ValueError(f"Module needs at least 2 observable wire pairs, got {n_obs}")

    def _init_wires(self):
        """Zero-initialize all external and parameter wires."""
        extl = self.extl
        for i in range(len(extl)):
            ltc, nxt = extl[i]
            for w in (ltc, nxt):
                if w.id not in self._state:
                    self._state[w.id] = zero_tensor(w.dtype)
        param = self.param
        for i in range(len(param)):
            w = param[i]
            if w.id not in self._state:
                self._state[w.id] = zero_tensor(w.dtype)

    def _execute(self, block_type):
        atoms = self.atoms
        for atom_idx in range(len(atoms)):
            atom = atoms[atom_idx]
            block = atom.init if block_type == "init" else atom.update
            for i in range(len(block)):
                term = block[i]
                read = [self._state[term.read[j].id] for j in range(len(term.read))]
                results = eval_itype(term.itype, read)
                for j in range(len(term.write)):
                    self._state[term.write[j].id] = results[j]

    def _latch(self):
        ctrl = self.ctrl
        for i in range(len(ctrl)):
            ltc, nxt = ctrl[i]
            nxt_id = nxt.id
            if nxt_id in self._state:
                self._state[ltc.id] = self._state[nxt_id].clone()

    def _read_wire(self, wire):
        return self._state[wire.id].detach().clone()

    def _symbolic_reset(self):
        """Reset the symbolic interpreter state."""
        self._state = {}
        self._init_wires()
        self._execute("init")
        self._latch()
        self._initialized = True

    def _symbolic_step(self, action):
        """Advance the symbolic interpreter by one step."""
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        self._state[self._action_pair[0].id] = action_tensor
        self._execute("update")
        self._latch()

    def __getattr__(self, name):
        wire_names = object.__getattribute__(self, '_wire_names')
        if name in wire_names:
            state = object.__getattribute__(self, '_state')
            wire = wire_names[name]
            if wire.id in state:
                val = state[wire.id]
                return val.item() if val.numel() == 1 else val.detach().clone()
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ============================================================================
# Env: wraps a gym.Env as a gym.Wrapper with symbolic Module
# ============================================================================

class Env(_SymbolicInterpreter, Module, gym.Wrapper):
    """A gym.Wrapper backed by a symbolic Module.

    Runs both the real gym.Env and the symbolic interpreter in lockstep.
    Gym methods (render, close, etc.) delegate to the original env.
    Symbolic state is available for inspection and validation.

    Usage:
        Env(gym_env)                    → extract Module, wrap env
        Env(gym_env, module1, module2)  → extract + compose with other modules
    """

    def __new__(cls, *args, **kwargs):
        # Separate args: raw gym.Envs need extraction, Modules go straight to composition
        raw_envs = []       # raw gym.Env instances (not already wrapped as Module)
        modules = []        # all Modules (including Env/NN instances)
        backing_env = None  # the gym.Env for delegation (unwrapped)

        for a in args:
            if isinstance(a, Module):
                modules.append(a)
                # If it's an Env (gym.Wrapper), unwrap to find the backing gym.Env
                if isinstance(a, Env):
                    env = a.unwrapped  # walk wrapper chain to base env
                    if backing_env is not None:
                        raise TypeError("Env requires exactly 1 gym.Env, got multiple")
                    backing_env = env
            elif isinstance(a, gym.Env):
                raw_envs.append(a)
            else:
                raise TypeError(f"Expected gym.Env or Module, got {type(a)}")

        if len(raw_envs) > 1 or (len(raw_envs) == 1 and backing_env is not None):
            raise TypeError("Env requires exactly 1 gym.Env, got multiple")

        if len(raw_envs) == 1:
            # Extract Module from the raw gym.Env
            gym_env = raw_envs[0]
            extracted = _extract_env_module(gym_env, **kwargs)
            wire_names = extracted.pop('_wire_names', {})
            env_module = Module.__new__(Module, **extracted)
            modules = [env_module] + modules
            backing_env = gym_env
        else:
            # Inherit wire_names from any Env in the modules
            wire_names = {}
            for a in args:
                if isinstance(a, Env):
                    wire_names.update(a._wire_names)

        if backing_env is None:
            raise TypeError("Env requires exactly 1 gym.Env, got 0")

        # Compose all modules
        if len(modules) == 1:
            instance = Module.__new__(cls, modules[0])
        else:
            instance = Module.__new__(cls, *modules)
        instance._wire_names = wire_names
        instance._backing_env = backing_env
        return instance

    def __init__(self, *args, **kwargs):
        backing_env = object.__getattribute__(self, '_backing_env')
        gym.Wrapper.__init__(self, backing_env)
        self._state = {}
        self._initialized = False
        self._setup_wire_pairs()

    def reset(self, *, seed=None, options=None):
        # Run real env
        gym_result = self.env.reset(seed=seed, options=options)
        # Run symbolic interpreter
        self._symbolic_reset()
        return gym_result

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        # Run real env
        gym_result = self.env.step(action)
        # Run symbolic interpreter
        self._symbolic_step(action)
        return gym_result


# ============================================================================
# Simulator: runs pure symbolic Modules as a gym.Env
# ============================================================================

class Simulator(_SymbolicInterpreter, Module, gym.Env):
    """A gym.Env that runs a symbolic Module via the term interpreter.

    For pure Modules (no backing gym.Env). Use Env() for gym.Env instances.

    Usage:
        Simulator(module)               → run a single module
        Simulator(module1, module2)     → compose and run
    """

    def __new__(cls, *modules):
        for m in modules:
            if isinstance(m, gym.Env) and not isinstance(m, Module):
                raise TypeError("Use Env() for gym.Env instances, Simulator is for pure Modules")
            if not isinstance(m, Module):
                raise TypeError(f"Expected Module, got {type(m)}")

        if len(modules) == 1:
            instance = Module.__new__(cls, modules[0])
        else:
            instance = Module.__new__(cls, *modules)
        instance._wire_names = {}
        return instance

    def __init__(self, *modules):
        gym.Env.__init__(self)
        self._state = {}
        self._initialized = False
        self._setup_wire_pairs()

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)
        self._symbolic_reset()
        obs = self._read_wire(self._observation_pair[0])
        return obs.numpy(), {}

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        self._symbolic_step(action)

        obs = self._read_wire(self._observation_pair[0])
        reward = self._read_wire(self._reward_pair[0]).item() if self._reward_pair else 0.0
        terminated = bool(self._read_wire(self._terminated_pair[0]).item()) if self._terminated_pair else False
        truncated = bool(self._read_wire(self._truncated_pair[0]).item()) if self._truncated_pair else False

        return obs.numpy(), reward, terminated, truncated, {}


# ============================================================================
# NN: wraps an nn.Module and extracts a symbolic Module
# ============================================================================

class NN(Module, nn.Module):
    """An nn.Module backed by a symbolic Module with live tensor references.

    Inherits both Module (symbolic reactive module) and nn.Module (trainable).

    Usage:
        wrapped = NN(nn_module_instance)
        wrapped.parameters()   # returns original nn.Module parameters
        wrapped(x)             # runs forward pass via original nn.Module
        wrapped.atoms          # symbolic module structure

    Training the original nn.Module automatically updates the symbolic module
    because IType.Tensor holds a reference to the live weight tensors.
    """

    def __new__(cls, nn_module, **kwargs):
        if not isinstance(nn_module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(nn_module)}")
        parts = _extract_nn_module(nn_module, **kwargs)
        return Module.__new__(cls, **parts)

    def __init__(self, nn_module, **kwargs):
        nn.Module.__init__(self)
        self.inner = nn_module

    def forward(self, x):
        return self.inner(x)
