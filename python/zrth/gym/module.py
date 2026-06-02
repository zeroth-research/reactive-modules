import torch
import inspect
import gymnasium as gym

from ..zrth import Module, Wire, DType, Term
from ..builder import LRATermBuilder
from ..analyzer import (
    convert_method, classify_attrs, infer_dtype, wire_pair, resolve_wire,
    AbstractValue,
)
from ..eval import eval_itype, execute_init, execute_update, read_wire, getattr_wire


# ============================================================================
# Gym-specific helpers
# ============================================================================


def _value_to_const_term(value, wire, builder):
    """Create a constant Term that writes a Python value to a wire."""
    if isinstance(value, bool):
        return builder.const_bool(value, output_wire=wire)
    elif isinstance(value, (int, float)):
        tensor = torch.tensor([float(value)], dtype=torch.float32)
        return builder.const(tensor, output_wire=wire)
    elif isinstance(value, torch.Tensor):
        return builder.const(value.clone(), output_wire=wire)
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


def _setup_wire_pairs(module):
    """Extract wire pair references from obs. Returns a dict of pairs."""
    n_obs = len(module.obs)
    pairs = {}
    if n_obs >= 5:
        pairs['action'] = module.obs[0]
        pairs['observation'] = module.obs[1]
        pairs['reward'] = module.obs[2]
        pairs['terminated'] = module.obs[3]
        pairs['truncated'] = module.obs[4]
    elif n_obs >= 2:
        pairs['action'] = module.obs[0]
        pairs['observation'] = module.obs[1]
        pairs['reward'] = None
        pairs['terminated'] = None
        pairs['truncated'] = None
    else:
        raise ValueError(f"Module needs at least 2 observable wire pairs, got {n_obs}")
    return pairs


# ============================================================================
# Env extraction
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

    _builder = LRATermBuilder()

    action_dtype = _builder.space_to_dtype(env_instance.action_space, is_action=True)
    observation_dtype = _builder.space_to_dtype(env_instance.observation_space, is_action=False)

    init_attrs = _instance_to_init_attrs(env_instance)

    prvt, params, attr_vals = classify_attrs(
        env_cls, ['reset', 'step'], init_attrs=init_attrs, base_cls=gym.Env
    )

    prvt_wires = {name: wire_pair(infer_dtype(name, attr_vals.get(name), _builder)) for name in prvt}

    # Bake parameters as constant terms (written to temp wires, no param interface needed)
    const_wires = {}
    const_terms = []
    for name in params:
        value = getattr(env_instance, name)
        dtype = infer_dtype(name, AbstractValue.const(value), _builder)
        wire = Wire(dtype)
        const_wires[name] = [wire, wire]  # fake pair so analyzer resolves self.name
        const_terms.append(_value_to_const_term(value, wire, _builder))

    action      = resolve_wire("action",      action_dtype,      user_wires["action"])
    observation = resolve_wire("observation", observation_dtype, user_wires["observation"])
    reward      = resolve_wire("reward",      DType.Float([1]), user_wires["reward"])
    terminated  = resolve_wire("terminated",  DType.Bool([1]),  user_wires["terminated"])
    truncated   = resolve_wire("truncated",   DType.Bool([1]),  user_wires["truncated"])

    wires = {action_param: action, **prvt_wires, **const_wires}

    # gym reset returns (obs, info) → 1 result wire; step returns 4
    reset_result = [observation[1]]
    step_result  = [observation[1], reward[1], terminated[1], truncated[1]]

    reset_terms = convert_method(env_cls.reset, wires, reset_result, cls=env_cls, builder=_builder)
    step_terms  = convert_method(env_cls.step,  wires, step_result,  cls=env_cls, builder=_builder)

    # Add defaults for reward/terminated/truncated in init block
    reset_terms += [
        _value_to_const_term(0.0, reward[1], _builder),
        _value_to_const_term(False, terminated[1], _builder),
        _value_to_const_term(False, truncated[1], _builder),
    ]

    # Prepend constant terms so wires have values before they're read
    reset_terms = const_terms + reset_terms
    step_terms  = [_value_to_const_term(getattr(env_instance, n), const_wires[n][0], _builder)
                   for n in params] + step_terms

    obs = [action, observation, reward, terminated, truncated]

    # Build name → (latched, next) wire mapping
    wire_names = {}
    for name, pair in prvt_wires.items():
        wire_names[name] = (pair[0], pair[1])  # (latched, next)

    return dict(init=reset_terms, update=step_terms, obs=obs, prvt=list(prvt_wires.values()),
                _wire_names=wire_names)


# ============================================================================
# zrth.gym.Wrapper
# ============================================================================

class Wrapper(Module, gym.Wrapper):
    """A gym.Wrapper backed by a symbolic Module.

    Runs both the real gym.Env and the symbolic interpreter in lockstep.
    Gym methods (render, close, etc.) delegate to the original env.
    Symbolic state is available for inspection and validation.

    Usage:
        from zrth.gym import Wrapper

        Wrapper(gym_env)                        → extract Module, wrap env
        Wrapper(gym_env, module1, module2)      → extract + compose with other modules
        Wrapper(wrapped_env, module1)           → unwrap + compose
    """

    def __new__(cls, *args, **kwargs):
        raw_envs = []
        modules = []
        backing_env = None

        for a in args:
            if isinstance(a, Module):
                modules.append(a)
                if isinstance(a, Wrapper):
                    env = a.unwrapped
                    if backing_env is not None:
                        raise TypeError("Wrapper requires exactly 1 gym.Env, got multiple")
                    backing_env = env
            elif isinstance(a, gym.Env):
                raw_envs.append(a)
            else:
                raise TypeError(f"Expected gym.Env or Module, got {type(a)}")

        if len(raw_envs) > 1 or (len(raw_envs) == 1 and backing_env is not None):
            raise TypeError("Wrapper requires exactly 1 gym.Env, got multiple")

        if len(raw_envs) == 1:
            gym_env = raw_envs[0]
            extracted = _extract_env_module(gym_env, **kwargs)
            wire_names = extracted.pop('_wire_names', {})
            env_module = Module.__new__(Module, **extracted)
            env_ctrl_ids = {w for w in env_module.atoms[0].ctrl}
            modules = [env_module] + modules
            backing_env = gym_env
        else:
            wire_names = {}
            env_ctrl_ids = set()
            for a in args:
                if isinstance(a, Wrapper):
                    wire_names.update(a._wire_names)
                    if a._env_atom_idx is not None:
                        env_ctrl_ids = {w for w in a.atoms[a._env_atom_idx].ctrl}

        if backing_env is None:
            raise TypeError("Wrapper requires exactly 1 gym.Env, got 0")

        if len(modules) == 1:
            instance = Module.__new__(cls, modules[0])
        else:
            instance = Module.__new__(cls, *modules)
        instance._wire_names = wire_names
        instance._backing_env = backing_env

        instance._env_atom_idx = None
        for idx, atom in enumerate(instance.atoms):
            if {w for w in atom.ctrl} == env_ctrl_ids:
                instance._env_atom_idx = idx
                break

        return instance

    def __init__(self, *args, **kwargs):
        backing_env = object.__getattribute__(self, '_backing_env')
        gym.Wrapper.__init__(self, backing_env)
        self._state = {}
        self._initialized = False
        self._pairs = _setup_wire_pairs(self)

    def __getattr__(self, name):
        return getattr_wire(self, name)

    def get_prvt(self, name):
        """Look up a private wire pair by name. Returns (latched, next)."""
        wire_names = object.__getattribute__(self, '_wire_names')
        if name not in wire_names:
            raise KeyError(f"no private wire named '{name}'")
        return wire_names[name]

    def _sync_private_state_from_env(self):
        """Read private state from the real env and write to symbolic next wires."""
        wire_names = object.__getattribute__(self, '_wire_names')
        for name, (_, nxt) in wire_names.items():
            if nxt is None:
                continue
            value = getattr(self.env, name, None)
            if value is not None:
                if isinstance(value, bool):
                    self._state[nxt] = torch.tensor([value])
                elif isinstance(value, (int, float)):
                    self._state[nxt] = torch.tensor([float(value)])
                elif isinstance(value, torch.Tensor):
                    self._state[nxt] = value.clone()

    def reset(self, *, seed=None, options=None):
        self._state = {}
        p = self._pairs
        env_atom_idx = object.__getattribute__(self, '_env_atom_idx')

        gym_result = self.env.reset(seed=seed, options=options)
        reset_obs = gym_result[0] if isinstance(gym_result, tuple) else gym_result

        for atom_idx, atom in enumerate(self.atoms):
            if env_atom_idx is not None and atom_idx == env_atom_idx:
                obs_tensor = torch.as_tensor(reset_obs, dtype=torch.float32)
                if obs_tensor.dim() == 0:
                    obs_tensor = obs_tensor.unsqueeze(0)
                self._state[p['observation'][1]] = obs_tensor
                if p['reward']:
                    self._state[p['reward'][1]] = torch.tensor([0.0])
                if p['terminated']:
                    self._state[p['terminated'][1]] = torch.tensor([0.0])
                if p['truncated']:
                    self._state[p['truncated'][1]] = torch.tensor([0.0])
                self._sync_private_state_from_env()
            else:
                for term in atom.init:
                    read = [self._state[w] for w in term.read]
                    results = eval_itype(term.itype, read)
                    for w, val in zip(term.write, results):
                        self._state[w] = val

        for ltc, nxt in self.ctrl:
            if nxt in self._state:
                self._state[ltc] = self._state[nxt].clone()
        self._initialized = True
        return read_wire(self._state, p['observation'][0]).numpy(), {}

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        p = self._pairs
        env_atom_idx = object.__getattribute__(self, '_env_atom_idx')

        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if isinstance(getattr(self.env, 'action_space', None), gym.spaces.Discrete):
            if action_tensor.numel() == 1:
                idx = int(action_tensor.item())
                one_hot = torch.zeros(self.env.action_space.n)
                one_hot[idx] = 1.0
                action_tensor = one_hot
        self._state[p['action'][0]] = action_tensor

        for atom_idx, atom in enumerate(self.atoms):
            if env_atom_idx is not None and atom_idx == env_atom_idx:
                gym_result = self.env.step(self._state[p['action'][0]])
                obs, reward, terminated, truncated = gym_result[0], gym_result[1], gym_result[2], gym_result[3]
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                if obs_tensor.dim() == 0:
                    obs_tensor = obs_tensor.unsqueeze(0)
                self._state[p['observation'][1]] = obs_tensor
                if p['reward']:
                    self._state[p['reward'][1]] = torch.tensor([float(reward)])
                if p['terminated']:
                    self._state[p['terminated'][1]] = torch.tensor([1.0 if terminated else 0.0])
                if p['truncated']:
                    self._state[p['truncated'][1]] = torch.tensor([1.0 if truncated else 0.0])
                self._sync_private_state_from_env()
            else:
                for term in atom.update:
                    read = [self._state[w] for w in term.read]
                    results = eval_itype(term.itype, read)
                    for w, val in zip(term.write, results):
                        self._state[w] = val

        for ltc, nxt in self.ctrl:
            if nxt in self._state:
                self._state[ltc] = self._state[nxt].clone()
        obs = read_wire(self._state, p['observation'][0])
        reward = read_wire(self._state, p['reward'][0]).item() if p['reward'] else 0.0
        terminated = bool(read_wire(self._state, p['terminated'][0]).item()) if p['terminated'] else False
        truncated = bool(read_wire(self._state, p['truncated'][0]).item()) if p['truncated'] else False
        return obs.numpy(), reward, terminated, truncated, {}


# ============================================================================
# zrth.gym.Env
# ============================================================================

class Env(Module, gym.Env):
    """A gym.Env that runs a symbolic Module via the term interpreter.

    For pure Modules (no backing gym.Env). Use Wrapper() for gym.Env instances.

    Usage:
        from zrth.gym import Env

        Env(module)                → run a single module
        Env(module1, module2)      → compose and run
    """

    def __new__(cls, *modules):
        for m in modules:
            if isinstance(m, gym.Env) and not isinstance(m, Module):
                raise TypeError("Use Wrapper() for gym.Env instances, Env is for pure Modules")
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
        self._pairs = _setup_wire_pairs(self)

    def __getattr__(self, name):
        return getattr_wire(self, name)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)
        self._state = {}

        execute_init(self._state, self.atoms)
        for ltc, nxt in self.ctrl:
            if nxt in self._state:
                self._state[ltc] = self._state[nxt].clone()
        self._initialized = True
        return read_wire(self._state, self._pairs['observation'][0]).numpy(), {}

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        p = self._pairs
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        self._state[p['action'][0]] = action_tensor
        execute_update(self._state, self.atoms)
        for ltc, nxt in self.ctrl:
            if nxt in self._state:
                self._state[ltc] = self._state[nxt].clone()

        obs = read_wire(self._state, p['observation'][0])
        reward = read_wire(self._state, p['reward'][0]).item() if p['reward'] else 0.0
        terminated = bool(read_wire(self._state, p['terminated'][0]).item()) if p['terminated'] else False
        truncated = bool(read_wire(self._state, p['truncated'][0]).item()) if p['truncated'] else False

        return obs.numpy(), reward, terminated, truncated, {}

    def get(self, wire):
        """Retrieve the current value of a wire."""
        if wire not in self._state:
            raise RuntimeError(f"wire {wire} not in state")
        return self._state[wire].clone()

    def state_dict(self):
        """Return a copy of the current state."""
        return dict(self._state)
