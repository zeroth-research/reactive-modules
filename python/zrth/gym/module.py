import torch
import inspect
import gymnasium as gym

from ..zrth import Module, Wire, DType, Term
from .. import IType
from ..builder import LRATermBuilder
from ..analyzer import (
    convert_method, classify_attrs, infer_dtype, wire_pair, resolve_wire,
    AbstractValue,
)
from ..eval import eval_itype, read_wire, getattr_wire


# ============================================================================
# Gym-specific helpers
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
        return Term(IType.LRA.ConstBool(value), [wire], [])
    elif isinstance(value, (int, float)):
        tensor = torch.tensor([float(value)], dtype=torch.float32)
        return Term(IType.LRA.ConstReal(tensor), [wire], [])
    elif isinstance(value, torch.Tensor):
        return Term(IType.LRA.ConstReal(value.clone()), [wire], [])
    else:
        raise ValueError(f"Cannot create constant term for {type(value).__name__}: {value}")


def _instance_to_init_attrs(instance):
    """Reconstruct init_attrs dict from a live instance's __dict__."""
    import numpy as np
    attrs = {}
    for name, value in instance.__dict__.items():
        if isinstance(value, (int, float, bool)):
            attrs[name] = AbstractValue.const(value)
        elif isinstance(value, torch.Tensor):
            attrs[name] = AbstractValue.const(value)
        elif isinstance(value, np.ndarray):
            # Represent as np.array(<list>) CallResult so infer_dtype can extract shape
            attrs[name] = AbstractValue.call_result(f"np.array({list(value.tolist())})")
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
# Env extraction
# ============================================================================

def _extract_env_module(env_instance, **kwargs):
    """Analyze a gym.Env instance and extract a symbolic Module.

    If env_instance is wrapped (e.g. TimeLimit from gym.make), the raw inner
    env is analyzed symbolically, but the wrapped instance remains the
    backing env for runtime delegation — so TimeLimit/OrderEnforcing etc.
    still apply during step/reset.
    """
    raw = env_instance.unwrapped
    env_cls = type(raw)

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

    # Reset so runtime-created attrs (e.g. self.state) are populated.
    # Reset the outermost wrapper so wrapper state is also initialized.
    env_instance.reset()

    init_attrs = _instance_to_init_attrs(raw)

    prvt, params, attr_vals = classify_attrs(
        env_cls, ['reset', 'step'], init_attrs=init_attrs, base_cls=gym.Env
    )

    prvt_wires = {name: wire_pair(infer_dtype(name, attr_vals.get(name))) for name in prvt}

    # Bake parameters as constant terms (written to temp wires, no param interface needed).
    # Primitives that can't be wire-typed (str, None) become static_attrs for compile-time use.
    const_wires = {}
    const_terms = []
    static_attrs = {}
    for name in params:
        value = getattr(env_instance, name)
        dtype = infer_dtype(name, AbstractValue.const(value))
        # Real-valued theories (LRA) have no Int type; promote scalar ints to Float.
        if dtype.is_int():
            dtype = DType.Float(list(dtype.shape))
        wire = Wire(dtype)
        const_wires[name] = [wire, wire]  # fake pair so analyzer resolves self.name
        const_terms.append(_value_to_const_term(value, wire))

    # Also expose any other primitive instance attributes as static values.
    # Useful when classify_attrs couldn't trace them (e.g. inside untraceable branches).
    for name, value in raw.__dict__.items():
        if name in static_attrs or name in const_wires or name in prvt_wires:
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            static_attrs[name] = value

    action      = resolve_wire("action",      action_dtype,      user_wires["action"])
    observation = resolve_wire("observation", observation_dtype, user_wires["observation"])
    reward      = resolve_wire("reward",      DType.Float([1]), user_wires["reward"])
    terminated  = resolve_wire("terminated",  DType.Bool([1]),  user_wires["terminated"])
    truncated   = resolve_wire("truncated",   DType.Bool([1]),  user_wires["truncated"])

    wires = {action_param: action, **prvt_wires, **const_wires}

    # gym reset returns (obs, info) → 1 result wire; step returns 4
    reset_result = [observation[1]]
    step_result  = [observation[1], reward[1], terminated[1], truncated[1]]

    _builder = LRATermBuilder()
    reset_terms = convert_method(env_cls.reset, wires, reset_result, cls=env_cls, builder=_builder)
    step_terms  = convert_method(env_cls.step,  wires, step_result,  cls=env_cls, builder=_builder)

    # Add defaults for reward/terminated/truncated in init block
    reset_terms += [
        Term(IType.LRA.ConstReal(torch.tensor([0.0])), [reward[1]], []),
        Term(IType.LRA.ConstBool(False), [terminated[1]], []),
        Term(IType.LRA.ConstBool(False), [truncated[1]], []),
    ]

    # Prepend constant terms so wires have values before they're read
    reset_terms = const_terms + reset_terms
    step_terms  = [_value_to_const_term(getattr(raw, n), const_wires[n][0])
                   for n in const_wires] + step_terms

    obs = [action, observation, reward, terminated, truncated]

    # Build name → (latched, next) wire mapping
    wire_names = {}
    for name, pair in prvt_wires.items():
        wire_names[name] = (pair[0], pair[1])  # (latched, next)

    return dict(init=reset_terms, update=step_terms, obs=obs, prvt=list(prvt_wires.values()),
                _wire_names=wire_names)


# ============================================================================
# zrth.gym.Env — unified wrapper for gym.Env and pure symbolic Modules
# ============================================================================

class Env(Module, gym.Wrapper):
    """A gym environment backed by a symbolic Module.

    Accepts gym.Env instances (with real env delegation) and/or pure Modules
    (fully symbolic). At most one gym.Env is allowed.

    Usage:
        from zrth.gym import Env

        Env(gym_env)                        → extract Module, wrap env
        Env(gym_env, module1, module2)      → extract + compose
        Env(wrapped_env, module1)           → unwrap + compose
        Env(module)                         → pure symbolic
        Env(module1, module2)               → compose + pure symbolic
        Env(..., interpret=True)            → run every atom through the IR interpreter
                                              (no real-env delegation)
    """

    def __new__(cls, *args, **kwargs):
        interpret = kwargs.pop('interpret', False)
        raw_envs = []
        modules = []
        backing_env = None

        for a in args:
            if isinstance(a, Module):
                modules.append(a)
                # If it's already an Env with a backing env, unwrap
                if isinstance(a, Env) and a._backing_env is not None:
                    if backing_env is not None:
                        raise ValueError("Env accepts at most 1 gym.Env, got multiple")
                    backing_env = a.unwrapped
            elif isinstance(a, gym.Env):
                raw_envs.append(a)
            else:
                raise TypeError(f"Expected gym.Env or Module, got {type(a)}")

        if len(raw_envs) > 1 or (len(raw_envs) == 1 and backing_env is not None):
            raise ValueError("Env accepts at most 1 gym.Env, got multiple")

        # Extract Module from raw gym.Env if present
        wire_names = {}
        env_ctrl_ids = set()

        if len(raw_envs) == 1:
            gym_env = raw_envs[0]
            extracted = _extract_env_module(gym_env, **kwargs)
            wire_names = extracted.pop('_wire_names', {})
            env_module = Module.__new__(Module, **extracted)
            env_ctrl_ids = {w for w in env_module.atoms[0].ctrl}
            modules = [env_module] + modules
            backing_env = gym_env
        else:
            # Inherit wire_names and env atom ctrl IDs from source Env
            for a in args:
                if isinstance(a, Env):
                    wire_names.update(a._wire_names)
                    if a._env_atom_idx is not None:
                        env_ctrl_ids = {w for w in a.atoms[a._env_atom_idx].ctrl}

        if not modules:
            raise TypeError("Env requires at least 1 argument")

        # Compose all modules
        if len(modules) == 1:
            instance = Module.__new__(cls, modules[0])
        else:
            instance = Module.__new__(cls, *modules)

        instance._wire_names = wire_names
        instance._backing_env = None if interpret else backing_env

        # Find env atom index in composed module (may be reordered by topo sort)
        instance._env_atom_idx = None
        if env_ctrl_ids and not interpret:
            for idx, atom in enumerate(instance.atoms):
                if {w for w in atom.ctrl} == env_ctrl_ids:
                    instance._env_atom_idx = idx
                    break

        return instance

    def __init__(self, *args, **kwargs):
        backing_env = object.__getattribute__(self, '_backing_env')
        if backing_env is not None:
            gym.Wrapper.__init__(self, backing_env)
        self._state = {}
        self._initialized = False

        # Extract wire pairs from obs
        n_obs = len(self.obs)
        self._pairs = {}
        if n_obs >= 5:
            self._pairs = {'action': self.obs[0], 'observation': self.obs[1],
                           'reward': self.obs[2], 'terminated': self.obs[3], 'truncated': self.obs[4]}
        elif n_obs >= 2:
            self._pairs = {'action': self.obs[0], 'observation': self.obs[1],
                           'reward': None, 'terminated': None, 'truncated': None}
        else:
            raise ValueError(f"Module needs at least 2 observable wire pairs, got {n_obs}")

        # If a composed atom drives the action wire (e.g. a controller in a
        # closed loop), step()'s `action` argument is ignored — the controller's
        # latched output is what the env atom sees.
        action_next = self._pairs['action'][1]
        self._action_driven = any(
            action_next in {w for w in atom.ctrl}
            for idx, atom in enumerate(self.atoms)
            if idx != self._env_atom_idx
        )

    def __getattr__(self, name):
        return getattr_wire(self, name)

    def get_prvt(self, name):
        """Look up a private wire pair by name. Returns (latched, next)."""
        wire_names = object.__getattribute__(self, '_wire_names')
        if name not in wire_names:
            raise KeyError(f"no private wire named '{name}'")
        return wire_names[name]

    def get(self, wire):
        """Retrieve the current value of a wire."""
        if wire not in self._state:
            raise RuntimeError(f"wire {wire} not in state")
        return self._state[wire].clone()

    def state_dict(self):
        """Return a copy of the current state."""
        return dict(self._state)

    # ── Execution ─────────────────────────────────────────────

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

    def _prepare_action(self, action):
        """Convert action to tensor, one-hot encode for Discrete spaces."""
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if self._backing_env and isinstance(
            getattr(self._backing_env, 'action_space', None), gym.spaces.Discrete
        ):
            if action_tensor.numel() == 1:
                idx = int(action_tensor.item())
                one_hot = torch.zeros(self._backing_env.action_space.n)
                one_hot[idx] = 1.0
                action_tensor = one_hot
        return action_tensor

    def reset(self, *, seed=None, options=None):
        self._state = {}
        p = self._pairs
        env_atom_idx = self._env_atom_idx

        # Reset real env if present
        reset_obs = None
        if self._backing_env:
            gym_result = self._backing_env.reset(seed=seed, options=options)
            reset_obs = gym_result[0] if isinstance(gym_result, tuple) else gym_result

        # Execute init block for each atom
        for atom_idx, atom in enumerate(self.atoms):
            if env_atom_idx is not None and atom_idx == env_atom_idx:
                # Env atom: write real env state to symbolic wires
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
                # Symbolic atom
                for term in atom.init:
                    read = [self._state[w] for w in term.read]
                    results = eval_itype(term.itype, read)
                    for w, val in zip(term.write, results):
                        self._state[w] = val

        # Latch
        for ltc, nxt in self.ctrl:
            if nxt in self._state:
                self._state[ltc] = self._state[nxt].clone()

        self._initialized = True
        return read_wire(self._state, p['observation'][0]).numpy(), {}

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        p = self._pairs
        env_atom_idx = self._env_atom_idx

        # Write action to symbolic state, unless an upstream atom drives it
        # (closed-loop with a composed controller — its latched output is used).
        if not self._action_driven:
            self._state[p['action'][0]] = self._prepare_action(action)

        # Execute update block for each atom
        for atom_idx, atom in enumerate(self.atoms):
            if env_atom_idx is not None and atom_idx == env_atom_idx:
                # Env atom: run real env
                gym_result = self._backing_env.step(self._state[p['action'][0]].detach())
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
                # Symbolic atom
                for term in atom.update:
                    read = [self._state[w] for w in term.read]
                    results = eval_itype(term.itype, read)
                    for w, val in zip(term.write, results):
                        self._state[w] = val

        # Latch
        for ltc, nxt in self.ctrl:
            if nxt in self._state:
                self._state[ltc] = self._state[nxt].clone()

        obs = read_wire(self._state, p['observation'][0])
        reward = read_wire(self._state, p['reward'][0]).item() if p['reward'] else 0.0
        terminated = bool(read_wire(self._state, p['terminated'][0]).item()) if p['terminated'] else False
        truncated = bool(read_wire(self._state, p['truncated'][0]).item()) if p['truncated'] else False
        return obs.numpy(), reward, terminated, truncated, {}
