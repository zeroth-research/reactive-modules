import torch
from zrth import Module, Wire, DType, IType, Term
from zrth.analyzer import (
    convert_method, analyze_init, infer_spaces, infer_layers,
    classify_attrs, infer_dtype, wire_pair, resolve_wire, wrap_init,
    AbstractValue,
)
from zrth.eval import eval_itype, zero_tensor
import gymnasium as gym
import torch.nn as nn
import inspect


# ============================================================================
# _ModuleEnv: wraps a Module as a runnable gym.Env
# ============================================================================

class _ModuleEnv(gym.Env):
    """A gym.Env backed by a symbolic Module, evaluated via the term interpreter."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self._state = {}
        self._initialized = False

        # Convention: obs = [action, observation, reward, terminated, truncated]
        n_obs = len(module.obs)
        if n_obs >= 5:
            self._action_pair = module.obs[0]
            self._observation_pair = module.obs[1]
            self._reward_pair = module.obs[2]
            self._terminated_pair = module.obs[3]
            self._truncated_pair = module.obs[4]
        elif n_obs >= 2:
            # Minimal: first pair is input, second is output
            self._action_pair = module.obs[0]
            self._observation_pair = module.obs[1]
            self._reward_pair = None
            self._terminated_pair = None
            self._truncated_pair = None
        else:
            raise ValueError(f"Module needs at least 2 observable wire pairs, got {n_obs}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._state = {}
        self._init_wires()
        self._execute("init")
        self._latch()
        self._initialized = True
        obs = self._read_wire(self._observation_pair[0])
        return obs.numpy(), {}

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        # Load action into the next wire of the action pair
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        self._state[self._action_pair[1].id] = action_tensor

        self._execute("update")
        self._latch()

        obs = self._read_wire(self._observation_pair[0])
        reward = self._read_wire(self._reward_pair[0]).item() if self._reward_pair else 0.0
        terminated = bool(self._read_wire(self._terminated_pair[0]).item()) if self._terminated_pair else False
        truncated = bool(self._read_wire(self._truncated_pair[0]).item()) if self._truncated_pair else False

        return obs.numpy(), reward, terminated, truncated, {}

    def _init_wires(self):
        """Zero-initialize all external and parameter wires."""
        extl = self.module.extl
        for i in range(len(extl)):
            ltc, nxt = extl[i]
            for w in (ltc, nxt):
                if w.id not in self._state:
                    self._state[w.id] = zero_tensor(w.dtype)
        param = self.module.param
        for i in range(len(param)):
            w = param[i]
            if w.id not in self._state:
                self._state[w.id] = zero_tensor(w.dtype)

    def _execute(self, block_type):
        atoms = self.module.atoms
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
        ctrl = self.module.ctrl
        for i in range(len(ctrl)):
            ltc, nxt = ctrl[i]
            nxt_id = nxt.id
            if nxt_id in self._state:
                self._state[ltc.id] = self._state[nxt_id].clone()

    def _read_wire(self, wire):
        return self._state[wire.id].clone()


# ============================================================================
# Helpers for forward wrapping (instance → Module)
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
            # Reconstruct call_repr for spaces so infer_spaces can work
            attrs[name] = _space_to_abstract(name, value)
        # Skip non-analyzable attributes
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


def _wrap_gym_env(env_instance, **kwargs):
    """Forward wrap: extract Module from a gym.Env instance."""
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

    # Infer spaces directly from the instance's attributes
    action_dtype = _space_to_dtype(env_instance.action_space, is_action=True)
    observation_dtype = _space_to_dtype(env_instance.observation_space, is_action=False)

    # Reconstruct init_attrs from live instance for classify_attrs fallback
    init_attrs = _instance_to_init_attrs(env_instance)

    prvt, params, attr_vals = classify_attrs(
        env_cls, ['reset', 'step'], init_attrs=init_attrs, base_cls=gym.Env
    )

    prvt_wires = {name: wire_pair(infer_dtype(name, attr_vals.get(name))) for name in prvt}

    # Bake parameters as constant terms instead of param wires
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

    wires  = {action_param: action, **prvt_wires}

    # gym reset returns (obs, info) → 1 result wire; step returns (obs, rew, term, trunc, info) → 4
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
    module = Module(init=reset_terms, update=step_terms, obs=obs, prvt=list(prvt_wires.values()))

    # Return a _ModuleEnv that also holds the original env for delegation
    wrapped = _WrappedEnv(module, env_instance)
    wrapped.action      = action
    wrapped.observation = observation
    wrapped.reward      = reward
    wrapped.terminated  = terminated
    wrapped.truncated   = truncated
    return wrapped


class _WrappedEnv(gym.Wrapper):
    """Wraps a gym.Env and attaches an extracted Module."""

    def __init__(self, module, env):
        super().__init__(env)
        self.module = module


# ============================================================================
# Env
# ============================================================================

class Env(Module, gym.Env):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        wrap_init(cls, ("action", "observation", "reward", "terminated", "truncated"))

    def __new__(cls, *args, **kwargs):
        # Reverse wrapping: Env(module) → runnable gym.Env
        if cls is Env and args and isinstance(args[0], Module):
            return _ModuleEnv(args[0])

        # Forward wrapping: Env(gym_env_instance) → Module + gym.Wrapper
        if cls is Env and args and isinstance(args[0], gym.Env):
            return _wrap_gym_env(args[0], **kwargs)

        action_param = next(p for p in inspect.signature(cls.step).parameters if p != "self")

        user_wires = {
            "action":      kwargs.pop("action",      None),
            "observation": kwargs.pop("observation", None),
            "reward":      kwargs.pop("reward",      None),
            "terminated":  kwargs.pop("terminated",  None),
            "truncated":   kwargs.pop("truncated",   None),
        }

        init_attrs = analyze_init(cls, args, kwargs)
        action_dtype, observation_dtype = infer_spaces(init_attrs)
        prvt, params, attr_vals = classify_attrs(
            cls, ['reset', 'step'], init_attrs=init_attrs, base_cls=Env
        )

        prvt_wires  = {name: wire_pair(infer_dtype(name, attr_vals.get(name))) for name in prvt}
        param_wires = {name: Wire(infer_dtype(name, attr_vals.get(name))) for name in params}

        action      = resolve_wire("action",      action_dtype,      user_wires["action"])
        observation = resolve_wire("observation", observation_dtype, user_wires["observation"])
        reward      = resolve_wire("reward",      DType.Float([1]), user_wires["reward"])
        terminated  = resolve_wire("terminated",  DType.Bool([1]),  user_wires["terminated"])
        truncated   = resolve_wire("truncated",   DType.Bool([1]),  user_wires["truncated"])

        wires  = {action_param: action, **prvt_wires}
        result = [observation[1], reward[1], terminated[1], truncated[1]]

        reset = convert_method(cls.reset, wires, result, cls=cls, params=param_wires)
        step  = convert_method(cls.step,  wires, result, cls=cls, params=param_wires)

        obs      = [action, observation, reward, terminated, truncated]
        instance = super().__new__(cls, init=reset, update=step, obs=obs, prvt=list(prvt_wires.values()))

        instance.action      = action
        instance.observation = observation
        instance.reward      = reward
        instance.terminated  = terminated
        instance.truncated   = truncated

        return instance

# ============================================================================
# NN
# ============================================================================

def _wrap_nn_module(nn_instance, **kwargs):
    """Forward wrap: extract Module from an nn.Module instance."""
    nn_cls = type(nn_instance)

    user_extl = kwargs.pop("extl", None)
    user_intf = kwargs.pop("intf", None)

    obs_param = next(
        p for p in inspect.signature(nn_cls.forward).parameters if p != "self"
    )

    # Infer layer structure from the actual nn.Module instance
    layers = {}
    for name, child in nn_instance.named_modules():
        if isinstance(child, nn.Linear):
            layers[name] = (child.in_features, child.out_features)

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
    forward = convert_method(nn_cls.forward, wires, result, cls=nn_cls, layers=layer_out_features)

    obs    = [extl, intf]
    module = Module(assign=forward, obs=obs)

    # Attach the module and original nn instance for reference
    nn_instance._module = module
    return module


class NN(Module, nn.Module):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        wrap_init(cls, ("extl", "intf"))

    def __new__(cls, *args, **kwargs):
        # Forward wrapping: NN(nn_module_instance) → Module
        if cls is NN and args and isinstance(args[0], nn.Module):
            return _wrap_nn_module(args[0], **kwargs)

        obs_param = next(p for p in inspect.signature(cls.forward).parameters if p != "self")

        user_extl = kwargs.pop("extl", None)
        user_intf = kwargs.pop("intf", None)

        init_attrs = analyze_init(cls, args, kwargs)
        layers     = infer_layers(init_attrs)
        layer_list = list(layers.values())
        obs_size   = layer_list[0][0]   # first layer's in_features
        qval_size  = layer_list[-1][1]  # last layer's out_features

        extl = resolve_wire("extl", DType.Float([obs_size]),  user_extl)
        intf = resolve_wire("intf", DType.Float([qval_size]), user_intf)

        # Combinatorial modules have no latched state, so the "input" wire is
        # index 1 (next) rather than index 0 (latched). The converter always
        # reads index 0 as the input, so we swap the pair here.
        wires  = {obs_param: [extl[1], extl[0]]}
        result = [intf[1]]

        layer_out_features = {name: out for name, (_, out) in layers.items()}
        forward = convert_method(cls.forward, wires, result, cls=cls, layers=layer_out_features)

        obs      = [extl, intf]
        instance = super().__new__(cls, assign=forward, obs=obs)

        # instance.extl = extl
        # instance.intf = intf

        return instance
