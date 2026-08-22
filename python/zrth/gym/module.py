import math
import torch
import inspect
import gymnasium as gym

from ..zrth import Module, Wire, LIA, BV, X, Var
from ..sort import Sort, Bool, Real, Int, BitVec
from ..builder import builder_for, _normalize_shape, _shape
from ..analyzer import (
    convert_method,
    classify_attrs,
    infer_dtype,
    wire_pair,
    resolve_wire,
    AbstractValue,
)
from ..eval import eval_itype, read_wire, getattr_wire


# ============================================================================
# Gym-specific helpers
# ============================================================================


def space_to_dtype(space, theory, is_action: bool) -> Sort:
    """Convert a gym.spaces.Space to a Sort for the given theory (always 2-D)."""
    if isinstance(space, gym.spaces.Discrete):
        n = space.n
        if theory is BV:
            bits = max(1, int(math.ceil(math.log2(n + 1))))
            return BitVec(bits, [1, 1])
        if is_action:
            return Real([1, n])
        return Int([1, 1]) if theory is LIA else Real([1, 1])
    elif isinstance(space, gym.spaces.Box):
        shape = _normalize_shape(list(space.shape))
        if theory is BV:
            return BitVec(32, shape)
        return Int(shape) if theory is LIA else Real(shape)
    elif isinstance(space, gym.spaces.MultiBinary):
        return Bool([1, space.n])
    else:
        raise ValueError(f"Unsupported gym space type: {type(space).__name__}")


def _normalize_attrs(attrs, prvt_names):
    """Resolve the user-supplied `attrs` into a {name: Sort | (latched, next) pair}
    override map for private-state wires.

    `attrs` may be:
      - None                          -> no overrides (everything inferred)
      - a single Sort                 -> apply that sort to every private attr
      - a dict {name: Sort | pair}    -> per-attr override; other attrs stay inferred

    Names not among the private-state attributes raise (typo guard). Unlisted attrs
    fall back to inference at the call site.
    """
    if attrs is None:
        return {}
    if isinstance(attrs, Sort):
        return {name: attrs for name in prvt_names}
    if isinstance(attrs, dict):
        unknown = set(attrs) - set(prvt_names)
        if unknown:
            raise ValueError(
                f"attrs names {sorted(unknown)} are not private-state attributes; "
                f"known: {sorted(prvt_names)}"
            )
        return dict(attrs)
    raise TypeError(
        f"attrs must be a Sort or a dict of them, got {type(attrs).__name__}"
    )


def _resolve_attr_wire(name, spec):
    """Turn an `attrs` override (a Sort, or a (latched, next) wire pair) into a
    validated [latched, next] wire pair."""
    if isinstance(spec, Sort):
        return wire_pair(spec)
    is_var = (
        isinstance(spec, Var)
        # and len(spec) == 2
        # and all(isinstance(w, Wire) for w in spec)
    )
    if is_var:
        if spec.dtype != X(spec).dtype:
            raise ValueError(
                f"attrs['{name}']: latched/next dtype mismatch "
                f"({spec.dtype} vs {X(spec).dtype})"
            )
        return spec
    raise TypeError(
        f"attrs['{name}'] must be a Sort or a (latched, next) wire pair, "
        f"got {type(spec).__name__}"
    )


def _value_to_const_term(value, wire, builder):
    """Create a constant Term that writes a Python value to a wire."""
    if isinstance(value, torch.Tensor):
        return builder.const(value.clone(), output_wire=wire)
    return builder.const_for_value(value, output_wire=wire)


def _ensure_1d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.dim() == 0 else t


def _to_wire_shape(t: torch.Tensor, wire) -> torch.Tensor:
    """Reshape a tensor to the wire's (2-D normalized) Sort shape when sizes match.

    Symbolic ops (e.g. Linear/Transpose) expect 2-D operands; the real env emits
    flat 1-D obs, so conform it to the declared wire shape for downstream atoms."""
    shape = _shape(wire.dtype)
    if t.numel() == int(torch.tensor(shape).prod().item()):
        return t.reshape(shape)
    return _ensure_1d(t)


# ============================================================================
# Env extraction
# ============================================================================


def _extract_env_module(env_instance, theory=None, **kwargs):
    """Analyze a gym.Env instance and extract a symbolic Module.

    If env_instance is wrapped (e.g. TimeLimit from gym.make), the raw inner
    env is analyzed symbolically, but the wrapped instance remains the
    backing env for runtime delegation — so TimeLimit/OrderEnforcing etc.
    still apply during step/reset.
    """
    raw = env_instance.unwrapped
    env_cls = type(raw)

    user_wires = {
        "action": kwargs.pop("action", None),
        "observation": kwargs.pop("observation", None),
        "reward": kwargs.pop("reward", None),
        "terminated": kwargs.pop("terminated", None),
        "truncated": kwargs.pop("truncated", None),
    }
    attrs = kwargs.pop("attrs", None)

    action_param = next(
        p for p in inspect.signature(env_cls.step).parameters if p != "self"
    )

    _builder = builder_for(theory)

    action_dtype = space_to_dtype(env_instance.action_space, theory, is_action=True)
    observation_dtype = space_to_dtype(
        env_instance.observation_space, theory, is_action=False
    )

    # Reset so runtime-created attrs (e.g. self.state) are populated, then
    # classify which self.* are private state vs read-only parameters (AST-based).
    env_instance.reset()
    prvt, params, _ = classify_attrs(env_cls, ["reset", "step"], base_cls=gym.Env)

    # Private-state sorts are always explicit: `attrs` must declare every one.
    attr_overrides = _normalize_attrs(attrs, prvt)
    missing = [n for n in prvt if n not in attr_overrides]
    if missing:
        raise ValueError(
            f"private-state sorts must be declared via attrs=; missing {sorted(missing)}"
        )
    prvt_wires = {name: _resolve_attr_wire(name, attr_overrides[name]) for name in prvt}

    # Bake parameters as constant terms (written to temp wires, no param interface
    # needed). Primitives that can't be wire-typed (str, None) become static_attrs
    # for compile-time use.
    const_wires = {}
    const_terms = []
    static_attrs = {}
    for name in params:
        value = getattr(raw, name)
        if value is None or isinstance(value, str):
            static_attrs[name] = value
            continue
        try:
            dtype = infer_dtype(name, AbstractValue.const(value), _builder)
        except ValueError:
            static_attrs[name] = value
            continue
        wire = Wire(dtype)
        const_wires[name] = wire  # a read-only constant: a single wire
        const_terms.append(_value_to_const_term(value, wire, _builder))

    # Also expose any other primitive instance attributes as static values.
    # Useful when classify_attrs couldn't trace them (e.g. inside untraceable branches).
    for name, value in raw.__dict__.items():
        if name in static_attrs or name in const_wires or name in prvt_wires:
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            static_attrs[name] = value

    action = resolve_wire("action", action_dtype, user_wires["action"])
    observation = resolve_wire(
        "observation", observation_dtype, user_wires["observation"]
    )
    reward = resolve_wire("reward", Real([1, 1]), user_wires["reward"])
    terminated = resolve_wire("terminated", Bool([1, 1]), user_wires["terminated"])
    truncated = resolve_wire("truncated", Bool([1, 1]), user_wires["truncated"])

    wires = {action_param: action, **prvt_wires, **const_wires}

    # gym reset returns (obs, info) → 1 result wire; step returns 4
    reset_result = [X(observation)]
    step_result = [X(observation), X(reward), X(terminated), X(truncated)]

    reset_terms = convert_method(
        env_cls.reset, wires, reset_result, cls=env_cls,
        theory=theory, static_attrs=static_attrs,
    )
    step_terms = convert_method(
        env_cls.step, wires, step_result, cls=env_cls,
        theory=theory, static_attrs=static_attrs,
    )

    # Add defaults for reward/terminated/truncated in init block
    reset_terms += [
        _value_to_const_term(0.0, X(reward), _builder),
        _value_to_const_term(False, X(terminated), _builder),
        _value_to_const_term(False, X(truncated), _builder),
    ]

    # Prepend constant terms so wires have values before they're read
    reset_terms = const_terms + reset_terms
    step_terms = [
                     _value_to_const_term(getattr(raw, n), const_wires[n], _builder)
                     for n in const_wires
                 ] + step_terms

    obs = [action, observation, reward, terminated, truncated]

    # Build name → (latched, next) wire mapping
    wire_names = {name: pair for name, pair in prvt_wires.items()}
    wire_names["action"] = action
    wire_names["observation"] = observation
    wire_names["reward"] = reward
    wire_names["terminated"] = terminated
    wire_names["truncated"] = truncated

    return dict(
        init=reset_terms,
        update=step_terms,
        vars=obs + list(prvt_wires.values()),
        hide=list(prvt_wires.values()),
        _wire_names=wire_names,
    )


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
        Env(..., theory=LIA|BV|LRA)         → select the term theory (default LRA)
        Env(gym_env, attrs=...)             → declare the private-state sorts/wires
                                              (required when the env has private state):
                                              a Sort for all, or {name: Sort | (latched, next)}
                                              per attr
    """

    def __new__(cls, *args, **kwargs):
        interpret = kwargs.pop("interpret", False)
        theory = kwargs.pop("theory", None)
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

        if "attrs" in kwargs and len(raw_envs) != 1:
            raise TypeError("attrs= only applies when extracting from a gym.Env")

        # Extract Module from raw gym.Env if present
        wire_names = {}
        env_ctrl_ids = set()

        if len(raw_envs) == 1:
            gym_env = raw_envs[0]
            extracted = _extract_env_module(gym_env, theory=theory, **kwargs)
            wire_names = extracted.pop("_wire_names", {})
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
        backing_env = object.__getattribute__(self, "_backing_env")
        if backing_env is not None:
            gym.Wrapper.__init__(self, backing_env)
        self._state = {}
        self._initialized = False

        # Extract wire pairs from obs
        n_obs = len(self.obs)
        if n_obs >= 5:
            self._pairs = {
                "action": self._wire_names["action"],
                "observation": self._wire_names["observation"],
                "reward": self._wire_names["reward"],
                "terminated": self._wire_names["terminated"],
                "truncated": self._wire_names["truncated"],
            }
        elif n_obs >= 2:
            self._pairs = {
                # TODO this positional/dimensional logic is dangerous
                "action": self.obs[0],
                "observation": self.obs[1],
                "reward": None,
                "terminated": None,
                "truncated": None,
            }
        else:
            raise ValueError(
                f"Module needs at least 2 observable wire pairs, got {n_obs}"
            )

        # If a composed atom drives the action wire (e.g. a controller in a
        # closed loop), step()'s `action` argument is ignored — the controller's
        # latched output is what the env atom sees.
        action_next = X(self._pairs["action"])
        self._action_driven = any(
            action_next in {X(w) for w in atom.ctrl}
            for idx, atom in enumerate(self.atoms)
            if idx != self._env_atom_idx
        )

    def __getattr__(self, name):
        return getattr_wire(self, name)

    def get_prvt(self, name):
        """Look up a private wire pair by name. Returns (latched, next)."""
        wire_names = object.__getattribute__(self, "_wire_names")
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
        wire_names = object.__getattribute__(self, "_wire_names")
        for name, var in wire_names.items():
            if not isinstance(var, Var):
                continue
            value = getattr(self.env, name, None)
            if value is not None:
                if isinstance(value, bool):
                    self._state[X(var)] = torch.tensor([1.0 if value else 0.0])
                elif isinstance(value, (int, float)):
                    self._state[X(var)] = torch.tensor([float(value)])
                elif isinstance(value, torch.Tensor):
                    self._state[X(var)] = value.clone()

    def _prepare_action(self, action):
        """Convert action to tensor, one-hot encode for Discrete spaces."""
        action_tensor = _ensure_1d(torch.as_tensor(action, dtype=torch.float32))
        if self._backing_env and isinstance(
                getattr(self._backing_env, "action_space", None), gym.spaces.Discrete
        ):
            if action_tensor.numel() == 1:
                idx = int(action_tensor.item())
                one_hot = torch.zeros(self._backing_env.action_space.n)
                one_hot[idx] = 1.0
                action_tensor = one_hot
        return action_tensor

    def _run_block(self, atom, get_block):
        """Interpret an atom's init/update block, writing results to state."""
        for term in get_block(atom):
            read = [self._state[w] for w in term.read]
            out_sort = term.write[0].dtype if len(term.write) else None
            results = eval_itype(term.itype, read, out_sort)
            for w, val in zip(term.write, results):
                self._state[w] = val

    def _latch(self):
        for ctrl in self.ctrl:
            if X(ctrl) in self._state:
                self._state[ctrl] = self._state[X(ctrl)].clone()

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
                self._state[X(p["observation"])] = _to_wire_shape(
                    torch.as_tensor(reset_obs, dtype=torch.float32),
                    X(p["observation"]),
                )
                if p["reward"]:
                    self._state[X(p["reward"])] = torch.tensor([0.0])
                if p["terminated"]:
                    self._state[X(p["terminated"])] = torch.tensor([0.0])
                if p["truncated"]:
                    self._state[X(p["truncated"])] = torch.tensor([0.0])
                self._sync_private_state_from_env()
            else:
                self._run_block(atom, lambda a: a.init)

        self._latch()
        self._initialized = True
        return read_wire(self._state, p["observation"]).numpy(), {}

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        p = self._pairs
        env_atom_idx = self._env_atom_idx

        # Write action to symbolic state, unless an upstream atom drives it
        # (closed-loop with a composed controller — its latched output is used).
        if not self._action_driven:
            self._state[p["action"]] = self._prepare_action(action)

        # Execute update block for each atom
        for atom_idx, atom in enumerate(self.atoms):
            if env_atom_idx is not None and atom_idx == env_atom_idx:
                # Env atom: run real env. A Box-actioned env expects its
                # space's shape, not the [1, n] wire shape.
                action_value = self._state[p["action"]].detach()
                space = getattr(self._backing_env, "action_space", None)
                if isinstance(space, gym.spaces.Box):
                    action_value = action_value.numpy().reshape(space.shape)
                gym_result = self._backing_env.step(
                    action_value
                )
                obs, reward, terminated, truncated, *_ = gym_result
                self._state[X(p["observation"])] = _to_wire_shape(
                    torch.as_tensor(obs, dtype=torch.float32),
                    X(p["observation"]),
                )
                if p["reward"]:
                    self._state[X(p["reward"])] = torch.tensor([float(reward)])
                if p["terminated"]:
                    self._state[X(p["terminated"])] = torch.tensor(
                        [1.0 if terminated else 0.0]
                    )
                if p["truncated"]:
                    self._state[X(p["truncated"])] = torch.tensor(
                        [1.0 if truncated else 0.0]
                    )
                self._sync_private_state_from_env()
            else:
                self._run_block(atom, lambda a: a.update)

        self._latch()
        obs = read_wire(self._state, p["observation"])
        reward = read_wire(self._state, p["reward"]).item() if p["reward"] else 0.0
        terminated = (
            bool(read_wire(self._state, p["terminated"]).item())
            if p["terminated"]
            else False
        )
        truncated = (
            bool(read_wire(self._state, p["truncated"]).item())
            if p["truncated"]
            else False
        )
        return obs.numpy(), reward, terminated, truncated, {}
