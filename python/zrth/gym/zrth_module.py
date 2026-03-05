from zrth import Module, Wire, DType
from zrth.analyzer import AbstractInterpreter, UnsupportedFeatureError, join_states, AbstractValue
from zrth.analyzer import convert_method
import gymnasium as gym
import torch.nn as nn
import inspect
import ast

# ============================================================================
# Type mappings
# ============================================================================

_PYTHON_TYPE_TO_DTYPE = {
    bool: DType.Bool,
    float: DType.Float,
    int: DType.Int,
}

_GYM_SPACE_TO_DTYPE = {
    "Discrete": DType.Int,
    "Box": DType.Float,
    "MultiBinary": DType.Bool,
}

# ============================================================================
# Small utilities
# ============================================================================

def _to_python(v):
    """Convert numpy scalars to plain Python types for the abstract interpreter."""
    if hasattr(v, 'item'):
        return v.item()
    return v

def _wire_pair(dtype):
    """Create a [latched, next] wire pair for the given dtype."""
    return [Wire(dtype), Wire(dtype)]

def _unwrap(method):
    """Follow __wrapped__ chain to get the original function."""
    while hasattr(method, '__wrapped__'):
        method = method.__wrapped__
    return method

# ============================================================================
# Wire resolution
# ============================================================================

def _resolve_wire(name, dtype, user_val=None):
    """Return a validated [latched, next] wire pair for an observable signal.

    If user_val is None, creates a fresh pair from dtype.
    If user_val is a wire pair, validates its dtype and returns it.
    """
    if user_val is None:
        return _wire_pair(dtype)
    is_pair = isinstance(user_val, (list, tuple)) and len(user_val) == 2 and all(isinstance(w, Wire) for w in user_val)
    if is_pair:
        for w in user_val:
            if w.dtype() != dtype:
                raise ValueError(
                    f"DType mismatch for '{name}': expected {dtype}, got {w.dtype()}"
                )
        return list(user_val)
    if isinstance(user_val, (list, tuple)) and len(user_val) > 0 and all(
        isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(w, Wire) for w in item)
        for item in user_val
    ):
        raise NotImplementedError("Tuple of wire pairs not yet supported")
    raise ValueError(
        f"Invalid wire format for '{name}': expected [Wire, Wire] or tuple of wire pairs"
    )

# ============================================================================
# __init__ wrapping
# ============================================================================

def _wrap_init(cls, kwargs_to_strip):
    """Wrap cls.__init__ to silently drop kwargs consumed by __new__.

    __new__ pops wire-override kwargs before the user's __init__ sees them.
    Without this wrapper those kwargs would reach the user's __init__ as
    unexpected keyword arguments. Only wraps if cls defines __init__ directly;
    an inherited __init__ was already wrapped when the ancestor was defined.
    """
    if '__init__' not in cls.__dict__:
        return
    original_init = cls.__dict__['__init__']
    def wrapped_init(self, *args, **kw):
        for k in kwargs_to_strip:
            kw.pop(k, None)
        return original_init(self, *args, **kw)
    wrapped_init.__wrapped__ = original_init
    cls.__init__ = wrapped_init

# ============================================================================
# Abstract interpreter helpers
# ============================================================================

def _analyze_init(cls, args, kwargs):
    """Run the abstract interpreter on cls.__init__ and return self.* attribute values.

    Returns:
        dict[str, AbstractValue] — the self.* attrs inferred from __init__
    """
    init = _unwrap(cls.__init__)
    param_names = [p for p in inspect.signature(init).parameters if p != "self"]
    arg_values = {param_names[i]: AbstractValue.const(_to_python(v)) for i, v in enumerate(args)}
    arg_values.update({k: AbstractValue.const(_to_python(v)) for k, v in kwargs.items()})
    return join_states(AbstractInterpreter(init).analyze(arg_values=arg_values)).attrs.get("self", {})


def _infer_spaces(self_attrs):
    """Extract action and observation DTypes from analyzed __init__ self attrs."""
    action_space_val = self_attrs.get("action_space")
    observation_space_val = self_attrs.get("observation_space")

    if action_space_val is None or action_space_val.call_repr is None:
        raise ValueError("Cannot infer action_space from __init__: not found or not a call")
    if observation_space_val is None or observation_space_val.call_repr is None:
        raise ValueError("Cannot infer observation_space from __init__: not found or not a call")

    a_name, a_args, a_kwargs = _parse_call_repr(action_space_val.call_repr)
    o_name, o_args, o_kwargs = _parse_call_repr(observation_space_val.call_repr)

    return (
        _gym_space_to_dtype(a_name, a_args, a_kwargs, is_action=True),
        _gym_space_to_dtype(o_name, o_args, o_kwargs, is_action=False),
    )


def _infer_layers(self_attrs):
    """Extract nn.Linear layer sizes from analyzed __init__ self attrs.

    Returns:
        dict {attr_name: (in_features, out_features)} for each nn.Linear attr.
    """
    layers = {}
    for attr_name, attr_val in self_attrs.items():
        if attr_val.call_repr is None:
            continue
        try:
            name, pos_args, _ = _parse_call_repr(attr_val.call_repr)
        except (ValueError, SyntaxError):
            continue
        if name == "Linear" and len(pos_args) >= 2:
            layers[attr_name] = (pos_args[0], pos_args[1])

    if not layers:
        raise ValueError("No nn.Linear layers found in __init__")

    return layers


def _parse_call_repr(call_repr):
    """Parse an AbstractValue call_repr string into (func_name, pos_args, kw_args).

    E.g. "spaces.Discrete(2)" -> ("Discrete", [2], {})
         "spaces.Box(low=0, high=10, shape=(1,))" -> ("Box", [], {"low": 0, "high": 10, "shape": (1,)})
    """
    node = ast.parse(call_repr, mode='eval').body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Expected a call expression, got: {call_repr}")
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        raise ValueError(f"Cannot extract function name from: {call_repr}")
    pos_args = [ast.literal_eval(a) for a in node.args]
    kw_args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return name, pos_args, kw_args


def _gym_space_to_dtype(space_name, pos_args, kw_args, is_action):
    """Convert a parsed gym space into a DType.

    For action (is_action=True): always Float, shape = number of actions.
    For observation (is_action=False): element type from _GYM_SPACE_TO_DTYPE.
    """
    dtype_fn = _GYM_SPACE_TO_DTYPE.get(space_name)
    if dtype_fn is None:
        raise ValueError(f"Unsupported gym space type: {space_name}")

    if space_name == "Discrete":
        n = pos_args[0]
        return DType.Float([n]) if is_action else dtype_fn([1])
    elif space_name == "Box":
        shape = kw_args.get("shape") or (pos_args[2] if len(pos_args) > 2 else None)
        if shape is None:
            raise ValueError("Box space requires a 'shape' argument")
        return dtype_fn(list(shape))
    else:  # MultiBinary
        return dtype_fn([pos_args[0]])

# ============================================================================
# Attribute classification
# ============================================================================

def _classify_attrs(cls, roots, init_attrs=None, base_cls=None):
    """Classify self.* attributes used in root methods (and their callees).

    Walks cls.__mro__ up to (but not including) base_cls, so only user-defined
    methods are analyzed — not framework methods from Env, Module, gym.Env, etc.

    Args:
        cls:        The class to analyze.
        roots:      Method names to start from (e.g. ['reset', 'step']).
        init_attrs: Optional dict[str, AbstractValue] from _analyze_init, used
                    as a fallback for attrs whose values are unknown in roots.
        base_cls:   Stop walking the MRO at this class (exclusive). Pass Env or
                    NN to exclude framework-level methods from analysis.

    Returns:
        prvt:      set — attributes both read and written (private mutable state)
        params:    set — attributes only read (constants set in __init__)
        attr_vals: dict[str, AbstractValue] — best-known value for each attr

    Raises:
        ValueError: if any attribute is written but never read back.
    """
    # Collect user-defined methods by walking the MRO up to base_cls.
    # Iterate most-derived-first; setdefault keeps the most-derived definition.
    methods = {}
    for klass in cls.__mro__:
        if klass is base_cls or klass is object:
            break
        for name, val in klass.__dict__.items():
            if callable(val) and not isinstance(val, (staticmethod, classmethod)):
                methods.setdefault(name, val)

    # Analyze each method individually
    summaries = {}
    for name, method in methods.items():
        try:
            merged = join_states(AbstractInterpreter(method).analyze())
        except (UnsupportedFeatureError, NotImplementedError):
            continue
        read_attrs    = {r.name[5:] for r in merged.reads  if r.name.startswith("self.")}
        written_attrs = {w.name[5:] for w in merged.writes if w.name.startswith("self.")}
        # self.foo reads where foo is a known method → calls, not data reads
        calls = read_attrs & set(methods.keys())
        read_attrs -= calls
        summaries[name] = (read_attrs, written_attrs, calls, merged.attrs.get("self", {}))

    # BFS from roots, following intra-class calls
    visited, queue = set(), list(roots)
    while queue:
        name = queue.pop()
        if name in visited or name not in summaries:
            continue
        visited.add(name)
        queue.extend(summaries[name][2])  # calls

    read_self, written_self, attr_vals = set(), set(), {}
    for name in visited:
        ra, wa, _, av = summaries[name]
        read_self    |= ra
        written_self |= wa
        for attr, val in av.items():
            existing = attr_vals.get(attr)
            if existing is None or (val.is_const() and not existing.is_const()):
                attr_vals[attr] = val

    prvt      = written_self & read_self
    params    = read_self - written_self
    write_only = written_self - read_self

    # Use init_attrs as a fallback for attrs with missing or non-const values
    if init_attrs:
        for attr in prvt | params:
            val = attr_vals.get(attr)
            init_val = init_attrs.get(attr)
            if init_val is not None and (val is None or not val.is_const()):
                attr_vals[attr] = init_val

    if write_only:
        raise ValueError(
            f"Attributes written in {roots} but never read back: {sorted(write_only)}. "
            f"These must be made observable."
        )

    return prvt, params, attr_vals

# ============================================================================
# DType inference
# ============================================================================

def _infer_shape_and_elem_type(value):
    """Recursively derive tensor shape and element type from a Python value."""
    if isinstance(value, bool):      # before int — bool subclasses int
        return [], bool
    if isinstance(value, (int, float)):
        return [], type(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Cannot infer shape from empty collection")
        inner_shape, elem_type = _infer_shape_and_elem_type(value[0])
        return [len(value)] + inner_shape, elem_type
    raise ValueError(f"Unsupported element type: {type(value).__name__}")


def _infer_dtype(name, abstract_value):
    """Infer a DType from an AbstractValue."""
    if abstract_value is None:
        raise ValueError(f"Cannot infer DType for '{name}': analyzer returned None")

    if abstract_value.is_const():
        shape, elem_type = _infer_shape_and_elem_type(abstract_value.value)
        dtype_fn = _PYTHON_TYPE_TO_DTYPE.get(elem_type)
        if dtype_fn is None:
            raise ValueError(
                f"Cannot infer DType for '{name}': unsupported element type '{elem_type.__name__}'"
            )
        return dtype_fn(shape or [1])

    if abstract_value.type_ is None:
        raise ValueError(f"Cannot infer DType for '{name}': analyzer returned {abstract_value}")
    dtype_fn = _PYTHON_TYPE_TO_DTYPE.get(abstract_value.type_)
    if dtype_fn is None:
        raise ValueError(
            f"Cannot infer DType for '{name}': unsupported Python type '{abstract_value.type_.__name__}'"
        )
    return dtype_fn([1])

# ============================================================================
# Env
# ============================================================================

class Env(Module, gym.Env):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _wrap_init(cls, ("action", "observation", "reward", "terminated", "truncated"))

    def __new__(cls, *args, **kwargs):
        action_param = next(p for p in inspect.signature(cls.step).parameters if p != "self")

        user_wires = {
            "action":      kwargs.pop("action",      None),
            "observation": kwargs.pop("observation", None),
            "reward":      kwargs.pop("reward",      None),
            "terminated":  kwargs.pop("terminated",  None),
            "truncated":   kwargs.pop("truncated",   None),
        }

        init_attrs = _analyze_init(cls, args, kwargs)
        action_dtype, observation_dtype = _infer_spaces(init_attrs)
        prvt, params, attr_vals = _classify_attrs(
            cls, ['reset', 'step'], init_attrs=init_attrs, base_cls=Env
        )

        prvt_wires  = {name: _wire_pair(_infer_dtype(name, attr_vals.get(name))) for name in prvt}
        param_wires = {name: Wire(_infer_dtype(name, attr_vals.get(name))) for name in params}

        action      = _resolve_wire("action",      action_dtype,      user_wires["action"])
        observation = _resolve_wire("observation", observation_dtype, user_wires["observation"])
        reward      = _resolve_wire("reward",      DType.Float([1]), user_wires["reward"])
        terminated  = _resolve_wire("terminated",  DType.Bool([1]),  user_wires["terminated"])
        truncated   = _resolve_wire("truncated",   DType.Bool([1]),  user_wires["truncated"])

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

class NN(Module, nn.Module):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _wrap_init(cls, ("extl", "intf"))

    def __new__(cls, *args, **kwargs):
        obs_param = next(p for p in inspect.signature(cls.forward).parameters if p != "self")

        user_extl = kwargs.pop("extl", None)
        user_intf = kwargs.pop("intf", None)

        init_attrs = _analyze_init(cls, args, kwargs)
        layers     = _infer_layers(init_attrs)
        layer_list = list(layers.values())
        obs_size   = layer_list[0][0]   # first layer's in_features
        qval_size  = layer_list[-1][1]  # last layer's out_features

        extl = _resolve_wire("extl", DType.Float([obs_size]),  user_extl)
        intf = _resolve_wire("intf", DType.Float([qval_size]), user_intf)

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
