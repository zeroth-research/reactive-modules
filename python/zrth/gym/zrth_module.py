from zrth import Module, Wire, DType
from zrth.analyzer import AbstractInterpreter, UnsupportedFeatureError, join_states, AbstractValue
from .converter import convert_method
import gymnasium as gym
import torch.nn as nn
import inspect
import ast

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

def _classify_attrs(cls, roots, init_attr_vals=None):
    """Classify self.* attributes used in the given root methods (and their callees).

    Returns:
        prvt: set of names that are both read and written (private state)
        params: set of names that are only read (set in __init__, treated as constants)

    Raises:
        ValueError: if any attribute is written but never read back
    """

    # Collect all methods from the class
    methods = {}
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        methods[name] = method

    # Analyze each method: extract self.* reads, writes, calls, and attr values
    summaries = {}
    for name, method in methods.items():
        try:
            interp = AbstractInterpreter(method)
            states = interp.analyze()
        except (UnsupportedFeatureError, NotImplementedError):
            continue
        merged = join_states(states)

        read_attrs = set()
        written_attrs = set()
        for r in merged.reads:
            if r.name.startswith("self."):
                read_attrs.add(r.name[5:])
        for w in merged.writes:
            if w.name.startswith("self."):
                written_attrs.add(w.name[5:])

        # self.xxx reads where xxx is a class method = calls (not data reads)
        calls = read_attrs & set(methods.keys())
        read_attrs -= calls

        # AbstractValue for each self.* attr (from merged.attrs)
        attr_values = merged.attrs.get("self", {})

        summaries[name] = (read_attrs, written_attrs, calls, attr_values)

    # BFS from roots, following calls
    visited = set()
    queue = list(roots)
    while queue:
        name = queue.pop()
        if name in visited or name not in summaries:
            continue
        visited.add(name)
        _, _, calls, _ = summaries[name]
        queue.extend(calls)

    # Collect self.* reads/writes/values from all relevant methods
    read_self = set()
    written_self = set()
    attr_vals = {}
    for name in visited:
        ra, wa, _, av = summaries[name]
        read_self |= ra
        written_self |= wa
        for attr, val in av.items():
            existing = attr_vals.get(attr)
            if existing is None or (val.is_const() and not existing.is_const()):
                attr_vals[attr] = val

    prvt = written_self & read_self
    write_only = written_self - read_self
    params = read_self - written_self

    # Fall back to __init__ values for prvt/param attrs that are Top/missing
    if init_attr_vals:
        for attr in prvt | params:
            val = attr_vals.get(attr)
            init_val = init_attr_vals.get(attr)
            if init_val is not None and (val is None or not val.is_const()):
                attr_vals[attr] = init_val

    if write_only:
        raise ValueError(
            f"Attributes written in {roots} but never read back: {sorted(write_only)}. "
            f"These must be made observable."
        )

    return prvt, params, attr_vals

def _infer_shape_and_elem_type(value):
    """Recursively derive tensor shape and element type from a Python value."""
    if isinstance(value, bool):  # before int — bool is a subclass of int
        return [], bool
    if isinstance(value, (int, float)):
        return [], type(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Cannot infer shape from empty collection")
        inner_shape, elem_type = _infer_shape_and_elem_type(value[0])
        return [len(value)] + inner_shape, elem_type
    raise ValueError(f"Unsupported element type: {type(value).__name__}")

def _infer_dtype(name, abstract_value):
    """Infer a DType from an AbstractValue."""
    if abstract_value is None:
        raise ValueError(
            f"Cannot infer DType for private attribute '{name}': "
            f"analyzer returned {abstract_value}"
        )

    # CONST: derive shape and element type from the actual value
    if abstract_value.is_const():
        shape, elem_type = _infer_shape_and_elem_type(abstract_value.value)
        if not shape:
            shape = [1]  # standalone scalar → 1-element tensor
        dtype_fn = _PYTHON_TYPE_TO_DTYPE.get(elem_type)
        if dtype_fn is None:
            raise ValueError(
                f"Cannot infer DType for private attribute '{name}': "
                f"unsupported element type '{elem_type.__name__}'"
            )
        return dtype_fn(shape)

    # TYPED: only scalar primitives (no shape info without a value)
    if abstract_value.type_ is None:
        raise ValueError(
            f"Cannot infer DType for private attribute '{name}': "
            f"analyzer returned {abstract_value}"
        )
    dtype_fn = _PYTHON_TYPE_TO_DTYPE.get(abstract_value.type_)
    if dtype_fn is None:
        raise ValueError(
            f"Cannot infer DType for private attribute '{name}': "
            f"unsupported Python type '{abstract_value.type_.__name__}'"
        )
    return dtype_fn([1])

def _parse_call_repr(call_repr):
    """Parse an AbstractValue call_repr string into (func_name, args, kwargs).

    E.g. "spaces.Discrete(2)" -> ("Discrete", [2], {})
         "spaces.Box(low=0, high=10, shape=(1,))" -> ("Box", [], {"low": 0, "high": 10, "shape": (1,)})
    """
    node = ast.parse(call_repr, mode='eval').body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Expected a call expression, got: {call_repr}")

    # Extract the short name (last segment of dotted name)
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

def _gym_space_to_dtype(space_name, pos_args, kw_args, is_q_values):
    """Convert a parsed gym space into a DType.

    For q_values (is_q_values=True): always Float, shape = number of actions.
    For observation (is_q_values=False): type from _GYM_SPACE_TO_DTYPE, shape from space.
    """
    dtype_fn = _GYM_SPACE_TO_DTYPE.get(space_name)
    if dtype_fn is None:
        raise ValueError(f"Unsupported gym space type: {space_name}")

    if space_name == "Discrete":
        n = pos_args[0]
        if is_q_values:
            return DType.Float([n])
        else:
            return dtype_fn([1])
    elif space_name == "Box":
        shape = kw_args.get("shape")
        if shape is None and len(pos_args) > 2:
            shape = pos_args[2]
        if shape is None:
            raise ValueError("Box space requires a 'shape' argument")
        return dtype_fn(list(shape))
    elif space_name == "MultiBinary":
        n = pos_args[0]
        return dtype_fn([n])
    else:
        raise ValueError(f"Unsupported gym space type: {space_name}")

def _infer_spaces_from_init(cls, args, kwargs):
    """Analyze __init__ to infer q_values and observation DTypes from gym spaces."""
    params = list(inspect.signature(cls.__init__).parameters.keys())
    params.remove("self")

    arg_values = {}
    for i, val in enumerate(args):
        arg_values[params[i]] = AbstractValue.const(val)
    for k, v in kwargs.items():
        arg_values[k] = AbstractValue.const(v)

    interp = AbstractInterpreter(cls.__init__)
    results = interp.analyze(arg_values=arg_values)
    merged = join_states(results)

    self_attrs = merged.attrs.get("self", {})
    action_space_val = self_attrs.get("action_space")
    observation_space_val = self_attrs.get("observation_space")

    if action_space_val is None or action_space_val.call_repr is None:
        raise ValueError("Cannot infer action_space from __init__: not found or not a call")
    if observation_space_val is None or observation_space_val.call_repr is None:
        raise ValueError("Cannot infer observation_space from __init__: not found or not a call")

    a_name, a_args, a_kwargs = _parse_call_repr(action_space_val.call_repr)
    o_name, o_args, o_kwargs = _parse_call_repr(observation_space_val.call_repr)

    q_values_dtype = _gym_space_to_dtype(a_name, a_args, a_kwargs, is_q_values=True)
    observation_dtype = _gym_space_to_dtype(o_name, o_args, o_kwargs, is_q_values=False)

    return q_values_dtype, observation_dtype, self_attrs

def _infer_layers_from_init(cls, args, kwargs):
    """Analyze __init__ to extract nn.Linear layers and their sizes.

    Returns:
        Ordered dict {name: (in_features, out_features)} for each nn.Linear layer.
    """
    params = list(inspect.signature(cls.__init__).parameters.keys())
    params.remove("self")

    arg_values = {}
    for i, val in enumerate(args):
        arg_values[params[i]] = AbstractValue.const(val)
    for k, v in kwargs.items():
        arg_values[k] = AbstractValue.const(v)

    interp = AbstractInterpreter(cls.__init__)
    results = interp.analyze(arg_values=arg_values)
    merged = join_states(results)

    self_attrs = merged.attrs.get("self", {})
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


class Env(Module, gym.Env):

    def __new__(cls, *args, **kwargs):
        q_values_dtype, observation_dtype, init_attrs = _infer_spaces_from_init(cls, args, kwargs)

        prvt, params, attr_vals = _classify_attrs(cls, ['reset', 'step'], init_attr_vals=init_attrs)

        prvt_wires = {}
        for name in prvt:
            dtype = _infer_dtype(name, attr_vals.get(name))
            prvt_wires[name] = [Wire(dtype), Wire(dtype)]

        param_wires = {}
        for name in params:
            dtype = _infer_dtype(name, attr_vals.get(name))
            param_wires[name] = Wire(dtype)

        q_values = [Wire(q_values_dtype), Wire(q_values_dtype)]
        observation = [Wire(observation_dtype), Wire(observation_dtype)]
        reward = [Wire(DType.Float([1])), Wire(DType.Float([1]))]
        terminated = [Wire(DType.Bool([1])), Wire(DType.Bool([1]))]
        truncated = [Wire(DType.Bool([1])), Wire(DType.Bool([1]))]

        wires = {
            'q_values': q_values,
            **prvt_wires,
        }
        result = [observation[1], reward[1], terminated[1], truncated[1]]

        reset = convert_method(cls.reset, wires, result, cls=cls, params=param_wires)
        step = convert_method(cls.step, wires, result, cls=cls, params=param_wires)

        obs = [q_values, observation, reward, terminated, truncated]
        return super().__new__(cls, init=reset, update=step, obs=obs, prvt=list(prvt_wires.values()))

class NN(Module, nn.Module):
    
    def __new__(cls, *args, **kwargs):
        layers = _infer_layers_from_init(cls, args, kwargs)
        layer_list = list(layers.values())
        obs_size = layer_list[0][0]   # first layer's in_features
        qval_size = layer_list[-1][1] # last layer's out_features

        observation = [Wire(DType.Float([obs_size])), Wire(DType.Float([obs_size]))]
        q_values = [Wire(DType.Float([qval_size])), Wire(DType.Float([qval_size]))]

        # Use forward's parameter name as wire key so the converter can resolve it.
        # Combinatorial atoms read next wires (index 1), so pass [next, latched]
        # — the converter reads index 0 which must be the next wire.
        forward_params = [p for p in inspect.signature(cls.forward).parameters if p != "self"]
        obs_param = forward_params[0]
        wires = {obs_param: [observation[1], observation[0]]}
        result = [q_values[1]]
        layer_out_features = {name: out for name, (_, out) in layers.items()}
        forward = convert_method(cls.forward, wires, result, cls=cls, layers=layer_out_features)

        obs = [observation, q_values]
        return super().__new__(cls, assign=forward, obs=obs)

