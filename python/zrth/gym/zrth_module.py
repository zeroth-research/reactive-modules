from zrth import Module, Wire, DType
from zrth.analyzer import AbstractInterpreter, UnsupportedFeatureError, join_states
from .converter import convert_method
import gymnasium as gym
import torch.nn as nn

def _classify_attrs(cls, roots):
    """Classify self.* attributes used in the given root methods (and their callees).

    Returns:
        prvt: set of names that are both read and written (private state)
        params: set of names that are only read (set in __init__, treated as constants)

    Raises:
        ValueError: if any attribute is written but never read back
    """
    import inspect

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

    if write_only:
        raise ValueError(
            f"Attributes written in {roots} but never read back: {sorted(write_only)}. "
            f"These must be made observable."
        )

    return prvt, params, attr_vals

_PYTHON_TYPE_TO_DTYPE = {
    bool: DType.Bool([1]),
    float: DType.Float([1]),
    int: DType.Int([1]),
}

def _infer_dtype(name, abstract_value):
    """Infer a DType from an AbstractValue's Python type."""
    if abstract_value is None or abstract_value.type_ is None:
        raise ValueError(
            f"Cannot infer DType for private attribute '{name}': "
            f"analyzer returned {abstract_value}"
        )
    dtype = _PYTHON_TYPE_TO_DTYPE.get(abstract_value.type_)
    if dtype is None:
        raise ValueError(
            f"Cannot infer DType for private attribute '{name}': "
            f"unsupported Python type '{abstract_value.type_.__name__}'"
        )
    return dtype

class Env(Module, gym.Env):

    def __new__(cls, *args, **kwargs):
        # TODO: use params
        prvt, params, attr_vals = _classify_attrs(cls, ['reset', 'step'])
        annotations = cls.__annotations__

        q_values_dtype = annotations.get('q_values')
        observation_dtype = annotations.get('observation')
        if q_values_dtype is None:
            raise ValueError("Add 'q_values: DType.XXX([shape])' to the class body.")
        if observation_dtype is None:
            raise ValueError("Add 'observation: DType.XXX([shape])' to the class body.")

        prvt_wires = {}
        for name in prvt:
            dtype = _infer_dtype(name, attr_vals.get(name))
            prvt_wires[name] = [Wire(dtype), Wire(dtype)]

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

        reset = convert_method(cls.reset, wires, result, cls=cls)
        step = convert_method(cls.step, wires, result, cls=cls)

        obs = [q_values, observation, reward, terminated, truncated]
        return super().__new__(cls, init=reset, update=step, obs=obs, prvt=list(prvt_wires.values()))

class NN(Module, nn.Module):
    
    def __new__(cls, *args, **kwargs):
        # TODO: trace the init for q_values and observation sizes
        observation = [Wire(DType.Float([1])), Wire(DType.Float([1]))]
        q_values = [Wire(DType.Float([2])), Wire(DType.Float([2]))]
        
        wires = {'observation': observation}
        result = [q_values[1]]
        forward = convert_method(cls.forward, wires, result, cls=cls)

        obs = [observation, q_values]
        return super().__new__(cls, assign=forward, obs=obs)

