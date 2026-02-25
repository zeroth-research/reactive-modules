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

    # Analyze each method: extract self.* reads, writes, and calls to other methods
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

        summaries[name] = (read_attrs, written_attrs, calls)

    # BFS from roots, following calls
    visited = set()
    queue = list(roots)
    while queue:
        name = queue.pop()
        if name in visited or name not in summaries:
            continue
        visited.add(name)
        _, _, calls = summaries[name]
        queue.extend(calls)

    # Collect self.* reads/writes from all relevant methods
    read_self = set()
    written_self = set()
    for name in visited:
        ra, wa, _ = summaries[name]
        read_self |= ra
        written_self |= wa

    prvt = written_self & read_self
    write_only = written_self - read_self
    params = read_self - written_self

    if write_only:
        raise ValueError(
            f"Attributes written in {roots} but never read back: {sorted(write_only)}. "
            f"These must be made observable."
        )

    return prvt, params

class Env(Module, gym.Env):

    def __new__(cls, *args, **kwargs):
        # TODO: use params
        prvt, params = _classify_attrs(cls, ['reset', 'step'])
        print('@@@@@@@@')
        print(prvt)
        print('@@@@@@@@')
        print(params)
        print('@@@@@@@@')
        annotations = cls.__annotations__
        
        q_values_dtype = annotations.get('q_values')
        observation_dtype = annotations.get('observation')
        if q_values_dtype is None:
            raise ValueError("Add 'q_values: DType.XXX([shape])' to the class body.")
        if observation_dtype is None:
            raise ValueError("Add 'observation: DType.XXX([shape])' to the class body.")
        
        prvt_wires = {}
        for name in prvt:
            if name not in annotations:
                raise ValueError(
                    f"Private attribute '{name}' needs a type annotation. "
                    f"Add '{name}: DType.XXX([shape])' to the class body."
                )
            dtype = annotations[name]
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

