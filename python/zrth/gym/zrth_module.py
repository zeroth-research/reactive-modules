from zrth import Module, Wire, DType
from zrth.gym.analyzer import AccessAnalyzer
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
    summaries = AccessAnalyzer().analyze(cls)

    visited = set()
    queue = list(roots)
    while queue:
        name = queue.pop()
        if name in visited or name not in summaries:
            continue
        visited.add(name)
        queue.extend(summaries[name].calls)
    relevant = {name: summaries[name] for name in visited}

    read_self = {}
    written_self = {}

    for _, sm in relevant.items():
        read_self.update(sm.read_attrs.get('self', {}))
        written_self.update(sm.written_attrs.get('self', {}))

    prvt = set(written_self.keys()) & set(read_self.keys())
    write_only = set(written_self.keys()) - set(read_self.keys())
    params = set(read_self.keys()) - set(written_self.keys())

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

