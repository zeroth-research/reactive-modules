from zrth import Module, Wire, DType
from zrth.analyzer import (
    convert_method, _analyze_init, _infer_spaces, _infer_layers,
    _classify_attrs, _infer_dtype, _wire_pair, _resolve_wire, _wrap_init,
)
import gymnasium as gym
import torch.nn as nn
import inspect

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
