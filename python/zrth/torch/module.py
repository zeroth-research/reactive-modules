import inspect
import torch.nn as nn

from ..zrth import Module as _BaseModule, DType
from ..analyzer import convert_method, resolve_wire


def _extract_nn_module(nn_instance, **kwargs):
    """Analyze an nn.Module instance and extract a symbolic Module.

    Uses live tensor references for weight/bias so that training updates
    flow through to the symbolic module automatically.
    """
    nn_cls = type(nn_instance)

    user_extl = kwargs.pop("extl", None)
    user_intf = kwargs.pop("intf", None)

    obs_param = next(
        p for p in inspect.signature(nn_cls.forward).parameters if p != "self"
    )

    # Infer layer structure from the actual nn.Module instance
    layers = {}
    live_layers = {}
    for name, child in nn_instance.named_modules():
        if isinstance(child, nn.Linear):
            layers[name] = (child.in_features, child.out_features)
            live_layers[name] = child

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
    forward = convert_method(
        nn_cls.forward, wires, result, cls=nn_cls,
        layers=layer_out_features, live_layers=live_layers,
    )

    obs = [extl, intf]
    return dict(assign=forward, obs=obs)


class Module(_BaseModule, nn.Module):
    """An nn.Module backed by a symbolic Module with live tensor references.

    Inherits both Module (symbolic reactive module) and nn.Module (trainable).

    Usage:
        from zrth.torch import Module

        wrapped = Module(nn_module_instance)
        wrapped.parameters()   # returns original nn.Module parameters
        wrapped(x)             # runs forward pass via original nn.Module
        wrapped.atoms          # symbolic module structure

    Training the original nn.Module automatically updates the symbolic module
    because IType.Tensor holds a reference to the live weight tensors.
    """

    def __new__(cls, nn_module, **kwargs):
        if not isinstance(nn_module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(nn_module)}")
        parts = _extract_nn_module(nn_module, **kwargs)
        return _BaseModule.__new__(cls, **parts)

    def __init__(self, nn_module, **kwargs):
        nn.Module.__init__(self)
        self.inner = nn_module

    def forward(self, x):
        return self.inner(x)
