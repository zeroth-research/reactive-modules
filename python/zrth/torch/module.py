import inspect
import torch.nn as nn

from ..zrth import Module as _BaseModule, X
from ..sort import Real
from ..builder import builder_for
from ..analyzer import convert_method, resolve_wire


def _numeric_sort(theory, n):
    """The theory's numeric vector sort of width `n` (LRA/None -> Real, LIA -> Int,
    BV -> BitVec32), derived the same way gym and dsl derive their interface sorts."""
    return builder_for(theory)._numeric_wire([1, n]).dtype


def _is_float_sort(sort) -> bool:
    match sort:
        case Real(_):
            return True
    return False


def _validate_theory_supports_nn(theory):
    """A neural module needs the matrix ops a `Linear`+`ReLU` net compiles to. LRA and
    LIA provide them; BV does not (no Transpose/Linear), so reject it with a clear error
    rather than a deep AttributeError from the builder."""
    ns = builder_for(theory)._ns
    missing = [op for op in ("Transpose", "Linear", "ReLU") if not hasattr(ns, op)]
    if missing:
        tname = getattr(theory, "__name__", "LRA")
        raise NotImplementedError(
            f"theory {tname} does not support neural modules (missing ops: "
            f"{', '.join(missing)}); use LRA or LIA"
        )


def _validate_weight_dtypes(live_layers, theory):
    """Weights are used as-is (no coercion), so their dtype must match the theory:
    floating-point for Real (LRA), integer for Int/BitVec (LIA/BV). Raise otherwise."""
    want_float = _is_float_sort(_numeric_sort(theory, 1))
    kind = "floating-point" if want_float else "integer"
    tname = getattr(theory, "__name__", "LRA")
    for name, layer in live_layers.items():
        for attr in ("weight", "bias"):
            t = getattr(layer, attr, None)
            if t is None:
                continue
            if t.is_floating_point() != want_float:
                raise TypeError(
                    f"theory {tname} expects {kind} weights, but layer "
                    f"'{name or '<root>'}'.{attr} has dtype {t.dtype}; "
                    f"quantise/cast the network to match the theory"
                )


def _extract_nn_module(nn_instance, theory=None, combinatorial=False, **kwargs):
    """Analyze an nn.Module instance and extract a symbolic Module.

    Uses live tensor references for weight/bias so that training updates
    flow through to the symbolic module automatically.

    ``combinatorial`` selects the kind of module built from ``forward``:
      * ``False`` (default) — a **sequential** module over the latched input,
        e.g. V(s): ``update`` reads the *latched* input and writes the next
        output; the initial output is havoced.
      * ``True`` — a **combinatorial** module: ``forward`` reads the awaited
        *next* input, so it computes the output in the same step, e.g. V(s').
    """
    nn_cls = type(nn_instance)

    user_extl = kwargs.pop("extl", None)
    user_ctrl = kwargs.pop("ctrl", None)
    if kwargs:
        raise TypeError(f"unknown arguments: {sorted(kwargs)}")

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
    obs_size = layer_list[0][0]
    qval_size = layer_list[-1][1]

    _validate_theory_supports_nn(theory)
    _validate_weight_dtypes(live_layers, theory)

    extl = resolve_wire("extl", _numeric_sort(theory, obs_size), user_extl)
    ctrl = resolve_wire("ctrl", _numeric_sort(theory, qval_size), user_ctrl)

    layer_out_features = {name: out for name, (_, out) in layers.items()}

    def _forward_terms(read_next: bool):
        # `forward` reads its input as the variable's *latched* slot (index 0). Put the
        # next wire there (read_next=True) to read s'; put the latched wire there
        # (read_next=False) to read s. The output is always the next intf wire.
        in_wire, other = (X(extl), extl) if read_next else (extl, X(extl))
        return convert_method(
            nn_cls.forward, {obs_param: [in_wire, other]}, [X(ctrl)], cls=nn_cls,
            layers=layer_out_features, live_layers=live_layers, theory=theory,
        )

    if combinatorial:
        # V(s'): memoryless, reads the awaited next input.
        assign = _forward_terms(True)
        return dict(init=assign, update=assign, vars=[ctrl, extl])
    else:
        # V(s): init havocs; update reads the latched input.
        return dict(update=_forward_terms(False), vars=[ctrl, extl])


class Module(_BaseModule, nn.Module):
    """An nn.Module backed by a symbolic Module with live tensor references.

    Inherits both Module (symbolic reactive module) and nn.Module (trainable).

    Usage:
        from zrth.torch import Module

        wrapped = Module(nn_module_instance)                    # sequential, V(s)
        wrapped = Module(nn_module_instance, theory=LIA)         # integer (e.g. quantised) net
        wrapped = Module(nn_module_instance, combinatorial=True) # combinatorial, V(s')
        wrapped.parameters()   # returns original nn.Module parameters
        wrapped(x)             # runs forward pass via original nn.Module
        wrapped.atoms          # symbolic module structure

    ``theory`` selects the term theory / interface sorts (LRA -> Real, LIA -> Int;
    BV is unsupported for neural modules). ``combinatorial`` selects whether
    ``forward`` becomes a sequential atom over the latched input (``False``,
    default) or a combinatorial atom over the next input (``True``).

    Training the original nn.Module automatically updates the symbolic module
    because the Linear op holds a reference to the live weight tensors.
    """

    def __new__(cls, nn_module, theory=None, combinatorial=False, **kwargs):
        if not isinstance(nn_module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(nn_module)}")
        parts = _extract_nn_module(
            nn_module, theory=theory, combinatorial=combinatorial, **kwargs
        )
        return _BaseModule.__new__(cls, **parts)

    def __init__(self, nn_module, theory=None, combinatorial=False, **kwargs):
        nn.Module.__init__(self)
        self.inner = nn_module

    def forward(self, x):
        return self.inner(x)
