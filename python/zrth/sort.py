from .zrth import Sort as _Sort
from abc import ABC as _ABC

import torch


class Sort(_ABC):
    """Sort type. Variants are available as module-level names."""


Sort.register(_Sort)

# makes sorts available on the base namespace
for _name in dir(_Sort):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_Sort, _name)


# Exhaustive on purpose: a new sort must be added here, never given a default.
_KINDS = (
    (_Sort.Bool, torch.bool, "boolean"),
    (_Sort.Int, torch.int64, "integer"),
    (_Sort.Real, torch.float32, "real"),
    (_Sort.BitVec, torch.int64, "bit-vector"),
)


def tensor_for(value, sort):
    """``value`` as a tensor of the kind ``sort`` requires (``sort`` may be a family or an
    instance). A value the conversion would alter is rejected, never silently cast."""
    for family, dtype, name in _KINDS:
        if sort is family or isinstance(sort, family):
            break
    else:
        raise TypeError(f"no tensor representation for sort {sort!r}")

    given = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if dtype is torch.bool and given.dtype is not torch.bool:
        raise TypeError(
            f"only a boolean value can be a {name} constant: "
            f"{given.flatten().tolist()} would collapse to True/False"
        )
    if dtype is torch.int64 and given.is_floating_point():
        raise TypeError(
            f"{name} constants cannot hold a floating-point value: "
            f"{given.flatten().tolist()} would be truncated"
        )
    return given.to(dtype)


# Required, not boilerplate: without it `from .sort import *` in the package __init__
# re-exports `torch`, replacing the `zrth.torch` submodule with the torch library.
__all__ = ["Sort", "tensor_for"] + [n for n in dir(_Sort) if not n.startswith("_")]
