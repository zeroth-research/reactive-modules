from .zrth import *
from .zrth import Sort as _Sort
from abc import ABC


class Sort(ABC):
    """Sort type. Variants are available as module-level names."""


Sort.register(_Sort)

# makes sorts available on the base namespace
for _name in dir(_Sort):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_Sort, _name)

from .builder import (
    builder_for,
    LRATermBuilder,
    LIATermBuilder,
    BVTermBuilder,
    TheoryError,
    NonLinearError,
)
from .gym import Env
from .smv import parse_smv
from .smt import z3

# Submodule access: from zrth.sugar import Module ; from zrth.gym import Env
from . import sugar as sugar
from . import gym as gym
from . import torch as torch
