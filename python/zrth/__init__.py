from .zrth import (
    Arith,
    Wire,
    DType,
    IType,
    Term,
    Module,
)


#####################################################################
# IType and DType
#####################################################################


# Add type aliases to the DType object
def Bool(*shape):
    return DType.Bool([*shape] if shape else [1])


def Int(*shape):
    return DType.Int([*shape] if shape else [1])


def Real(*shape):
    return DType.Real([*shape] if shape else [1])


def Float(*args):
    return DType.Float([*args] if args else [1])


from .gym import Wrapper, Env
from .smv import parse_smv
from .smt import z3

# Submodule access: from zrth.gym import Env, Wrapper
#                   from zrth.torch import Module
from . import gym as gym
from . import torch as torch


__all__ = [
    "Wire",
    "DType",
    "IType",
    "Term",
    "Module",
    "Wrapper",
    "Env",
]
