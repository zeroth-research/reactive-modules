from .zrth import Wire, DType, IType, Term, Module


#####################################################################
# IType and DType
#####################################################################


# Add type aliases to the DType object
def Bool(*shape):
    return DType.Bool([*shape])


def Int(*shape):
    return DType.Int([*shape])


def Real(*shape):
    return DType.Real([*shape])


def Float(*args):
    return DType.Float([*args])


from .gym import Wrapper, Env
from .smv import parse_smv

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
