"""Shared helpers for the verith limit-probe fixtures."""
import torch
from zrth import Module, Wire, Term, Var, X, LIA, LRA, BV, Int, Real, Bool, BitVec


def i(v, r=1, c=1):
    return torch.full((r, c), v, dtype=torch.int64)


def f(v, r=1, c=1):
    return torch.full((r, c), float(v), dtype=torch.float32)
