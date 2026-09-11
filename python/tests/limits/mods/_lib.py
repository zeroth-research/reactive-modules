"""Shared helpers for the verith limit-probe fixtures."""
import torch


def i(v, r=1, c=1):
    """An `r x c` integer tensor filled with `v`."""
    return torch.full((r, c), v, dtype=torch.int64)


def f(v, r=1, c=1):
    """An `r x c` float tensor filled with `v`."""
    return torch.full((r, c), float(v), dtype=torch.float32)
