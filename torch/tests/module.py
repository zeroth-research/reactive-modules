import sys
from os.path import dirname, join as pathjoin
sys.path.append(pathjoin(dirname(__file__), "../python"))

from zrmtorch.module import Module
import torch as tch


def init():
    Guard[True] >> [
        next(x) == tch.Tensor([0, 0, 0]),
        next(y) == next(y0),
        next(z) == next(z0)
    ]

def update():
    Guard[(x < y | ((x < z) & (x < y)))] >> [next(x) == (x + 1)],
    Guard[~(x < y | (x < z))]            >> [next(x) == tch.Tensor([0, 0, 0])]

module = Module(["x", "y", "z", "y0", "z0"], init, update)
module.to_html("/tmp/mod.html", open=True)

