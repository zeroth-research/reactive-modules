import sys
from os.path import dirname, join as pathjoin
sys.path.append(pathjoin(dirname(__file__), "../python"))

from bindings.term import Var
import bindings

import torch as tch


Ctx = bindings.Context

ctx = Ctx()

def init():
    zrm.gc(True, [next(x) == tch.Tensor([0, 0, 0]), next(y) == next(y0), next(z) == next(z0)])

def update():
   # zrm.gc(x < y | ((x < z) & (x < y)), [next(x) == (x + 1)]),
   # zrm.gc(~(x < y | (x < z)), [next(x) == tch.Tensor([0, 0, 0])])

    Guard[(x < y | ((x < z) & (x < y)))] >> [next(x) == (x + 1)],
    Guard[~(x < y | (x < z))]            >> [next(x) == tch.Tensor([0, 0, 0])]

module = ctx.module(["x", "y", "z", "y0", "z0"], init, update)
module.dbg()
module.to_html(ctx.context_, "/tmp/mod.html")

