import sys
from os.path import dirname, join as pathjoin
sys.path.append(pathjoin(dirname(__file__), "../python"))

from zrmtorch.module import Module
import torch as tch


def fun(x, y):
    # straightline programs can be written without using our API
    t = tch.Tensor([1, 1, 1])
    s = (2 * x * t + y).sum()

    # however, we cannot do branching (if-then-else) when tracing,
    # so for conditions and branches, we need to use our API
    b = zrm.eq(s, 0)
    return zrm.ifthenelse(b, x, x + 1)


module = Module.from_fn(fun)
module.to_html("/tmp/fn.html", open=True)


