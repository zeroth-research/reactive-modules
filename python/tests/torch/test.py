import sys
from os.path import dirname, join as pathjoin


import zrth.torch as ztch
from torch import Tensor


class MyModule(ztch.Module):

    def init(self, extl):
        # extl is a vector with dimension 2
        return extl * Tensor([[0, 0], [1, 0], [0, 1]])

    def update(self, state, inp):
        # state = (x, y, z) is a vector with dimension 3,
        # inp = (y0, z0) is a vector with dimension 2
        result1 = state + Tensor([1, 0, 0])
        result2 = state * Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        x = state * Tensor([1, 0, 0])
        y = state * Tensor([0, 1, 0])
        z = state * Tensor([0, 0, 1])
        return result1

    # return Choose(
    #    Case((x < y | (x < z)), result1),
    #    Case(~(x < y | (x < z)), result2),
    # )


m = MyModule(ctrl="xyz", extl="yz0")
# m.to_html("/tmp/torch.html", open=True)
