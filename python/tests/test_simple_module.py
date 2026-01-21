import sys
from os.path import dirname, join as pathjoin

from torch import Tensor

from zrth import DType
from zrth.expr import nxt, ite, sym
from zrth.module import Module


class MyModule(Module):
    def init(self, extl):
        # extl is a vector with dimension 2
        return Tensor([[0, 0], [1, 0], [0, 1]]) @ nxt(extl)

    def update(self, state, inp):
        # state = (x, y, z) is a vector with dimension 3,
        # inp = (y0, z0) is a vector with dimension 2
        result1 = state + Tensor([1, 0, 0])
        result2 = Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ state
        x = Tensor([1, 0, 0]) @ state
        y = Tensor([0, 1, 0]) @ state
        z = Tensor([0, 0, 1]) @ state

        cond = (x < y) or (x < z)
        return ite(cond, result1, result2)


def test_ctor():
    m = MyModule(ctrl="xyz: Tensor<3>", extl="yz0: Tensor<2>")
    assert m
    print(m)
    # m.to_html("/tmp/torch.html", open=True)


def test_execute_concrete():
    m = MyModule(ctrl="xyz: Tensor<3>", extl="yz0: Tensor<2>")
    assert m

    state = Tensor([0, 1, 2])
    extl = Tensor([3, 4])
    print("## Concrete state:")
    print(m.update(state, extl))
    print("-------")


def test_execute_symbolic():
    m = MyModule(ctrl="xyz: Tensor<3>", extl="yz0: Tensor<2>")
    assert m

    extl = sym("extl", DType.Tensor([2]))
    s = sym("state", DType.Tensor([3]))
    s_init = m.init(extl)
    print("## Symbolic init:")
    print(s_init)

    extl = extl.fresh("extl2")
    s = m.update(s_init, extl)
    print("## Symbolic state:")
    print(s)

    extl = extl.fresh("extl3")
    s = m.update(s, extl)
    print("## Symbolic state:")
    print(s)


if __name__ == "__main__":
    test_execute_concrete()
    test_execute_symbolic()
