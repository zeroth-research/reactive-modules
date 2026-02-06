import sys
from os.path import dirname, join as pathjoin

from torch import IntTensor

from zrth import DType
from zrth.expr import nxt, ite, sym
from zrth import ReactiveModule


class MyModule(ReactiveModule):
    def init(self, extl):
        # extl is a vector with dimension 2
        return IntTensor([[0, 0], [1, 0], [0, 1]]) @ nxt(extl)

    def update(self, state, inp):
        # state = (x, y, z) is a vector with dimension 3,
        # inp = (y0, z0) is a vector with dimension 2
        result1 = state + IntTensor([1, 0, 0])
        result2 = IntTensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ state
        x = IntTensor([1, 0, 0]) @ state
        y = IntTensor([0, 1, 0]) @ state
        z = IntTensor([0, 0, 1]) @ state

        cond = (x < y) or (x < z)
        return ite(cond, result1, result2)


def test_ctor():
    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m
    print(m)
    # m.to_html("/tmp/torch.html", open=True)


def test_execute_concrete():
    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m

    state = IntTensor([0, 1, 2])
    extl = IntTensor([3, 4])
    print("## Concrete state:")
    print(m.update(state, extl))
    print("-------")


def test_execute_symbolic():
    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m

    extl = sym("extl", DType.TensorInt([2]))[0]
    s = sym("state", DType.TensorInt([3]))[0]
    s_init = m.init(extl)
    print("## Symbolic init:")
    print(s_init)

    extl = sym("extl2", DType.TensorInt([2]))[0]
    s = m.update(s_init, extl)
    print("## Symbolic state:")
    print(s)

    extl = sym("extl3", DType.TensorInt([2]))[0]
    s = m.update(s, extl)
    print("## Symbolic state:")
    print(s)


if __name__ == "__main__":
    test_execute_concrete()
    test_execute_symbolic()
