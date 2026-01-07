import sys
from os.path import dirname, join as pathjoin


from torch import Tensor

from zrth.torch import Module as TorchModule, IfThen


class MyModule(TorchModule):
    def init(self, extl):
        # extl is a vector with dimension 2
        return Tensor([[0, 0], [1, 0], [0, 1]]) @ self.nxt(extl)

    def update(self, state, inp):
        # state = (x, y, z) is a vector with dimension 3,
        # inp = (y0, z0) is a vector with dimension 2
        result1 = state + Tensor([1, 0, 0])
        result2 = Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ state
        x = Tensor([1, 0, 0]) @ state
        y = Tensor([0, 1, 0]) @ state
        z = Tensor([0, 0, 1]) @ state

        cond = (x < y) or (x < z)
        return self.choose(
            IfThen(cond, result1),
            IfThen(~cond, result2),
        )


def test_ctor():
    m = MyModule(ctrl="xyz", extl="yz0")
    assert m
    m.dbg()
    # m.to_html("/tmp/torch.html", open=True)


def test_execute_concrete():
    m = MyModule(ctrl="xyz", extl="yz0")
    assert m

    extl = m.constant([3, 4])
    s = m.init(extl)
    print("## Concrete init:")
    print(s)
    extl = m.constant([4, 2])
    print("## Concrete state:")
    print(m.update(s, extl))
    print("-------")


def test_execute_symbolic():
    m = MyModule(ctrl="xyz", extl="yz0")
    assert m

    extl = m.fresh_variable()
    s = m.fresh_variable()
    s_init = m.init(s)
    print("## Symbolic init:")
    print(s_init)

    extl = m.fresh_variable()
    s = m.update(s_init, extl)
    print("## Symbolic state:")
    print(s)

    extl = m.fresh_variable()
    s = m.update(s, extl)
    print("## Symbolic state:")
    print(s)


if __name__ == "__main__":
    test_execute_concrete()
    test_execute_symbolic()
