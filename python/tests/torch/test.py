import sys
from os.path import dirname, join as pathjoin


from torch import Tensor

from zrth.torch import Module as TorchModule


class MyModule(TorchModule):

    def init(self, extl_nxt):
        # extl is a vector with dimension 2
        return Tensor([[0, 0], [1, 0], [0, 1]]) @ extl_nxt

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
           (cond, result1),
           (~cond, result2),
        )


m = MyModule(ctrl="xyz", extl="yz0")
m.dbg()
# m.to_html("/tmp/torch.html", open=True)


extl = m.constant([3, 4])
s = m.init(extl)
print("## Concrete init:")
print(s)
extl = m.constant([4, 2])
print("## Concrete state:")
print(m.update(s, extl))

print("-------")


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

