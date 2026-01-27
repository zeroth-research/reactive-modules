from torch import IntTensor

from zrth import DType
from zrth.expr import nxt, ite, sym
from zrth import ReactiveModule
import zrth.bmc as bmc


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


def test_unroll_auto():
    """
    Automated module unrolling
    """

    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m
    # m.to_html("/tmp/torch.html", open=True)

    U = bmc.ModuleUnrolling(m)
    U.init()
    for i in range(10):
        U.step()
    # U.dbg()
    # U.to_html("/tmp/unroll.html", open=True)


def test_unroll_semimanual():
    """
    Semi-manual module unrolling
    """

    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m

    U = bmc.WiredTransitions()
    T = m.init_as_transition()

    # wire in the initial transition (since the unrolling is empty,
    # this is basically `push`)
    U.wire_transition(T)

    T = m.update_as_transition()
    # wire the `update` transition 3x to the unrolling
    for i in range(3):
        U.wire_transition(T)

    U.dbg()


if __name__ == "__main__":
    test_unroll_auto()
    test_unroll_semimanual()

# # ################################
# # Manual module unrolling (still without manually creating states)
# # ################################

#
# # ################################################################
# # Fully manual module unrolling (sketch, code not working (yet?))
# # ################################################################
# I = m.init_as_transition()
#
# raise NotImplementedError("Not implemented from here")
# last_out = I.output(ctx)
# I = I.as_pysmt_expr()
#
# unrolling = [I]
#
# T = m.update_as_transition()
# T_in = T.input(ctx)
# T_out = T.output(ctx)
# T_env = [e for e_pair in T.env(ctx) for e in e_pair]
# T = T.as_pysmt_expr()
#
# for i in range(3):
#     new_out = [smt.fresh_var(v.dtype()) for v in last_out]
#     new_env = [smt.fresh_var(v.dtype()) for v in T_env]
#     new_T = T.subst(
#         list(zip(T_in, last_out))
#         + list(zip(T_out, new_out))
#         + list(zip(T_env, new_env))
#     )
#     unrolling.append(new_T)
#     last_out = new_out
