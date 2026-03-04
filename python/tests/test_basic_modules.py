# from zrth import ReactiveModule
# from torch import IntTensor
#
# ######################################################################
# # Torch
# ######################################################################
#
#
# class TorchModule(ReactiveModule):
#     def init(self, extl):
#         # extl is a vector with dimension 2
#         return IntTensor([[0, 0], [1, 0], [0, 1]]) @ nxt(extl)
#
#     def update(self, state, inp):
#         # state = (x, y, z) is a vector with dimension 3,
#         # inp = (y0, z0) is a vector with dimension 2
#         result1 = state + IntTensor([1, 0, 0])
#         result2 = IntTensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ state
#         x = IntTensor([1, 0, 0]) @ state
#         y = IntTensor([0, 1, 0]) @ state
#         z = IntTensor([0, 0, 1]) @ state
#
#         cond = (x < y) or (x < z)
#         return ite(cond, result1, result2)
#
#
# def test_counter_torch():
#     m_torch = TorchModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
#     assert m_torch
#     print(m_torch)
#
#
# ######################################################################
# # Obligations
# ######################################################################
#
#
# # def buchi(a, b, c):
# #     return Or(a.Equals(b), a.Equals(c))
# #
# #
# # def inv(a, b, c):
# #     return Or(a <= b, a <= c)
# #
# #
# # def rank(a, b, c):
# #     return Plus(
# #         Ite(b - a >= Int(0), b - a, Int(0)), Ite(c - a >= Int(0), c - a, Int(0))
# #     )
# #
# #
# # def is_valid(pre, post):
# #     # print("PRE: ", pre.serialize())
# #     # print("POST: ", post.serialize())
# #     # print("PROVING: ", And(pre, Not(post)).simplify().serialize())
# #     m = get_model(And(pre, Not(post)), solver_name="cvc5")
# #     if m is None:
# #         return True
# #     return False
# #
# #
# # def test_obligations():
# #     from pytest import importoskip
# #     importorskip("cvc5")
# #
# #     x, y, z = (Symbol(v, INT) for v in ("x", "y", "z"))
# #     y0, z0 = (Symbol(v, INT) for v in ("y0", "z0"))
# #
# #     m_smt = SmtModule(ctrl=(x, y, z), extl=(y0, z0))
# #
# #     def obligation1(m):
# #         return And(smt.nxt(y0) >= Int(0), smt.nxt(z0) >= Int(0)), inv(*m.init((y0, z0)))
# #
# #     # TODO: now the obligation uses m.update() which already are PySMT formulas.
# #     # We need to translate update from the reactive module to PySMT and use it.
# #     def obligation2(m):
# #         return (
# #             And(inv(x, y, z), Not(buchi(x, y, z))),
# #             rank(*m.update((x, y, z), None)) < rank(x, y, z),
# #         )
# #
# #     obligations = [obligation1(m_smt), obligation2(m_smt)]
# #
# #     failed = False
# #     for n, (pre, post) in enumerate(obligations):
# #         print(f"Obligation {n} ... ", end="")
# #         if is_valid(pre, post):
# #             print("\033[1;32mproved\033[0m")
# #         else:
# #             print("\033[1;31NOT proved\033[0m")
# #             failed = True
# #             break
# #
# #     if failed:
# #         print("\033[1;31mProof failed!\033[0m")
# #     else:
# #         print("\033[1;32mAll proved!\033[0m")
# #
# #     assert not failed
# #
#
# if __name__ == "__main__":
#     test_counter_torch()
