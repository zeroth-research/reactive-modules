from io import StringIO
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import (
    Symbol,
    Or,
    Int,
    Not,
    Ite,
    And,
    Plus,
    simplify,
    get_model,
)
from pysmt.typing import INT
import zrth.torch as ztch
from torch import Tensor
import zrth.smt as smt
import zrth.toy as toy
import sympy as sp

######################################################################
# SMT
######################################################################


class SmtModule(smt.Module):

    def init(self, extl_nxt) -> None:
        y0, z0 = extl_nxt
        return Int(0), y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = Ite(Or(x < y, x < z), x + Int(1), Int(0))

        return xn, y, z


x, y, z = (Symbol(v, INT) for v in ("x", "y", "z"))
y0, z0 = (Symbol(v, INT) for v in ("y0", "z0"))
m_smt = SmtModule(ctrl=(x, y, z), extl=(y0, z0))
# m_smt.to_html("/tmp/smt.html", open=True)


######################################################################
# Toy
######################################################################


class ToyModule(toy.Module):

    def init(self, extl_nxt) -> None:
        y0, z0 = extl_nxt
        return 0, y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = Ite(Or(x < y, x < z), x + Int(1), Int(0))
       # xn = Choose(
       #    Case(Or(x < y, x < z), x + Int(1)),
       #    Case(Not(Or(x < y, x < z)), Int(0)),
       # )

        return xn, y, z


m_toy = ToyModule("x: Int, y: Int, z: Int", ("y0: Int", "z0: Int"))
# m_toy.dbg()
# m_toy.to_html("/tmp/toy.html", open=True)


######################################################################
# Torch
######################################################################


class TorchModule(ztch.Module):

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


m_torch = TorchModule(ctrl="xyz", extl="yz0")
m_torch.dbg()


######################################################################
# Obligations
######################################################################

# compose modules
# m = m1 | m2
m = m_toy.translate_to('smt')
print(m)
m.dbg()


def buchi(a, b, c):
    return Or(a.Equals(b), a.Equals(c))


def inv(a, b, c):
    return Or(a <= b, a <= c)


def rank(a, b, c):
    return Plus(
        Ite(b - a >= Int(0), b - a, Int(0)), Ite(c - a >= Int(0), c - a, Int(0))
    )


smtlib_str = m.to_smtlib()


print(smtlib_str)
script = SmtLibParser().get_script(StringIO(smtlib_str))
print(script)

# print('=======')
# exprs = []
# for a in script.filter_by_command_name("assert"):
#     print(a.args)
#     exprs.extend(a.args)
#
# X, Y, Z = exprs[-3:]
# print(X, Y, Z)
# print(And(exprs))
#


def obligation1(m):
    return And(y0 >= Int(0), z0 >= Int(0)), inv(*m.init((y0, z0)))


# TODO: now the obligation uses m.update() which already are PySMT formulas.
# We need to translate update from the reactive module to PySMT and use it.
def obligation2(m):
    return (
        And(inv(x, y, z), Not(buchi(x, y, z))),
        rank(*m.update((x, y, z), None)) < rank(x, y, z),
    )


def is_valid(pre, post):
    # print("PRE: ", pre.serialize())
    # print("POST: ", post.serialize())
    # print("PROVING: ", And(pre, Not(post)).simplify().serialize())
    m = get_model(And(pre, Not(post)), solver_name="cvc5")
    if m is None:
        return True
    return False


obligations = [obligation1(m_smt), obligation2(m_smt)]


failed = False
for n, (pre, post) in enumerate(obligations):
    print(f"Obligation {n} ... ", end="")
    if is_valid(pre, post):
        print("\033[1;32mproved\033[0m")
    else:
        print("\033[1;31NOT proved\033[0m")
        failed = True
        break

if failed:
    print("\033[1;31mProof failed!\033[0m")
else:
    print("\033[1;32mAll proved!\033[0m")
