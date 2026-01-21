from sys import stderr
from io import StringIO
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import (
    TRUE,
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
from zrth.module import ReactiveModule
from torch import Tensor
import zrth.smt as smt
from zrth.expr import nxt, ite

from pysmt.environment import Environment, reset_env, get_env
import pytest


# make sure every test gets its own new PySMT environment
# to avoid Symbol clashes
from .smt.test_basic import pysmt_fresh_env

######################################################################
# SMT
######################################################################


class SmtModule(smt.Module):
    def init(self, extl) -> None:
        y0, z0 = extl
        return Int(0), self.nxt(y0), self.nxt(z0)  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = Ite(Or(x < y, x < z), x + Int(1), Int(0))

        return xn, y, z


def test_counter_smt():
    # other tests will use PySMT too, so isolate the environment

    x, y, z = (Symbol(v, INT) for v in ("x", "y", "z"))
    y0, z0 = (Symbol(v, INT) for v in ("y0", "z0"))
    m_smt = SmtModule(ctrl=(x, y, z), extl=(y0, z0))
    assert m_smt
    # m_smt.to_html("/tmp/smt.html", open=True)


######################################################################
# Torch
######################################################################


class TorchModule(ReactiveModule):
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


def test_counter_torch():
    m_torch = TorchModule(ctrl="xyz: Tensor<3>", extl="yz0: Tensor<2>")
    assert m_torch
    print(m_torch)


######################################################################
# Obligations
######################################################################


def buchi(a, b, c):
    return Or(a.Equals(b), a.Equals(c))


def inv(a, b, c):
    return Or(a <= b, a <= c)


def rank(a, b, c):
    return Plus(
        Ite(b - a >= Int(0), b - a, Int(0)), Ite(c - a >= Int(0), c - a, Int(0))
    )


def is_valid(pre, post):
    # print("PRE: ", pre.serialize())
    # print("POST: ", post.serialize())
    # print("PROVING: ", And(pre, Not(post)).simplify().serialize())
    m = get_model(And(pre, Not(post)), solver_name="cvc5")
    if m is None:
        return True
    return False


def test_obligations():
    pytest.importorskip("cvc5")

    x, y, z = (Symbol(v, INT) for v in ("x", "y", "z"))
    y0, z0 = (Symbol(v, INT) for v in ("y0", "z0"))

    m_smt = SmtModule(ctrl=(x, y, z), extl=(y0, z0))

    def obligation1(m):
        return And(smt.nxt(y0) >= Int(0), smt.nxt(z0) >= Int(0)), inv(*m.init((y0, z0)))

    # TODO: now the obligation uses m.update() which already are PySMT formulas.
    # We need to translate update from the reactive module to PySMT and use it.
    def obligation2(m):
        return (
            And(inv(x, y, z), Not(buchi(x, y, z))),
            rank(*m.update((x, y, z), None)) < rank(x, y, z),
        )

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

    assert not failed


if __name__ == "__main__":
    test_counter_toy()
    test_counter_smt()
    test_counter_torch()
