from pysmt.shortcuts import (
    Symbol,
    Or,
    LT,
    Int,
    Not,
    Ite,
    Plus,
    Real,
    Minus,
    And,
    Div,
    Times,
    Equals,
    Bool,
    get_model,
    Iff,
)
from pysmt.typing import INT, REAL, BOOL
from pysmt.logics import QF_NRA
import zrth.smt as smt

from pysmt.environment import Environment, reset_env, get_env
import pytest


# make sure every test gets its own new PySMT environment
# to avoid Symbol clashes
@pytest.fixture(autouse=True)
def pysmt_fresh_env():
    reset_env()
    get_env().enable_infix_notation = True


nxt = smt.nxt


class Module(smt.Module):
    def init(self, extl) -> None:
        y0, z0 = extl
        return Int(0), nxt(y0), nxt(z0)  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl

        cond = Or(x < y, x < z)
        xn = Ite(cond, x + Int(1), Int(0))

        return xn, y, z


class Inv(smt.Module):
    def init(self, state) -> None:
        x, y, z = state
        return Or(nxt(x) <= nxt(y), nxt(x) <= nxt(z))

    def update(self, inv, state) -> None:
        x, y, z = state
        return Or(nxt(x) <= nxt(y), nxt(x) <= nxt(z))


def is_valid(pre, post):
    # print(" PRE: ", pre.serialize())
    # print(" POST: ", post.serialize())
    # print(" PROVING: ", And(pre, Not(post)).simplify().serialize())
    m = get_model(And(pre, Not(post)), solver_name="cvc5", logic=QF_NRA)
    if m is None:
        # print("\033[1;34m.. PROVED\033[0m")
        return True
    # print("\033[1;31m.. NOT PROVED\033[0m\n", m)
    return False


def test_obligations():
    x, y, z, y0, z0 = (Symbol(v, INT) for v in ("x", "y", "z", "y0", "z0"))
    inv = Symbol("inv", BOOL)

    ctx = smt.Context()
    m = Module(ctrl=(x, y, z), extl=(y0, z0), ctx=ctx)
    print("=== m ===")
    m_inv = Inv(ctrl=(inv,), extl=(x, y, z), ctx=ctx)
    print("=== m_inv ===")

    m_obl = smt.Module.parallel([m, m_inv])
    # m_obl.to_html(ctx.unwrap(), "/tmp/m_obl.html")
    print(m_obl)

    print(m_obl.to_smtlib())

    # Init Obligations

    # failed = False
    # obligations = [
    #     obligation1(m),
    #     obligation2(m),
    # ]
    # for n, (pre, post) in enumerate(obligations):
    #     print(f"Obligation {n + 1}\n", end="")
    #     if is_valid(pre, post):
    #         print("\033[1;32mProved\033[0m")
    #     else:
    #         print("\033[1;31mNOT Proved\033[0m")
    #         failed = True
    #         break
    #
    # if failed:
    #     print("\033[1;31mProof Failed!\033[0m")
    # else:
    #     print("\033[1;32mAll Proved!\033[0m")
    #
    # assert not failed
