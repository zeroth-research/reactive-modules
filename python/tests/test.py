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
import zrth.smt as smt
import zrth.toy as toy


class SmtModule(smt.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return Int(0), y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = Ite(Or(x < y, x < z), x + Int(1), Int(0))

        return xn, y, z


x, y, z = (Symbol(v, INT) for v in ("x", "y", "z"))
y0, z0 = (Symbol(v, INT) for v in ("y0", "z0"))
m1 = SmtModule(ctrl=(x, y, z), extl=(y0, z0))
m1.to_html("/tmp/smt.html", open=True)


class ToyModule(toy.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return 0, y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = Choose(
            Case(Or(x < y, x < z), x + Int(1)),
            Case(Not(Or(x < y, x < z)), Int(0)),
        )

        return xn, y, z


m2 = ToyModule("x: Int, y: Int, z: Int", ("y0: Int", "z0: Int"))
# m2.dbg()
m2.to_html("/tmp/toy.html", open=True)


# m = m1 | m2
m = m1


def buchi(a, b, c):
    return Or(a.Equals(b), a.Equals(c))


def inv(a, b, c):
    return Or(a <= b, a <= c)


def rank(a, b, c):
    return Plus(
        Ite(b - a >= Int(0), b - a, Int(0)), Ite(c - a >= Int(0), c - a, Int(0))
    )


# FIXME: now the obligation uses m.init() which already are PySMT formulas.
# We need to translate init from the reactive module to PySMT and use it
# In this example, it does not make any difference as the PySMT formulas
# should be exactly the same (or equivalent), because we simply go
# PySMT -> list of terms -> PySMT. It becomes interesting when the list
# of terms gets modified (e.g., by composing with another module).
def obligation1(m):
    return And(y0 >= Int(0), z0 >= Int(0)), inv(*m.init((y0, z0)))


# FIXME: now the obligation uses m.update() which already are PySMT formulas.
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
        # print("\033[1;34m.. PROVED\033[0m")
        return True
    # print("\033[1;31m.. NOT PROVED\033[0m\n", m)
    return False


failed = False
obligations = [obligation1(m1), obligation2(m1)]
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
