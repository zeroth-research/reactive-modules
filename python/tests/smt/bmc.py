from pysmt.shortcuts import Symbol, Or, LT, Int, Not, Ite, Plus, Real, Minus, And, Div, Times, Equals, Bool, get_model, Iff
from pysmt.typing import INT, REAL, BOOL
from pysmt.logics import QF_NRA
import zrth.smt as smt


class Module(smt.Module):

    def init(self, extl_nxt) -> None:
        a, b, c = extl_nxt
        x_i = Plus(a, Real(3.24))
        y_i = Minus(b, Int(42))
        z_i = And(c, Not(Bool(True)))
        return x_i, y_i, z_i

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        x_u = Div(Real(12.3), Times(x, Real(4.2)))
        y_u = y
        z_u = Ite(Or(LT(x, Real(50.05)), Equals(y, Int(0))),
                  Not(z), Bool(False))
        return x_u, y_u, z_u


x = Symbol("x", REAL)
y = Symbol("y", INT)
z = Symbol("z", BOOL)

a = Symbol("a", REAL)
b = Symbol("b", INT)
c = Symbol("c", BOOL)

ctrl = (x, y, z)
extl = (a, b, c)

m = Module(ctrl=ctrl, extl=extl)
# m.dbg()
# m.to_html("/tmp/smt.html", open=False)


###############

# I, T = m.to_transitions()
#
# U = smt.Unrolling()
# state = m.fresh_variables()
# U += Transition(m.new_env(), I, state)
#
# for i in range(5):
#   old_state = state
#   state = m.fresh_variables()
#   new_extl = m.fresh_env()
#
#   # unroll
#   U += Transition(old_state, T, (new_state, new_extl))


U = smt.ModuleUnrolling(m)
U.init()
for i in range(5):
    U.step()
U.dbg()

U = smt.Unrolling()
U.wire_transition(m.init_as_transition())

T = m.update_as_transitions()
for i in range(3):
    U.wire_transition(T)

U.dbg()
# U.terms()
# U.as_pysmt_expr()
# print(U.last_state())
