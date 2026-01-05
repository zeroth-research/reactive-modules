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

# ################################
# Automated module unrolling
# ################################
U = smt.ModuleUnrolling(m)
U.init()
for i in range(5):
    U.step()
# U.dbg()

# ################################
# Manual module unrolling (still without manually creating states)
# ################################
ctx = m.ctx()
U = smt.Unrolling()
T = m.init_as_transition()

# wire in the initial transition (since the unrolling is empty,
# this is basically `push`)
U.wire_transition(T, ctx)

T = m.update_as_transition()
# wire the `update` transition 3x to the unrolling
for i in range(3):
    U.wire_transition(T, ctx)

U.dbg()


# ################################################################
# Fully manual module unrolling (sketch, code not working (yet?))
# ################################################################
I = m.init_as_transition()

raise NotImplementedError("Not implemented from here")
last_out = I.output(ctx)
I = I.as_pysmt_expr()

unrolling = [I]

T = m.update_as_transition()
T_in = T.input(ctx)
T_out = T.output(ctx)
T_env = [e for e_pair in T.env(ctx) for e in e_pair]
T = T.as_pysmt_expr()

for i in range(3):
    new_out = [smt.fresh_var(v.dtype()) for v in last_out]
    new_env = [smt.fresh_var(v.dtype()) for v in T_env]
    new_T = T.subst(list(zip(T_in, last_out)) +
                    list(zip(T_out, new_out)) +
                    list(zip(T_env, new_env)))
    unrolling.append(new_T)
    last_out = new_out
