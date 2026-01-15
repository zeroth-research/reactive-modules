from io import StringIO
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
    Solver,
    substitute,
)
from pysmt.smtlib.parser import SmtLibParser
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


def test_obligations():
    x, y, z, y0, z0 = (Symbol(v, INT) for v in ("x", "y", "z", "y0", "z0"))
    inv = Symbol("inv", BOOL)

    ctx = smt.Context()
    m = Module(ctrl=(x, y, z), extl=(y0, z0), ctx=ctx)
    m_inv = Inv(ctrl=(inv,), extl=(x, y, z), ctx=ctx)
    m_obl = smt.Module.parallel([m, m_inv])

    print(m_obl)

    ############
    ## Prove the invariant
    ############

    # first, reset the PySMT environment to make sure
    # nothing interfers with creating modules
    reset_env()
    get_env().enable_infix_notation = True

    parser = SmtLibParser()

    # get smtlib string for variables
    decls = m_obl.to_smtlib("variables")

    # declare variables (`get_script` will load them into the PySMT environment)
    parser.get_script(StringIO(decls))

    # map module variables to variables in the formula
    env = get_env()
    fm = env.formula_manager
    intf_wires = {
        s: (
            fm.get_symbol(f"w{ctx.get_wire_id(s)}"),
            fm.get_symbol(f"w{ctx.get_wire_id(nxt(s))}"),
        )
        for s in (x, y, z, y0, z0, inv)
    }

    # get the smtlib string for invariant predicate (it doesn't matter if we use init or update here)
    inv_str = m_inv.to_smtlib("init")
    init_str = m_obl.to_smtlib("init")
    update_str = m_obl.to_smtlib("update")

    solver = Solver(name="cvc5", logic="LIA")

    ### Base case

    solver.push()

    # add assumption on initial inputs
    solver.add_assertion(intf_wires[y0][1] >= 0)
    solver.add_assertion(intf_wires[z0][1] >= 0)

    # check invariant violation
    solver.add_assertion(Not(intf_wires[inv][1]))

    script = parser.get_script(StringIO(decls + init_str))
    for formula in (cmd.args[0] for cmd in script.commands if cmd.name == "assert"):
        solver.add_assertion(formula)

    if solver.solve():
        print("\033[1;31mBase case failed!\033[0m")
        print("CEX:")
        for s, w in intf_wires.items():
            print(f"{s} ({w}) = {(solver.get_value(w[0]), solver.get_value(w[1]))}")
    else:
        print("\033[1;32mBase case proved!\033[0m")

    solver.pop()

    ### Induction step

    solver.push()

    # get the invariant and re-map the next values (which represet the output)
    # to latched values (so that they represent the input) to the update
    S = {s[1]: s[0] for s in intf_wires.values()}
    script = parser.get_script(StringIO(decls + inv_str))
    for formula in (cmd.args[0] for cmd in script.commands if cmd.name == "assert"):
        formula = substitute(formula, S)
        solver.add_assertion(formula)

    # state that the assertion holds
    solver.add_assertion(intf_wires[inv][0])

    # basic assumptions
    # TODO: we should not need these!
    solver.add_assertion(intf_wires[x][0] >= 0)
    solver.add_assertion(intf_wires[y][0] >= 0)
    solver.add_assertion(intf_wires[z][0] >= 0)

    # take the update step
    script = parser.get_script(StringIO(decls + update_str))
    for formula in (cmd.args[0] for cmd in script.commands if cmd.name == "assert"):
        solver.add_assertion(formula)

    # check invariant violation
    solver.add_assertion(Not(intf_wires[inv][1]))

    if solver.solve():
        print("\033[1;31mInduction step failed!\033[0m")
        print("CEX:")
        for s, w in intf_wires.items():
            print(f"{s} ({w}) = {(solver.get_value(w[0]), solver.get_value(w[1]))}")
        print("------")
        print(solver.get_model())
    else:
        print("\033[1;32mInduction step proved!\033[0m")

    solver.pop()
