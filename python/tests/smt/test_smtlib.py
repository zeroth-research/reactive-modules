from pysmt.shortcuts import Symbol, Or, Int, Ite
from pysmt.typing import INT
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import Solver
from io import StringIO
import zrth.smt as smt

nxt = smt.nxt

class Module(smt.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return Int(0), nxt(y0), nxt(z0)

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = Ite(Or(x < y, x < z), x + Int(1), Int(0))

        return xn, y, z

x, y, z = (Symbol(v, INT) for v in ("x", "y", "z"))
y0, z0 = (Symbol(v, INT) for v in ("y0", "z0"))

ctrl = (x, y, z)
extl = (y0, z0)

m = Module(ctrl=ctrl, extl=extl)
# m.dbg()
m.to_html("/tmp/smt.html", open=False)

smtlib_str = m.to_smtlib()

obligations_smtlib = """

;;; Obligations

;; Obligation 1: (y0 >= 0) ∧ (z0 >= 0) => inv(init(y0, z0))
; where inv(a, b, c) = (a <= b) ∨ (a <= c)
; After init: x'=0, y'=y0', z'=z0'
; So: inv(x' = 0, y' = y0', z' = z0') = (0 <= y0') ∨ (0 <= z0')

(assert (not (=> (and (>= w1 0) (>= w3 0))
            (or (<= w0 w2) (<= w0 w4)))))

;; Obligation 2: inv(x,y,z) ∧ ¬buchi(x,y,z) => rank(x',y',z') < rank(x,y,z)
; where buchi(a,b,c) = (a = b) ∨ (a = c)
; and rank(a,b,c) = max(0, b-a) + max(0, c-a)

(assert (not (=> (and (or (<= w5 w6) (<= w5 w8))
                 (not (or (= w5 w6) (= w5 w8))))
            (< (+ (ite (>= (- w2 w0) 0) (- w2 w0) 0)
                  (ite (>= (- w4 w0) 0) (- w4 w0) 0))
               (+ (ite (>= (- w6 w5) 0) (- w6 w5) 0)
                  (ite (>= (- w8 w5) 0) (- w8 w5) 0))))))

(check-sat)
(get-model)
"""

# Append obligations to the SMT-LIB module string
full_smtlib = smtlib_str + obligations_smtlib

with open("./tests/smt/module_with_obligations.smt2", "w") as f:
    f.write(full_smtlib)

parser = SmtLibParser()
script = parser.get_script(StringIO(full_smtlib))

# Get all assertions from the script
from pysmt.shortcuts import And
assertions = [cmd.args[0] for cmd in script.commands if cmd.name == "assert"]
formula = And(assertions)

# Check satisfiability (unsat means the obligations are proven)
with Solver() as solver:
    solver.add_assertion(formula)
    result = solver.solve()
    
    if result:
        print("\033[1;31mSAT - Obligations FAILED! Found counterexample:\033[0m")
        model = solver.get_model()
        print(model)
    else:
        print("\033[1;32mUNSAT - All obligations PROVED!\033[0m")