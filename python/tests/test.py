from pysmt.shortcuts import Symbol, Or, LT, Int, Not, Ite
import zrth.smt as smt
import zrth.toy as toy

class SmtModule(smt.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return 0, y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn =  Ite(Or(x < y, x < z), x + Int(1), Int(0))

        return xn, y, z


m1 = SmtModule("x: Int, y: Int, z: Int", ("y0: Int", "z0: Int"))
m1.dbg()
m1.to_html("/tmp/smt.html", open=True)


class ToyModule(toy.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return 0, y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn =  Choose(
            Case(Or(x < y, x < z), x + Int(1)),
            Case(Not(Or(x < y, x < z)), Int(0)),
        )

        return xn, y, z


m2 = ToyModule("x: Int, y: Int, z: Int", ("y0: Int", "z0: Int"))
m2.dbg()
m2.to_html("/tmp/toy.html", open=True)


