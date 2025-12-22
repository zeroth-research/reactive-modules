from pysmt.shortcuts import Symbol, Or, LT, Int, Not
import zrth.toy as toy


class Module(toy.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return 0, self.nxt(y0), self.nxt(z0)  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        yn = Choose(Case(True, True))
        xn = Choose(
            Case(yn, x + 1),
            Case(Or(x < y, x < z), x + Int(1)),
            Case(Not(Or(x < y, x < z)), Int(0)),
        )

        return xn, y, z


m1 = Module("x: Int, y: Int, z: Int", ("y0: Int", "z0: Int"))
m1.dbg()
m1.to_html("/tmp/toy.html", open=True)

# m2 = Module()
#
# m2.rename({'x': 'w'})
# m2.hide('y', 'z')
#
# # Compose the modules
# m = m1 | m2
#
# smt_m = m.to('smt')
