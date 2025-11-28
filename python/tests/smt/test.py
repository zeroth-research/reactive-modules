from pysmt.shortcuts import Symbol, Or, LT, Int, Not, Ite
import zrth.smt as smt


class Module(smt.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return 0, y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = Ite(Or(x < y, x < z), x + Int(1), Int(0))

        return xn, y, z


m1 = Module("x: Int, y: Int, z: Int", ("y0: Int", "z0: Int"))
m1.dbg()
m1.to_html("/tmp/smt.html", open=True)

# m2 = Module()
#
# m2.rename({'x': 'w'})
# m2.hide('y', 'z')
#
# # Compose the modules
# m = m1 | m2
#
# smt_m = m.to('smt')
