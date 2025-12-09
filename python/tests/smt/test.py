from pysmt.shortcuts import Symbol, Or, LT, Int, Not, Ite, Plus, Real, Minus, And, Div, Times, Equals, Bool, get_model, Iff
from pysmt.typing import INT, REAL, BOOL
from pysmt.logics import QF_NRA
import zrth.smt as smt

class Module(smt.Module):

    def init(self, extl) -> None:
        a, b, c = extl
        x_i = Plus(a, Real(3.24))
        y_i = Minus(b, Int(42))
        z_i = And(c, Not(Bool(True)))
        return x_i, y_i, z_i

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        x_u = Div(Real(12.3), Times(x, Real(4.2)))
        y_u = y
        z_u = Ite(Or(LT(x, Real(50.05)), Equals(y, Int(0))), Not(z), Bool(False))
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
m.to_html("/tmp/smt.html", open=False)

# Init Obligations
def obligation1(m):
    x_i, y_i, z_i = m.init(extl)
    pre = And(a >= Real(0), b >= Int(42), Or(c, Not(c)))
    post = And(x_i >= Real(0), y_i >= Int(0), Not(z_i))
    return pre, post

# Update Obligations on x and y
def obligation2(m):
    x_u, y_u, z_u = m.update(ctrl, extl)
    pre = And(x > Real(0), y >= Int(0))
    post = And(x_u >= Real(0), y_u >= Int(0))
    return pre, post

# When condition triggers, z gets flipped
def obligation3(m):
    x_u, y_u, z_u = m.update(ctrl, extl)
    pre = And(Or(LT(x, Real(50.05)), Equals(y, Int(0))), Not(z))
    post = z_u
    return pre, post

# When condition doesn't trigger, z_u is always False
def obligation4(m):
    x_u, y_u, z_u = m.update(ctrl, extl)
    pre = And(x >= Real(50.05), Not(Equals(y, Int(0))))
    post = Not(z_u)
    return pre, post

# When condition triggers, z gets flipped (general case)
def obligation5(m):
    x_u, y_u, z_u = m.update(ctrl, extl)
    pre = Or(LT(x, Real(50.05)), Equals(y, Int(0)))
    post = Iff(z_u, Not(z))
    return pre, post

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

failed = False
obligations = [obligation1(m), obligation2(m), obligation3(m), obligation4(m), obligation5(m)]
for n, (pre, post) in enumerate(obligations):
    print(f"Obligation {n+1}\n", end="")
    if is_valid(pre, post):
        print("\033[1;32mProved\033[0m")
    else:
        print("\033[1;31mNOT Proved\033[0m")
        failed = True
        break

if failed:
    print("\033[1;31mProof Failed!\033[0m")
else:
    print("\033[1;32mAll Proved!\033[0m")


##################################################################
# COUNTER

# class Module(smt.Module):

#     def init(self, extl) -> None:
#         y0, z0 = extl
#         return 0, y0, z0  # = x, y, z

#     def update(self, ctrl, extl) -> None:
#         x, y, z = ctrl
#         xn = Ite(Or(x < y, x < z), x + Int(1), Int(0))

#         return xn, y, z


# m1 = Module("x: Int, y: Int, z: Int", ("y0: Int", "z0: Int"))
# m1.dbg()
# m1.to_html("/tmp/smt.html", open=False)

# def buchi(a, b, c):
#     return Or(a.Equals(b), a.Equals(c))


# def inv(a, b, c):
#     return Or(a <= b, a <= c)


# def rank(a, b, c):
#     return Plus(
#         Ite(b - a >= Int(0), b - a, Int(0)), Ite(c - a >= Int(0), c - a, Int(0))
#     )

# def obligation1(m):
#     return And(y0 >= Int(0), z0 >= Int(0)), inv(*m.init((y0, z0)))

# def obligation2(m):
#     return (
#         And(inv(x, y, z), Not(buchi(x, y, z))),
#         rank(*m.update((x, y, z), None)) < rank(x, y, z),
#     )

# def is_valid(pre, post):
#     # print("PRE: ", pre.serialize())
#     # print("POST: ", post.serialize())
#     # print("PROVING: ", And(pre, Not(post)).simplify().serialize())
#     m = get_model(And(pre, Not(post)), solver_name="cvc5")
#     if m is None:
#         # print("\033[1;34m.. PROVED\033[0m")
#         return True
#     # print("\033[1;31m.. NOT PROVED\033[0m\n", m)
#     return False

# failed = False
# obligations = [obligation1(m1), obligation2(m1)]
# for n, (pre, post) in enumerate(obligations):
#     print(f"Obligation {n} ... ", end="")
#     if is_valid(pre, post):
#         print("\033[1;32mproved\033[0m")
#     else:
#         print("\033[1;31NOT proved\033[0m")
#         failed = True
#         break

# if failed:
#     print("\033[1;31mProof failed!\033[0m")
# else:
#     print("\033[1;32mAll proved!\033[0m")

##################################################################
# COMPOSITION

# m2 = Module()
#
# m2.rename({'x': 'w'})
# m2.hide('y', 'z')
#
# # Compose the modules
# m = m1 | m2
#
# smt_m = m.to('smt')

##################################################################