from torch import IntTensor, tensor

# from zrth import DType, get_ctx, Context
# from zrth.eexpr import *
import zrth.expr as expr

#
# def expr_direct():
#     s1, s2 = Sym("x", "Int"), Sym("y", "Int")
#     a = Expr("arith.add", s1, to_expr(6))
#     m = Expr("arith.mul", to_expr(2), a)
#     b = Expr("arith.mul", to_expr(2), s2)
#     g = Expr("cmp.lt", b, a)
#     n = Expr("logic.not", g)
#
#     ctx = get_ctx()
#     assert all(ctx is e.ctx() for e in (a, m, b, g, n))
#
#     return n
#
#
# def expr_ctors():
#     s1, s2 = Sym("x", "Int"), Sym("y", "Int")
#     a = Add(s1, 6)
#     m = Mul(2, a)
#     b = Mul(2, s2)
#     g = Lt(b, a)
#     n = Not(g)
#
#     ctx = get_ctx()
#     assert all(ctx is e.ctx() for e in (a, m, b, g, n))
#
#     return n
#
#
# def expr_funs():
#     s1, s2 = sym("x", "Int"), sym("y", "Int")
#     a = add(s1, 6)
#     m = mul(2, a)
#     b = mul(2, s2)
#     g = lt(b, a)
#     n = lnot(g)
#
#     ctx = get_ctx()
#     assert all(ctx is e.ctx() for e in (a, m, b, g, n))
#
#     return n
#
#
# def expr_methods():
#     s1, s2 = sym("x", "Int"), sym("y", "Int")
#     a = s1.add(6)
#     m = const(2).mul(a)
#     b = const(2).mul(s2)
#     g = b.lt(a)
#     n = g.lnot()
#
#     ctx = get_ctx()
#     assert all(ctx is e.ctx() for e in (a, m, b, g, n))
#
#     return n
#
#
# def test_eq_1():
#     ed = expr_direct()
#     ec = expr_ctors()
#     ef = expr_funs()
#     em = expr_methods()
#
#     assert ed == ec
#     assert ed == ef
#     assert ed == em
#     assert ec == ef
#     assert ec == em
#     assert ef == em
#     assert em == em
#
#
# def test_concrete():
#     a = add(3, 6)
#     m = mul(2, a)
#     b = mul(2, 7)
#     g = lt(b, a)
#     n = lnot(g)
#
#     assert a == 9
#     assert m == 18
#     assert b == 14
#     assert not g
#     assert n
#
#
# def test_argmax_concrete():
#     assert argmax(IntTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])) == 7
#     assert argmax(IntTensor([[2, 2, 2], [2, 2, 2]])) == 0
#


def test_terminal():
    a = expr.Real(0.9)
    b = expr.Real("b")


def test_boolean():
    a = expr.Bool(True)
    b = expr.Bool("a")
    c = a & b
    d = a | c
    e = expr.conj(a, b, c, d)

    print("\nd = ", e, "\n")


def test_bitarray():
    a = expr.Bool(tensor([True, True]))
    b = expr.Bool("a", shape=(2,))
    c = a & b
    d = a | c
    e = expr.conj(a, b, c, d)

    print("\nd = ", e, "\n")


def test_arith():
    a = expr.Real(2.1)
    b = expr.Real("a")
    c = a + b
    d = a / c
    e = expr.mul(a, b, c, d)

    print("\nd = ", e, "\n")


def test_predicate():
    a = expr.Real(tensor([2.1, 3.1]))
    b = expr.Real("a", shape=(2,))
    c = a <= b
    d = a == b

    print("\nc iff", c)
    print("d iff", d, "\n")


def test_ite():
    a = expr.Real(torch.tensor([2.1, 3.1]))
    b = expr.Real("b", shape=(2,))
    c = expr.Real("c")

    d = expr.ite(c, a, b)

    print("\nd =", d)
    print("type(d) =", type(d))


def test_dtype_comparison():
    a = expr.Real("a")
    b = expr.Real("b", shape=[2])
    c = expr.Bool("c", shape=[2])
    assert type(a.dtype) == type(b.dtype)
    assert type(b.dtype) != type(c.dtype)


def test_matmul():
    a = expr.Real(tensor([[2.1, 3.1], [4.1, 5.1]]))
    b = expr.Real("b", shape=(2, 2))
    c = a @ b
    print("\nc =", c)

    d = expr.Real("d", shape=(2,))
    e = a @ d
    print("e =", e)


def test_argmax():
    a = expr.Real("a", shape=(2,))
    b = expr.argmax(a)
    print("\nb =", b)
