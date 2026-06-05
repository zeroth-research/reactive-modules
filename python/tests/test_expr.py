import pytest
from torch import IntTensor, tensor
import zrth.expr as expr
from zrth import IType


def test_terminal():
    a = expr.Real(0.9, theory=IType.LRA)
    b = expr.Real("b", theory=IType.LRA)


def test_boolean():
    a = expr.Bool(True, theory=IType.LIA)
    b = expr.Bool("a", theory=IType.LIA)
    c = a & b
    d = a | c
    e = expr.conj(a, b, c, d)

    print("\nd = ", e, "\n")


def test_bitarray():
    a = expr.Bool(tensor([True, True]), theory=IType.LIA)
    b = expr.Bool("a", theory=IType.LIA, shape=(2,))
    c = a & b
    d = a | c
    e = expr.conj(a, b, c, d)

    print("\nd = ", e, "\n")


def test_arith():
    # LRA has Add / Sub / Linear (and Min/Max/ReLU/Argmax) but no general
    # `Mul` or `Div`, so the test only exercises the supported subset.
    a = expr.Real(2.1, theory=IType.LRA)
    b = expr.Real("a", theory=IType.LRA)
    c = a + b
    d = c - a

    print("\nd = ", d, "\n")


def test_predicate():
    a = expr.Real(tensor([2.1, 3.1]), theory=IType.LRA)
    b = expr.Real("a", theory=IType.LRA, shape=(2,))
    c = a <= b
    d = a == b

    print("\nc iff", c)
    print("d iff", d, "\n")


def test_ite():
    a = expr.Real(tensor([2.1, 3.1]), theory=IType.LRA)
    b = expr.Real("b", theory=IType.LRA, shape=(2,))
    c = expr.Bool("c", theory=IType.LIA)

    d = expr.ite(c, a, b)

    print("\nd =", d)
    print("type(d) =", type(d))


def test_dtype_comparison():
    a = expr.Real("a", theory=IType.LRA)
    b = expr.Real("b", theory=IType.LRA, shape=[2])
    c = expr.Bool("c", theory=IType.LIA, shape=[2])
    # DType is a single Python class — distinguish via the predicates.
    assert a.dtype.is_real() and b.dtype.is_real()
    assert c.dtype.is_bool()
    assert not b.dtype.is_bool()


def test_matmul():
    a = expr.Real(tensor([[2.1, 3.1], [4.1, 5.1]]), theory=IType.LRA)
    b = expr.Real("b", theory=IType.LRA, shape=(2, 2))
    c = a @ b
    print("\nc =", c)

    d = expr.Real("d", theory=IType.LRA, shape=(2, 1))  # column-major: 2 features, 1 batch item
    e = a @ d
    print("e =", e)


def test_argmax():
    a = expr.Real("a", theory=IType.LRA, shape=(2,))
    b = expr.argmax(a)
    print("\nb =", b)
