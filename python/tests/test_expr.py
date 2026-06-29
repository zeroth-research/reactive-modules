import pytest
from torch import IntTensor, tensor
import zrth.expr as expr
from zrth import LRA, LIA, Sort


def test_terminal():
    a = expr.Real(0.9, theory=LRA)
    b = expr.Real("b", theory=LRA)


def test_boolean():
    a = expr.Bool(True, theory=LIA)
    b = expr.Bool("a", theory=LIA)
    c = a & b
    d = a | c
    e = expr.conj(a, b, c, d)

    print("\nd = ", e, "\n")


def test_bitarray():
    a = expr.Bool(tensor([True, True]), theory=LIA)
    b = expr.Bool("a", theory=LIA, shape=(2,))
    c = a & b
    d = a | c
    e = expr.conj(a, b, c, d)

    print("\nd = ", e, "\n")


def test_arith():
    # LRA has Add / Sub / Linear (and Min/Max/ReLU/Argmax) but no general
    # `Mul` or `Div`, so the test only exercises the supported subset.
    a = expr.Real(2.1, theory=LRA)
    b = expr.Real("a", theory=LRA)
    c = a + b
    d = c - a

    print("\nd = ", d, "\n")


def test_predicate():
    a = expr.Real(tensor([2.1, 3.1]), theory=LRA)
    b = expr.Real("a", theory=LRA, shape=(2,))
    c = a <= b
    d = a == b

    print("\nc iff", c)
    print("d iff", d, "\n")


def test_ite():
    a = expr.Real(tensor([2.1, 3.1]), theory=LRA)
    b = expr.Real("b", theory=LRA, shape=(2,))
    c = expr.Bool("c", theory=LIA)

    d = expr.ite(c, a, b)

    print("\nd =", d)
    print("type(d) =", type(d))


def test_dtype_comparison():
    a = expr.Real("a", theory=LRA)
    b = expr.Real("b", theory=LRA, shape=[2])
    c = expr.Bool("c", theory=LIA, shape=[2])
    # Sorts expose no predicate methods — distinguish via isinstance on the variant.
    assert isinstance(a.dtype, Sort.Real) and isinstance(b.dtype, Sort.Real)
    assert isinstance(c.dtype, Sort.Bool)
    assert not isinstance(b.dtype, Sort.Bool)


def test_matmul():
    a = expr.Real(tensor([[2.1, 3.1], [4.1, 5.1]]), theory=LRA)
    b = expr.Real("b", theory=LRA, shape=(2, 2))
    c = a @ b
    print("\nc =", c)

    d = expr.Real("d", theory=LRA, shape=(2, 1))  # column-major: 2 features, 1 batch item
    e = a @ d
    print("e =", e)


def test_argmax():
    a = expr.Real("a", theory=LRA, shape=(2,))
    b = expr.argmax(a)
    print("\nb =", b)
