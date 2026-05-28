import pytest
from torch import IntTensor, tensor
import zrth.expr as expr
from zrth import IType, set_theory


@pytest.fixture(autouse=True)
def _theory():
    # Tests below build Real/Bool expressions — LRA covers both fragments.
    set_theory(IType.LRA)


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
    # LRA has Add / Sub / Linear (and Min/Max/ReLU/Argmax) but no general
    # `Mul` or `Div`, so the test only exercises the supported subset.
    a = expr.Real(2.1)
    b = expr.Real("a")
    c = a + b
    d = c - a

    print("\nd = ", d, "\n")


def test_predicate():
    a = expr.Real(tensor([2.1, 3.1]))
    b = expr.Real("a", shape=(2,))
    c = a <= b
    d = a == b

    print("\nc iff", c)
    print("d iff", d, "\n")


def test_ite():
    a = expr.Real(tensor([2.1, 3.1]))
    b = expr.Real("b", shape=(2,))
    c = expr.Bool("c")

    d = expr.ite(c, a, b)

    print("\nd =", d)
    print("type(d) =", type(d))


def test_dtype_comparison():
    a = expr.Real("a")
    b = expr.Real("b", shape=[2])
    c = expr.Bool("c", shape=[2])
    # DType is a single Python class — distinguish via the predicates.
    assert a.dtype.is_real() and b.dtype.is_real()
    assert c.dtype.is_bool()
    assert not b.dtype.is_bool()


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
