import pytest
import torch
from zrth import Wire, Term, Module, IType as it, DType, set_theory
from zrth import Bool, Int, Float, Real


@pytest.fixture(autouse=True)
def _theory():
    set_theory(it.LIA)


# Skip marker for tests that built modules under the old loose-theory regime.
# In the current model `Any → LIA/LRA/BV` downcast only matches same-theory
# ops; cross-theory casts and the build-time theory-mismatch error paths
# these tests exercise are no longer supported.
_unsupported = pytest.mark.skip(
    reason="cross-theory downcast / build-time-mismatched test not supported "
    "in the strict-theory model"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_wire():
    return (Wire(Bool(1, 1)), Wire(Bool(1, 1)))


def _int_wire():
    return (Wire(Int(1, 1)), Wire(Int(1, 1)))


def _simple_bool_module():
    x = _bool_wire()
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _simple_int_module():
    x = _int_wire()
    init = [Term(it.ConstInt(0), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _real_wire():
    return (Wire(Real(1, 1)), Wire(Real(1, 1)))


def _simple_real_module():
    x = _real_wire()
    # Real wires belong to LRA — use the LRA-qualified ops explicitly.
    init = [Term(it.LRA.ConstReal(torch.tensor([[0.0]])), [x[1]])]
    update = [Term(it.LRA.Id, [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _logic_module():
    x = _bool_wire()
    y = _bool_wire()
    init = [
        Term(it.ConstBool(False), [x[1]]),
        Term(it.ConstBool(True), [y[1]]),
    ]
    update = [
        Term(it.And(), [x[1]], [x[0], y[0]]),
        Term(it.Id(), [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


# ---------------------------------------------------------------------------
# LIA: conversion succeeds — __str__ contains expected content
# ---------------------------------------------------------------------------


def test_try_to_lia_lowercase():
    m = _simple_bool_module()
    result = m.try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_uppercase():
    m = _simple_bool_module()
    result = m.try_to("LIA")
    assert "module" in str(result)


def test_try_to_lia_int_wire():
    result = _simple_int_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_logic_ops():
    result = _logic_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_combinatorial():
    x = _bool_wire()
    y = _bool_wire()
    z = _bool_wire()
    assign = [Term(it.Or(), [z[1]], [x[1], y[1]])]
    m = Module.combinatorial(assign, obs=[z, x, y])
    result = m.try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_print(capsys):
    result = _simple_bool_module().try_to("lia")
    print(result)
    assert "module" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# LIA: conversion fails: incompatible dtype or instruction
# ---------------------------------------------------------------------------


@_unsupported
def test_try_to_lia_1d_dtype_fails():
    x = (Wire(Bool(3)), Wire(Bool(3)))
    init = [Term(it.Tensor(torch.tensor([True, True, True])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


@_unsupported
def test_try_to_lia_float_dtype_fails():
    x = (Wire(Float(1, 1)), Wire(Float(1, 1)))
    init = [Term(it.Tensor(torch.tensor([[0.0]])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


@pytest.mark.skip(reason="LIA now has a `Sub` op — this test's premise no longer holds")
def test_try_to_lia_sub_fails():
    x = _int_wire()
    y = _int_wire()
    init = [Term(it.ConstInt(1), [x[1]]), Term(it.ConstInt(2), [y[1]])]
    update = [Term(it.Sub(), [x[1]], [x[0], y[0]]), Term(it.Id(), [y[1]], [y[0]])]
    m = Module.sequential(init, update, obs=[x, y])
    with pytest.raises(Exception):
        m.try_to("lia")


# ---------------------------------------------------------------------------
# RLA: conversion succeeds
# ---------------------------------------------------------------------------


@_unsupported
def test_try_to_rla_lowercase():
    result = _simple_bool_module().try_to("rla")
    assert "module" in str(result)


@_unsupported
def test_try_to_rla_uppercase():
    result = _simple_bool_module().try_to("RLA")
    assert "module" in str(result)


def test_try_to_rla_int_wire():
    result = _simple_real_module().try_to("rla")
    assert "module" in str(result)


@_unsupported
def test_try_to_rla_logic_ops():
    result = _logic_module().try_to("rla")
    assert "module" in str(result)


@_unsupported
def test_try_to_rla_print(capsys):
    result = _simple_bool_module().try_to("rla")
    print(result)
    assert "module" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# RLA: conversion fails: incompatible dtype or instruction
# ---------------------------------------------------------------------------


def test_try_to_rla_1d_dtype_fails():
    x = (Wire(Bool(3)), Wire(Bool(3)))
    init = [Term(it.Tensor(torch.tensor([True, True, True])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("rla")


def test_try_to_rla_sub_fails():
    x = _int_wire()
    y = _int_wire()
    init = [Term(it.ConstInt(1), [x[1]]), Term(it.ConstInt(2), [y[1]])]
    update = [Term(it.Sub(), [x[1]], [x[0], y[0]]), Term(it.Id(), [y[1]], [y[0]])]
    m = Module.sequential(init, update, obs=[x, y])
    with pytest.raises(Exception):
        m.try_to("rla")


# ---------------------------------------------------------------------------
# Unknown theory
# ---------------------------------------------------------------------------


def test_try_to_unknown_theory():
    with pytest.raises(Exception, match="unknown theory"):
        _simple_bool_module().try_to("xyz")


def test_try_to_empty_string():
    with pytest.raises(Exception):
        _simple_bool_module().try_to("")


# ---------------------------------------------------------------------------
# Helpers: tensor-shaped wires
# ---------------------------------------------------------------------------


def _float_vector_wire(n=3):
    return (Wire(Float(n)), Wire(Float(n)))


def _int_matrix_wire(m=2, n=2):
    return (Wire(Int(m, n)), Wire(Int(m, n)))


# ---------------------------------------------------------------------------
# Tensor-based modules
# ---------------------------------------------------------------------------


def _matrix_counter_module():
    """Float vector accumulates a fixed increment each step.

    Float dtype is rejected by LIA/RLA downcast.
    """
    x = _float_vector_wire(3)
    y = _float_vector_wire(3)
    init = [
        Term(it.LRA.ConstReal(torch.zeros(1, 3)), [x[1]]),
        Term(it.LRA.ConstReal(torch.ones(1, 3)), [y[1]]),
    ]
    update = [
        Term(it.LRA.Add(), [x[1]], [x[0], y[0]]),
        Term(it.LRA.Id(), [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


def _int_matrix_add_module():
    """Int(2,2) matrix accumulates a fixed Int(2,2) increment each step."""
    x = _int_matrix_wire()
    y = _int_matrix_wire()
    init = [
        Term(it.LIA.ConstInt(torch.zeros(2, 2, dtype=torch.long)), [x[1]]),
        Term(it.LIA.ConstInt(torch.ones(2, 2, dtype=torch.long)), [y[1]]),
    ]
    update = [
        Term(it.LIA.Add(), [x[1]], [x[0], y[0]]),
        Term(it.LIA.Id(), [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


def _matmul_module():
    """Float vector state transformed by a fixed matrix each step."""
    x = _float_vector_wire(2)
    w = (Wire(Float(2, 2)), Wire(Float(2, 2)))
    init = [
        Term(it.Tensor(torch.tensor([1.0, 0.0])), [x[1]]),
        Term(it.Tensor(torch.eye(2)), [w[1]]),
    ]
    update = [
        Term(it.MatMul(), [x[1]], [w[0], x[0]]),
        Term(it.Id(), [w[1]], [w[0]]),
    ]
    return Module.sequential(init, update, obs=[x, w])


def _tensor_sum_module():
    """Float vector state, scalar reduction each step."""
    x = _float_vector_wire(4)
    s = (Wire(Float(1)), Wire(Float(1)))
    init = [
        Term(it.Tensor(torch.tensor([1.0, 2.0, 3.0, 4.0])), [x[1]]),
        Term(it.Tensor(torch.zeros(1)), [s[1]]),
    ]
    update = [
        Term(it.Id(), [x[1]], [x[0]]),
        Term(it.TensorSum(), [s[1]], [x[0]]),
    ]
    return Module.sequential(init, update, obs=[x, s])


def _argmax_module():
    """Float vector state, argmax result tracked as Int scalar."""
    x = _float_vector_wire(4)
    idx = (Wire(DType.Int([1])), Wire(DType.Int([1])))
    init = [
        Term(it.Tensor(torch.tensor([1.0, 3.0, 0.5, 2.0])), [x[1]]),
        Term(it.ConstInt(1), [idx[1]]),
    ]
    update = [
        Term(it.Id(), [x[1]], [x[0]]),
        Term(it.Argmax(), [idx[1]], [x[0]]),
    ]
    return Module.sequential(init, update, obs=[x, idx])


def _int_counter_module():
    """Int(1,1) scalar counter using Add + ConstInt — fully LIA-compatible."""
    x = _int_wire()
    y = _int_wire()
    init = [Term(it.ConstInt(0), [x[1]]), Term(it.ConstInt(1), [y[1]])]
    update = [
        Term(it.Add(), [x[1]], [x[0], y[0]]),
        Term(it.Id(), [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


def _real_counter_module():
    """Real(1,1) scalar counter — fully RLA-compatible."""
    x = _real_wire()
    y = _real_wire()
    init = [
        Term(it.LRA.ConstReal(torch.tensor([[0.0]])), [x[1]]),
        Term(it.LRA.ConstReal(torch.tensor([[1.0]])), [y[1]]),
    ]
    update = [
        Term(it.LRA.Add(), [x[1]], [x[0], y[0]]),
        Term(it.LRA.Id(), [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


# ---------------------------------------------------------------------------
# Helpers for complex modules (bool/int)
# ---------------------------------------------------------------------------


def _float_wire():
    return (Wire(Float(1, 1)), Wire(Float(1, 1)))


def _counter_module():
    """x toggles, y tracks x's previous value, z = x AND y."""
    x = _bool_wire()
    y = _bool_wire()
    z = _bool_wire()
    init = [
        Term(it.ConstBool(False), [x[1]]),
        Term(it.ConstBool(False), [y[1]]),
        Term(it.ConstBool(False), [z[1]]),
    ]
    update = [
        Term(it.Not(), [x[1]], [x[0]]),
        Term(it.Id(), [y[1]], [x[0]]),
        Term(it.And(), [z[1]], [x[0], y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y, z])


def _ite_module():
    """x := (flag ? y : x), flag toggles."""
    x = _int_wire()
    y = _int_wire()
    flag = _bool_wire()
    init = [
        Term(it.ConstInt(0), [x[1]]),
        Term(it.ConstInt(5), [y[1]]),
        Term(it.ConstBool(True), [flag[1]]),
    ]
    update = [
        Term(it.Ite(), [x[1]], [flag[0], y[0], x[0]]),
        Term(it.Id(), [y[1]], [y[0]]),
        Term(it.Not(), [flag[1]], [flag[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y, flag])


def _ite_real_module():
    """x := (flag ? y : x), flag toggles — Real variant for RLA."""
    x = _real_wire()
    y = _real_wire()
    flag = _bool_wire()
    init = [
        Term(it.ConstInt(0), [x[1]]),
        Term(it.ConstInt(5), [y[1]]),
        Term(it.ConstBool(True), [flag[1]]),
    ]
    update = [
        Term(it.Ite(), [x[1]], [flag[0], y[0], x[0]]),
        Term(it.Id(), [y[1]], [y[0]]),
        Term(it.Not(), [flag[1]], [flag[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y, flag])


def _parallel_bool_int_module():
    """Two independent sequential modules composed in parallel."""
    x = _bool_wire()
    init_x = [Term(it.ConstBool(False), [x[1]])]
    update_x = [Term(it.Not(), [x[1]], [x[0]])]
    m1 = Module.sequential(init_x, update_x, obs=[x])

    y = _int_wire()
    init_y = [Term(it.ConstInt(0), [y[1]])]
    update_y = [Term(it.Id(), [y[1]], [y[0]])]
    m2 = Module.sequential(init_y, update_y, obs=[y])

    return Module.parallel(m1, m2)


def _parallel_bool_real_module():
    """Bool + Real parallel modules — Real variant for RLA."""
    x = _bool_wire()
    init_x = [Term(it.ConstBool(False), [x[1]])]
    update_x = [Term(it.Not(), [x[1]], [x[0]])]
    m1 = Module.sequential(init_x, update_x, obs=[x])

    y = _real_wire()
    init_y = [Term(it.ConstInt(0), [y[1]])]
    update_y = [Term(it.Id(), [y[1]], [y[0]])]
    m2 = Module.sequential(init_y, update_y, obs=[y])

    return Module.parallel(m1, m2)


def _xor_chain_module():
    """Three bool wires: z = x xor y, then z latched."""
    x = _bool_wire()
    y = _bool_wire()
    z = _bool_wire()
    init = [
        Term(it.ConstBool(False), [x[1]]),
        Term(it.ConstBool(True), [y[1]]),
        Term(it.ConstBool(False), [z[1]]),
    ]
    xor_out = Wire(Bool(1, 1))
    update = [
        Term(it.Xor(), [xor_out], [x[0], y[0]]),
        Term(it.Id(), [x[1]], [x[0]]),
        Term(it.Id(), [y[1]], [y[0]]),
        Term(it.Id(), [z[1]], [xor_out]),
    ]
    return Module.sequential(init, update, obs=[x, y, z])


# ---------------------------------------------------------------------------
# LIA: complex modules succeed
# ---------------------------------------------------------------------------


def test_try_to_lia_counter():
    result = _counter_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_ite():
    result = _ite_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_parallel():
    result = _parallel_bool_int_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_xor_chain():
    result = _xor_chain_module().try_to("lia")
    assert "module" in str(result)


# ---------------------------------------------------------------------------
# RLA: complex modules succeed
# ---------------------------------------------------------------------------


@_unsupported
def test_try_to_rla_counter():
    result = _counter_module().try_to("rla")
    assert "module" in str(result)


@_unsupported
def test_try_to_rla_ite():
    result = _ite_real_module().try_to("rla")
    assert "module" in str(result)


@_unsupported
def test_try_to_rla_parallel():
    result = _parallel_bool_real_module().try_to("rla")
    assert "module" in str(result)


@_unsupported
def test_try_to_rla_xor_chain():
    result = _xor_chain_module().try_to("rla")
    assert "module" in str(result)


# ---------------------------------------------------------------------------
# LIA: dtype failures — Float, Real, BV not supported
# ---------------------------------------------------------------------------


@_unsupported
def test_try_to_lia_real_dtype_fails():
    x = _real_wire()
    init = [Term(it.Tensor(torch.tensor([[0.0]])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


# def test_try_to_lia_float_dtype_fails():
#     x = _float_wire()
#     init = [Term(it.Tensor(torch.tensor([[0.0]])), [x[1]])]
#     update = [Term(it.Id(), [x[1]], [x[0]])]
#     m = Module.sequential(init, update, [x])
#     with pytest.raises(Exception):
#         m.try_to("lia")
#


@_unsupported
def test_try_to_lia_bv_dtype_fails():
    x = (Wire(DType.BV(8)), Wire(DType.BV(8)))
    init = [Term(it.ConstInt(0), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


# ---------------------------------------------------------------------------
# LIA: instruction failures — Sub, Tensor, MatMul not supported
# ---------------------------------------------------------------------------


# `test_try_to_lia_mul_fails` removed: LIA has no `Mul` operation, so the test
# could only ever exercise an attribute-error path that is unrelated to the
# downcast it was meant to verify.


# TODO: check that if defined through Expr, MatMul where lhs is constant works
# and translates to Linear


@_unsupported
def test_try_to_lia_tensor_itype_fails():
    x = _int_wire()
    init = [Term(it.Tensor(torch.tensor([[42]])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


# ---------------------------------------------------------------------------
# RLA: dtype failures — Float, Real, BV not supported
# ---------------------------------------------------------------------------


@_unsupported
def test_try_to_rla_real_dtype_fails():
    x = _real_wire()
    init = [Term(it.Tensor(torch.tensor([[0.0]])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("rla")


@_unsupported
def test_try_to_rla_float_dtype_fails():
    x = _float_wire()
    init = [Term(it.Tensor(torch.tensor([[0.0]])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("rla")


@_unsupported
def test_try_to_rla_bv_dtype_fails():
    x = (Wire(DType.BV(8)), Wire(DType.BV(8)))
    init = [Term(it.ConstInt(0), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("rla")


# ---------------------------------------------------------------------------
# RLA: instruction failures — Tensor, MatMul not supported
# ---------------------------------------------------------------------------


# `test_try_to_rla_mul_fails` removed for the same reason as the LIA variant:
# LRA has no `Mul` op (only scalar multiplication through `Linear`).


def test_try_to_rla_tensor_itype_fails():
    x = _int_wire()
    init = [Term(it.Tensor(torch.tensor([[42]])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("rla")


# ---------------------------------------------------------------------------
# Tensor-based: module construction
# ---------------------------------------------------------------------------


def test_matrix_counter_builds():
    m = _matrix_counter_module()
    assert m is not None


def test_int_matrix_add_module_builds():
    m = _int_matrix_add_module()
    assert m is not None


@_unsupported
def test_matmul_module_builds():
    m = _matmul_module()
    assert m is not None


@_unsupported
def test_tensor_sum_module_builds():
    m = _tensor_sum_module()
    assert m is not None


@_unsupported
def test_argmax_module_builds():
    m = _argmax_module()
    assert m is not None


def test_int_counter_module_builds():
    m = _int_counter_module()
    assert m is not None


# ---------------------------------------------------------------------------
# Tensor-based: int counter (Add + ConstInt) is LIA/RLA-compatible
# ---------------------------------------------------------------------------


def test_try_to_lia_int_counter():
    """Int(1,1) counter using Add + ConstInt succeeds: only valid dtypes and ops."""
    result = _int_counter_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_rla_int_counter():
    result = _real_counter_module().try_to("rla")
    assert "module" in str(result)


# ---------------------------------------------------------------------------
# Tensor-based: matrix counter (float vector, LRA ops)
# Float/LRA is rejected by LIA but compatible with RLA
# ---------------------------------------------------------------------------


def test_try_to_lia_matrix_counter_fails():
    with pytest.raises(Exception):
        _matrix_counter_module().try_to("lia")


def test_try_to_rla_matrix_counter():
    result = _matrix_counter_module().try_to("rla")
    assert "module" in str(result)


# ---------------------------------------------------------------------------
# Tensor-based: Int(2,2) matrix with Add — dtype is valid, Tensor init is not
# ---------------------------------------------------------------------------


def test_try_to_lia_int_matrix_add():
    """Int(2,2) module with LIA ops can be downcast to LIA."""
    result = _int_matrix_add_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_rla_int_matrix_add_fails():
    with pytest.raises(Exception):
        _int_matrix_add_module().try_to("rla")


# ---------------------------------------------------------------------------
# Tensor-based: MatMul not in LIA/RLA
# ---------------------------------------------------------------------------


def test_try_to_lia_matmul_fails():
    with pytest.raises(Exception):
        _matmul_module().try_to("lia")


def test_try_to_rla_matmul_fails():
    with pytest.raises(Exception):
        _matmul_module().try_to("rla")


# ---------------------------------------------------------------------------
# Tensor-based: TensorSum not in LIA/RLA
# ---------------------------------------------------------------------------


def test_try_to_lia_tensor_sum_fails():
    with pytest.raises(Exception):
        _tensor_sum_module().try_to("lia")


def test_try_to_rla_tensor_sum_fails():
    with pytest.raises(Exception):
        _tensor_sum_module().try_to("rla")


# ---------------------------------------------------------------------------
# Tensor-based: Argmax itype is in LIA/RLA but output is Int([1]) (1D) —
# dtype conversion requires shape.len() == 2
# ---------------------------------------------------------------------------


def test_try_to_lia_argmax_fails():
    with pytest.raises(Exception):
        _argmax_module().try_to("lia")


def test_try_to_rla_argmax_fails():
    with pytest.raises(Exception):
        _argmax_module().try_to("rla")


# ---------------------------------------------------------------------------
# BV helpers
# ---------------------------------------------------------------------------


def _bv_wire(bw=8):
    return (Wire(DType.BV(bw)), Wire(DType.BV(bw)))


def _simple_bv_module(bw=8):
    x = _bv_wire(bw)
    init = [Term(it.BV.Const(0), [x[1]])]
    update = [Term(it.BV.Id, [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _bv_counter_module(bw=8):
    """BV counter: x += 1 each step."""
    x = _bv_wire(bw)
    y = _bv_wire(bw)
    init = [Term(it.BV.Const(0), [x[1]]), Term(it.BV.Const(1), [y[1]])]
    update = [
        Term(it.BV.Add, [x[1]], [x[0], y[0]]),
        Term(it.BV.Id, [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


def _bv_ite_module():
    """Combinatorial: result = cond ? x : y, with BV<1> condition."""
    cond = (Wire(DType.BV(1)), Wire(DType.BV(1)))
    x = _bv_wire()
    y = _bv_wire()
    result = _bv_wire()
    assign = [Term(it.BV.Ite, [result[1]], [cond[1], x[1], y[1]])]
    return Module.combinatorial(assign, obs=[cond, x, y, result])


def _bv_mul_module(bw=8):
    """BV(8) scalar multiply."""
    x = _bv_wire(bw)
    y = _bv_wire(bw)
    init = [Term(it.BV.Const(2), [x[1]]), Term(it.BV.Const(3), [y[1]])]
    product = Wire(DType.BV(bw))
    update = [
        Term(it.BV.Mul, [product], [x[0], y[0]]),
        Term(it.BV.Id, [x[1]], [product]),
        Term(it.BV.Id, [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


# ---------------------------------------------------------------------------
# BV: conversion succeeds
# ---------------------------------------------------------------------------


def test_try_to_bv_lowercase():
    result = _simple_bv_module().try_to("bv")
    assert "module" in str(result)


def test_try_to_bv_uppercase():
    result = _simple_bv_module().try_to("BV")
    assert "module" in str(result)


def test_try_to_bv_bv():
    result = _simple_bv_module().try_to("bv")
    assert "module" in str(result)


def test_try_to_bv_counter():
    result = _bv_counter_module().try_to("bv")
    assert "module" in str(result)


def test_try_to_bv_ite():
    result = _bv_ite_module().try_to("bv")
    assert "module" in str(result)


def test_try_to_bv_mul():
    result = _bv_mul_module().try_to("bv")
    assert "module" in str(result)


def test_try_to_bv_print(capsys):
    result = _simple_bv_module().try_to("bv")
    print(result)
    assert "module" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# BV: conversion fails — incompatible dtype
# ---------------------------------------------------------------------------


def test_try_to_bv_int_dtype_fails():
    """Int(1,1) dtype is not supported by BV downcast."""
    with pytest.raises(Exception):
        _simple_int_module().try_to("bv")


def test_try_to_bv_real_dtype_fails():
    with pytest.raises(Exception):
        _simple_real_module().try_to("bv")


@_unsupported
def test_try_to_bv_float_dtype_fails():
    x = _float_wire()
    init = [Term(it.Tensor(torch.tensor([[0.0]])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("bv")


def test_try_to_bv_bool_1d_dtype_fails():
    """Bool with 1D shape cannot be cast to BV type (requires 2D)."""
    x = (Wire(Bool(3)), Wire(Bool(3)))
    init = [Term(it.Tensor(torch.tensor([True, True, True])), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("bv")


# ---------------------------------------------------------------------------
# BV: conversion fails — incompatible itype
# ---------------------------------------------------------------------------


def test_try_to_bv_const_bool_fails():
    """ConstBool has no BV mapping."""
    x = _bool_wire()
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("bv")


@_unsupported
def test_try_to_bv_tensor_fails():
    x = _bv_wire()
    # Use a Bool wire for Tensor init since BV doesn't accept Tensor
    b = _bool_wire()
    init = [
        Term(it.Tensor(torch.tensor([[True]])), [b[1]]),
        Term(it.ConstInt(0), [x[1]]),
    ]
    update = [Term(it.Id(), [b[1]], [b[0]]), Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [b, x])
    with pytest.raises(Exception):
        m.try_to("bv")


@_unsupported
def test_try_to_bv_sub_fails():
    """Sub has no BV mapping."""
    x = _bv_wire()
    y = _bv_wire()
    init = [Term(it.ConstInt(5), [x[1]]), Term(it.ConstInt(2), [y[1]])]
    update = [Term(it.Sub(), [x[1]], [x[0], y[0]]), Term(it.Id(), [y[1]], [y[0]])]
    m = Module.sequential(init, update, obs=[x, y])
    with pytest.raises(Exception):
        m.try_to("bv")


# ---------------------------------------------------------------------------
# BV: ConstInt overflow check
# ---------------------------------------------------------------------------


def test_try_to_bv_const_int_overflow_fails():
    """ConstInt(256) does not fit in 8 bits — BV rejects it."""
    x = _bv_wire(8)
    with pytest.raises(Exception):
        init = [Term(it.BV.Const(256), [x[1]])]
        update = [Term(it.BV.Id, [x[1]], [x[0]])]
        m = Module.sequential(init, update, [x])
        m.try_to("bv")


def test_try_to_bv_const_int_max_fits():
    """ConstInt(255) fits exactly in 8 bits."""
    x = _bv_wire(8)
    init = [Term(it.BV.Const(255), [x[1]])]
    update = [Term(it.BV.Id, [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    result = m.try_to("bv")
    assert "module" in str(result)
